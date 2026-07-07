"""Configurable Airflow DAG for running mini-swe-agent and evaluating results.

Pipeline: prepare_run -> run_agent -> run_eval -> summarize_and_log
Agent and eval steps run in isolated Docker containers via DockerOperator.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline.helpers import (
    RUNS_DIR,
    build_run_config,
    collect_metrics,
    log_mlflow_run,
    prepare_run_dir,
    write_manifest,
)

DOCKER_IMAGE = "mlops-assignment:latest"
CONTAINER_WORKDIR = "/mlops-assignment"
CONTAINER_RUNS_DIR = f"{CONTAINER_WORKDIR}/runs"


@dag(
    dag_id="evaluate_agent",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    params={
        "split": Param("test", type="string", description="SWE-bench split"),
        "subset": Param("verified", type="string", description="SWE-bench subset"),
        "workers": Param(5, type="integer", description="Number of parallel workers"),
        "model": Param(
            "nebius/moonshotai/Kimi-K2.6",
            type="string",
            description="LLM model identifier",
        ),
        "task_slice": Param(
            "0:3",
            type="string",
            description="Slice of SWE-bench tasks (e.g. '0:3')",
        ),
        "run_id": Param(
            "",
            type="string",
            description="Run ID (auto-generated if empty)",
        ),
        "cost_limit": Param(
            0,
            type="number",
            description="Cost limit per instance (0 = unlimited)",
        ),
    },
)
def evaluate_agent():

    @task
    def prepare_run(**context) -> dict:
        params = context["params"]
        run_config = build_run_config(params)
        prepare_run_dir(run_config)
        return run_config

    # Jinja templates pulling values from prepare_run's XCom
    run_id_tpl = "{{ ti.xcom_pull(task_ids='prepare_run')['run_id'] }}"
    subset_tpl = "{{ ti.xcom_pull(task_ids='prepare_run')['subset'] }}"
    split_tpl = "{{ ti.xcom_pull(task_ids='prepare_run')['split'] }}"
    model_tpl = "{{ ti.xcom_pull(task_ids='prepare_run')['model'] }}"
    slice_tpl = "{{ ti.xcom_pull(task_ids='prepare_run')['task_slice'] }}"
    workers_tpl = "{{ ti.xcom_pull(task_ids='prepare_run')['workers'] }}"

    common_docker_kwargs = dict(
        image=DOCKER_IMAGE,
        api_version="auto",
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
        mounts=[
            Mount(
                source="/var/run/docker.sock",
                target="/var/run/docker.sock",
                type="bind",
            ),
            Mount(
                source=str(RUNS_DIR),
                target=CONTAINER_RUNS_DIR,
                type="bind",
            ),
            # add this mount to use cached model in sub docker containers.
            Mount(
                source=str(Path.home() / ".cache" / "huggingface"),
                target="/root/.cache/huggingface",
                type="bind",
            ),
        ],
        mount_tmp_dir=False,
        environment={
            "NEBIUS_API_KEY": os.environ.get("NEBIUS_API_KEY", ""),
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "MSWEA_COST_TRACKING": "ignore_errors",
        },
        working_dir=CONTAINER_WORKDIR,
    )

    run_agent = DockerOperator(
        task_id="run_agent",
        command=[
            "mini-extra", "swebench",
            "--subset", subset_tpl,
            "--split", split_tpl,
            "--model", model_tpl,
            "--slice", slice_tpl,
            "--workers", workers_tpl,
            "-o", f"{CONTAINER_RUNS_DIR}/{run_id_tpl}/run-agent",
        ],
        **common_docker_kwargs,
    )

    run_eval = DockerOperator(
        task_id="run_eval",
        command=[
            "bash", "-c",
            (
                "python -m swebench.harness.run_evaluation "
                "--dataset_name princeton-nlp/SWE-bench_Verified "
                f"--predictions_path {CONTAINER_RUNS_DIR}/{run_id_tpl}/run-agent/preds.json "
                f"--max_workers {workers_tpl} "
                f"--run_id {run_id_tpl} "
                f"&& cp -r logs/run_evaluation/{run_id_tpl}/* "
                f"{CONTAINER_RUNS_DIR}/{run_id_tpl}/run-eval/logs/ "
                f"&& cp nebius__*.{run_id_tpl}.json "
                f"{CONTAINER_RUNS_DIR}/{run_id_tpl}/run-eval/summary.json"
            ),
        ],
        **common_docker_kwargs,
    )

    @task
    def summarize_and_log(run_config: dict) -> dict:
        run_dir = RUNS_DIR / run_config["run_id"]
        eval_dir = run_dir / "run-eval"

        metrics = collect_metrics(eval_dir)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        write_manifest(run_config, run_dir)
        log_mlflow_run(run_config, metrics, str(run_dir))

        return {"run_id": run_config["run_id"], "metrics": metrics}

    # DAG wiring
    config = prepare_run()
    config >> run_agent >> run_eval >> summarize_and_log(config)


evaluate_agent()