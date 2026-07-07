"""Configurable Airflow DAG for running mini-swe-agent and evaluating results.

Pipeline: prepare_run -> run_agent -> run_eval -> upload_artifacts -> summarize_and_log
Agent and eval steps run in isolated Docker containers via DockerOperator.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline.helpers import (  # noqa: E402
    RUNS_DIR,
    build_run_config,
    collect_metrics,
    log_mlflow_run,
    prepare_run_dir,
    upload_run_to_s3,
    write_manifest,
)

DOCKER_IMAGE = "mlops-assignment:latest"
CONTAINER_WORKDIR = "/mlops-assignment"
CONTAINER_RUNS_DIR = f"{CONTAINER_WORKDIR}/runs"

# Host-side paths for DockerOperator sibling containers.
# When Airflow itself runs in a container, Mount sources must be HOST paths
# (containers are created by the host Docker daemon), so we take them from
# environment variables. Fallback to local paths for standalone mode.
HOST_PROJECT_DIR = os.environ.get("HOST_PROJECT_DIR", str(PROJECT_ROOT))
HOST_RUNS_DIR = f"{HOST_PROJECT_DIR}/runs"
HOST_HF_CACHE_DIR = os.environ.get(
    "HOST_HF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface")
)


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

    @task(retries=1, retry_delay=timedelta(minutes=1))
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
    dataset_tpl = (
        "{{ 'princeton-nlp/SWE-bench_Verified' "
        "if ti.xcom_pull(task_ids='prepare_run')['subset'] == 'verified' "
        "else 'princeton-nlp/SWE-bench_Lite' }}"
    )

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
                source=HOST_RUNS_DIR,
                target=CONTAINER_RUNS_DIR,
                type="bind",
            ),
            Mount(
                source=HOST_HF_CACHE_DIR,
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
        retries=1,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(hours=2),
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
                f"--dataset_name {dataset_tpl} "
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

    @task(
        retries=2,
        retry_delay=timedelta(minutes=1),
        execution_timeout=timedelta(minutes=10),
    )
    def upload_artifacts(run_config: dict) -> str:
        """Upload the run directory to S3-compatible object storage (MinIO)."""
        run_dir = RUNS_DIR / run_config["run_id"]

        bucket = os.environ.get("S3_BUCKET", "mlops-runs")
        endpoint_url = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")

        return upload_run_to_s3(run_dir, bucket, endpoint_url)

    @task(
        retries=2,
        retry_delay=timedelta(minutes=1),
        execution_timeout=timedelta(minutes=10),
    )
    def summarize_and_log(run_config: dict, artifact_s3_uri: str) -> dict:
        run_dir = RUNS_DIR / run_config["run_id"]
        eval_dir = run_dir / "run-eval"

        metrics = collect_metrics(eval_dir)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        write_manifest(run_config, run_dir, artifact_s3_uri)
        log_mlflow_run(run_config, metrics, str(run_dir), artifact_s3_uri)

        return {"run_id": run_config["run_id"], "metrics": metrics}

    # DAG wiring
    config = prepare_run()
    s3_uri = upload_artifacts(config)
    config >> run_agent >> run_eval >> s3_uri
    summarize_and_log(config, s3_uri)


evaluate_agent()