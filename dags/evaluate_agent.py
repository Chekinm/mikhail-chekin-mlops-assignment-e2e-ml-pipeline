"""Configurable Airflow DAG for running mini-swe-agent and evaluating results.

Pipeline: prepare_run -> run_agent -> run_eval -> summarize_and_log
"""

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from airflow.decorators import dag, task
from airflow.models.param import Param

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


def build_run_config(params: dict) -> dict:
    """Build a normalized run config dict from Airflow params."""
    run_id = params.get("run_id", "") or (
        f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    )
    return {
        "run_id": run_id,
        "split": params["split"],
        "subset": params["subset"],
        "workers": params["workers"],
        "model": params["model"],
        "task_slice": params["task_slice"],
        "cost_limit": params["cost_limit"],
        "created_at": datetime.utcnow().isoformat(),
    }


def prepare_run_dir(run_config: dict) -> Path:
    """Create the run directory tree and write config.json."""
    run_dir = RUNS_DIR / run_config["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run-agent").mkdir(exist_ok=True)
    (run_dir / "run-eval").mkdir(exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    return run_dir


def run_agent_batch(run_config: dict, run_dir: Path) -> Path:
    """Run mini-swe-agent batch and return the path to preds.json."""
    agent_dir = run_dir / "run-agent"

    cmd = [
        "uv", "run", "mini-extra", "swebench",
        "--subset", run_config["subset"],
        "--split", run_config["split"],
        "--model", run_config["model"],
        "--slice", run_config["task_slice"],
        "--workers", str(run_config["workers"]),
        "-o", str(agent_dir),
    ]

    if run_config.get("cost_limit", 0) > 0:
        cmd.extend(["--cost-limit", str(run_config["cost_limit"])])

    print(f"Running agent: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={**os.environ, "MSWEA_COST_TRACKING": "ignore_errors"},
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"run_agent failed with exit code {result.returncode}")

    preds_path = agent_dir / "preds.json"
    if not preds_path.exists():
        raise FileNotFoundError(f"preds.json not found at {preds_path}")

    return preds_path


def run_swebench_eval(run_config: dict, preds_path: Path, run_dir: Path) -> Path:
    """Run SWE-bench evaluation harness and return the eval output directory."""
    eval_dir = run_dir / "run-eval"

    dataset_name = (
        "princeton-nlp/SWE-bench_Verified"
        if run_config["subset"] == "verified"
        else "princeton-nlp/SWE-bench_Lite"
    )

    eval_run_id = run_config["run_id"]

    cmd = [
        "uv", "run", "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset_name,
        "--predictions_path", str(preds_path),
        "--max_workers", str(run_config["workers"]),
        "--run_id", eval_run_id,
    ]

    print(f"Running eval: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"run_eval failed with exit code {result.returncode}")

    # swebench writes logs to: logs/run_evaluation/<run_id>/<model_slug>/...
    eval_logs_src = PROJECT_ROOT / "logs" / "run_evaluation" / eval_run_id
    if eval_logs_src.exists():
        shutil.copytree(eval_logs_src, eval_dir / "logs", dirs_exist_ok=True)

    # swebench writes summary to: <model_slug>.<split>.json in CWD
    model_slug = run_config["model"].replace("/", "__")
    summary_name = f"{model_slug}.{run_config['split']}.json"
    summary_src = PROJECT_ROOT / summary_name
    if summary_src.exists():
        shutil.copy2(summary_src, eval_dir / "summary.json")

    return eval_dir


def collect_metrics(eval_dir: Path) -> dict:
    """Parse evaluation summary and return a metrics dict."""
    summary_path = eval_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    submitted = summary.get("submitted_instances", 0)
    return {
        "total_instances": summary.get("total_instances", 0),
        "submitted_instances": submitted,
        "completed_instances": summary.get("completed_instances", 0),
        "resolved_instances": summary.get("resolved_instances", 0),
        "unresolved_instances": summary.get("unresolved_instances", 0),
        "error_instances": summary.get("error_instances", 0),
        "resolve_rate": (
            summary.get("resolved_instances", 0) / submitted
            if submitted > 0
            else 0.0
        ),
    }


def log_mlflow_run(run_config: dict, metrics: dict, artifact_uri: str) -> None:
    """Log parameters, metrics, and key artifacts to MLflow."""
    import mlflow

    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment("swe-bench-evaluation")

    with mlflow.start_run(run_name=run_config["run_id"]):
        mlflow.log_params({
            "run_id": run_config["run_id"],
            "model": run_config["model"],
            "split": run_config["split"],
            "subset": run_config["subset"],
            "task_slice": run_config["task_slice"],
            "workers": run_config["workers"],
            "cost_limit": run_config["cost_limit"],
        })
        mlflow.log_metrics(metrics)

        run_dir = Path(artifact_uri)
        for artifact_name in ["config.json", "metrics.json", "manifest.json"]:
            artifact_path = run_dir / artifact_name
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))


# ---------------------------------------------------------------------------