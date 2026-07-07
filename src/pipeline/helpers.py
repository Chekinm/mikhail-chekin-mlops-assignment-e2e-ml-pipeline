"""Helper functions for the evaluate_agent Airflow DAG.
"""

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def write_manifest(run_config: dict, run_dir: Path) -> dict:
    """Write manifest.json pointing to the important run artifacts."""
    manifest = {
        "run_id": run_config["run_id"],
        "config": "config.json",
        "predictions": "run-agent/preds.json",
        "trajectories": "run-agent/",
        "eval_logs": "run-eval/logs/",
        "eval_summary": "run-eval/summary.json",
        "metrics": "metrics.json",
        "artifact_uri": str(run_dir),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def log_mlflow_run(run_config: dict, metrics: dict, artifact_uri: str) -> None:
    """Log parameters, metrics, and key artifacts to MLflow."""
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
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