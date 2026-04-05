from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, ContextManager

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.sklearn as mlflow_sklearn
    import mlflow.xgboost as mlflow_xgboost
except ImportError:  # pragma: no cover - depends on optional environment state
    mlflow = None
    mlflow_sklearn = None
    mlflow_xgboost = None


def start_run(run_name: str) -> ContextManager[object]:
    if mlflow is None:
        logger.warning("MLflow is not installed; training will continue without experiment logging.")
        return nullcontext()
    mlflow.set_experiment("Patient_Readmission_Models")
    return mlflow.start_run(run_name=run_name)


def log_param(key: str, value: Any) -> None:
    if mlflow is not None:
        mlflow.log_param(key, value)


def log_params(params: dict[str, Any]) -> None:
    if mlflow is not None:
        mlflow.log_params(params)


def log_metrics(metrics: dict[str, int | float]) -> None:
    if mlflow is not None:
        mlflow.log_metrics(metrics)


def log_artifact(path: str) -> None:
    if mlflow is not None:
        mlflow.log_artifact(path)


def log_sklearn_model(model: Any, artifact_path: str) -> None:
    if mlflow_sklearn is not None:
        mlflow_sklearn.log_model(model, artifact_path)


def log_xgboost_model(model: Any, artifact_path: str) -> None:
    if mlflow_xgboost is not None:
        mlflow_xgboost.log_model(model, artifact_path)
