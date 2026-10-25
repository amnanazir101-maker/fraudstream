"""MLflow experiment tracking and model registry.

Every trained model is logged as a run and registered as a new version of one registered model.
The version serving live traffic carries the alias "champion".
"""

from __future__ import annotations

import io
import logging
import warnings
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import version

import mlflow
import pandas as pd
from mlflow import MlflowClient

from fraudstream.config import Config
from fraudstream.features import FEATURES
from fraudstream.model import ModelBundle

CHAMPION = "champion"
_REQUIREMENTS = [
    f"{package}=={version(package)}" for package in ("lightgbm", "scikit-learn", "pandas", "numpy")
]


def use_experiment(config: Config, name: str) -> None:
    """Point MLflow at our database and keep the experiment's artifacts in the data folder."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    mlflow.set_tracking_uri(config.mlflow_uri)
    if mlflow.get_experiment_by_name(name) is None:
        artifacts = (config.data_dir / "mlartifacts").resolve().as_uri()
        mlflow.create_experiment(name, artifact_location=artifacts)
    mlflow.set_experiment(name)


class Registry:
    def __init__(self, config: Config) -> None:
        use_experiment(config, config.experiment)
        self.client = MlflowClient()
        self.name = config.model_name

    def log(
        self,
        bundle: ModelBundle,
        sample: pd.DataFrame,
        *,
        run_name: str,
        params: dict,
        metrics: dict,
        tags: dict | None = None,
    ) -> str:
        """Log a model run and register it as a new version. Returns the version number."""
        with mlflow.start_run(run_name=run_name), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tags(tags or {})
            info = mlflow.lightgbm.log_model(
                bundle.model,
                name="model",
                input_example=sample[FEATURES].head(3),
                pip_requirements=_REQUIREMENTS,  # skips slow requirement inference
            )
            mlflow.log_dict(
                {
                    "threshold": bundle.threshold,
                    "reference": bundle.reference,
                    "metrics": bundle.metrics,
                    "trained_until_day": bundle.trained_until_day,
                },
                "bundle.json",
            )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):  # MLflow prints
            version = str(mlflow.register_model(info.model_uri, self.name).version)
        bundle.version = version
        return version

    def promote(self, version: str) -> None:
        self.client.set_registered_model_alias(self.name, CHAMPION, version)

    def champion(self) -> ModelBundle:
        """The version serving live traffic."""
        return self.load(self.client.get_model_version_by_alias(self.name, CHAMPION).version)

    def load(self, version: str) -> ModelBundle:
        """Any registered version, with its threshold and drift reference."""
        details = self.client.get_model_version(self.name, str(version))
        model = mlflow.lightgbm.load_model(f"models:/{self.name}/{version}")
        meta = mlflow.artifacts.load_dict(f"runs:/{details.run_id}/bundle.json")
        return ModelBundle(
            model=model,
            threshold=meta["threshold"],
            reference=meta["reference"],
            metrics=meta["metrics"],
            version=str(details.version),
            trained_until_day=meta["trained_until_day"],
        )
