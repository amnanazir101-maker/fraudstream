"""FraudStream: real-time card-fraud detection with drift monitoring and automatic retraining."""

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep MLflow's import-time hint quiet

__version__ = "0.1.0"
