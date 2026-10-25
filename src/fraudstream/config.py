"""Settings, read from FRAUDSTREAM_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path


def _coerce(raw: str, default: object) -> object:
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, Path):
        return Path(raw)
    return raw


@dataclass(frozen=True)
class Config:
    """Every knob of the system. Set any field with FRAUDSTREAM_<FIELD_NAME>."""

    data_dir: Path = Path("data")
    tracking_uri: str = ""  # MLflow; empty means sqlite:///<data_dir>/mlflow.db
    experiment: str = "fraudstream"
    model_name: str = "fraud-detector"

    # Simulated world
    seed: int = 7
    customers: int = 2000
    terminals: int = 5000
    history_days: int = 90  # days simulated before the model goes live
    label_delay_days: int = 3  # fraud labels arrive this many days after the payment

    # Live stream
    live_days: int = 30
    attack_after_days: int = 10  # card-testing attack starts this many days into the stream

    # Monitoring and retraining
    drift_threshold: float = 0.2  # population stability index that counts as drift
    retrain_after_label_days: int = 3  # labelled days needed after drift before retraining
    retrain_window_days: int = 60  # most recent labelled days used to retrain
    retrain_cooldown_days: int = 5

    @property
    def mlflow_uri(self) -> str:
        if self.tracking_uri:
            return self.tracking_uri
        return f"sqlite:///{(self.data_dir / 'mlflow.db').resolve().as_posix()}"

    @property
    def attack_day(self) -> int:
        return self.history_days + self.attack_after_days

    @classmethod
    def from_env(cls, **overrides: object) -> Config:
        values: dict[str, object] = {}
        for field in fields(cls):
            raw = os.getenv(f"FRAUDSTREAM_{field.name.upper()}")
            if raw not in (None, ""):
                values[field.name] = _coerce(raw, field.default)
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)
