"""A small simulated world, trained once per test session and shared by the tests."""

from __future__ import annotations

import pytest

from fraudstream.config import Config
from fraudstream.pipeline import build_features, simulate_history, train_initial


@pytest.fixture(scope="session")
def small_config(tmp_path_factory) -> Config:
    return Config(
        data_dir=tmp_path_factory.mktemp("data"),
        customers=600,
        terminals=1500,
        history_days=60,
        live_days=16,
        attack_after_days=4,
        retrain_after_label_days=2,
        retrain_cooldown_days=3,
        retrain_window_days=50,
    )


@pytest.fixture(scope="session")
def trained(small_config: Config) -> dict:
    """Simulate, build features and train once; tests read the saved artifacts."""
    history = simulate_history(small_config)
    table = build_features(small_config)
    report = train_initial(small_config, table)
    return {"config": small_config, "history": history, "table": table, "report": report}
