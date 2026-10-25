"""Batch steps: simulate the history, build features, train and register the first model."""

from __future__ import annotations

import json
import pickle

import pandas as pd

from fraudstream.config import Config
from fraudstream.features import FEATURES, FeatureStore, label_events, replay
from fraudstream.metrics import evaluate
from fraudstream.model import (
    LGBM_PARAMS,
    evaluate_bundle,
    time_split,
    train_baseline,
    train_bundle,
)
from fraudstream.registry import Registry
from fraudstream.simulator import DAY, TransactionSimulator, World, make_world

VALID_DAYS = 14
TEST_DAYS = 14


def simulate_history(config: Config) -> pd.DataFrame:
    """Create the world and the transactions that happened before the model goes live."""
    world = make_world(config.customers, config.terminals, config.seed)
    world.save(config.data_dir)
    simulator = TransactionSimulator(world, seed=config.seed)
    history = pd.concat(simulator.days(0, config.history_days), ignore_index=True)
    history.to_parquet(config.data_dir / "history.parquet", index=False)
    return history


def build_features(config: Config) -> pd.DataFrame:
    """Replay the history through the feature store (point-in-time correct).

    Saves the training table, the feature store's state at go-live, and the labels that are
    still on their way (they arrive during the live stream).
    """
    world = World.load(config.data_dir)
    history = pd.read_parquet(config.data_dir / "history.parquet")
    store = FeatureStore(world.customer_xy, world.terminal_xy, config.label_delay_days)
    labels = label_events(history, config.label_delay_days)
    go_live = config.history_days * DAY
    features = replay(store, history, labels[labels["t"] < go_live])

    table = pd.concat([history[["tx_id", "day", "is_fraud", "scenario"]], features], axis=1)
    table.to_parquet(config.data_dir / "features.parquet", index=False)
    with open(config.data_dir / "feature_store.pkl", "wb") as file:
        pickle.dump(store, file)
    pending = labels[labels["t"] >= go_live].merge(history[["tx_id", "scenario"]], on="tx_id")
    pending.to_parquet(config.data_dir / "pending_labels.parquet", index=False)
    return table


def train_initial(config: Config, table: pd.DataFrame) -> dict:
    """Train LightGBM and a logistic-regression baseline, log both, promote LightGBM."""
    train, valid, test = time_split(table, VALID_DAYS, TEST_DAYS)
    bundle = train_bundle(train, valid, config.seed)
    bundle.metrics = evaluate_bundle(bundle, test)

    baseline, baseline_threshold = train_baseline(train, valid)
    baseline_metrics = evaluate(
        test["is_fraud"].to_numpy(),
        baseline.predict_proba(test[FEATURES])[:, 1],
        baseline_threshold,
    )

    registry = Registry(config)
    split = {
        "train_days": f"{train['day'].min()}-{train['day'].max()}",
        "valid_days": f"{valid['day'].min()}-{valid['day'].max()}",
        "test_days": f"{test['day'].min()}-{test['day'].max()}",
    }
    params = {**LGBM_PARAMS, **split, "best_iteration": bundle.model.best_iteration_,
              "threshold": round(bundle.threshold, 4), "features": len(FEATURES)}  # fmt: skip
    version = registry.log(
        bundle,
        test,
        run_name="initial-model",
        params=params,
        metrics={f"test_{k}": v for k, v in bundle.metrics.items()},
        tags={"trigger": "initial training", "model": "lightgbm"},
    )
    registry.promote(version)
    _log_baseline(baseline_metrics, split)

    report = {
        "model_version": version,
        "split": split,
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "fraud_rate": float(table["is_fraud"].mean()),
        "lightgbm": bundle.metrics,
        "logistic_regression": baseline_metrics,
    }
    (config.data_dir / "training_report.json").write_text(json.dumps(report, indent=2))
    return report


def _log_baseline(metrics: dict, split: dict) -> None:
    import mlflow

    with mlflow.start_run(run_name="baseline-logistic-regression"):
        mlflow.log_params({**split, "model": "logistic regression (balanced)"})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.set_tags({"trigger": "baseline", "model": "logistic_regression"})
