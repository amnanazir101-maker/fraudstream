"""Benchmark on real data: the ULB credit-card fraud dataset from OpenML.

284,807 card payments made by European card holders over two days in September 2013, of which
492 are fraud (0.17%). Features V1-V28 are PCA components (the originals are confidential).
Source: Dal Pozzolo et al., "Calibrating Probability with Undersampling for Unbalanced
Classification", IEEE CIDM 2015. OpenML dataset 1597.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraudstream.config import Config
from fraudstream.metrics import best_f1_threshold, evaluate
from fraudstream.model import LGBM_PARAMS

URL = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"
COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "hour"]


def load_creditcard(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "creditcard.parquet"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        with httpx.stream("GET", URL, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with open(partial, "wb") as file:
                for chunk in response.iter_bytes(1 << 20):
                    file.write(chunk)
        partial.replace(path)
    frame = pd.read_parquet(path)
    frame["Class"] = pd.to_numeric(frame["Class"].astype(str).str.strip("'"))
    frame["hour"] = (frame["Time"] // 3600) % 24
    return frame.sort_values("Time", kind="stable", ignore_index=True)


def run_benchmark(config: Config) -> dict:
    """Time-ordered split (60% train, 20% validation, 20% test); LightGBM vs logistic regression."""
    frame = load_creditcard(config.data_dir)
    n = len(frame)
    train, valid, test = (
        frame[: int(n * 0.6)],
        frame[int(n * 0.6) : int(n * 0.8)],
        frame[int(n * 0.8) :],
    )

    model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=config.seed)
    model.fit(
        train[COLUMNS], train["Class"],
        eval_X=(valid[COLUMNS],), eval_y=(valid["Class"],), eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )  # fmt: skip
    threshold = best_f1_threshold(valid["Class"], model.predict_proba(valid[COLUMNS])[:, 1])
    lightgbm = evaluate(
        test["Class"].to_numpy(), model.predict_proba(test[COLUMNS])[:, 1], threshold
    )

    baseline = make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000),
    )
    baseline.fit(train[COLUMNS], train["Class"])
    base_threshold = best_f1_threshold(valid["Class"], baseline.predict_proba(valid[COLUMNS])[:, 1])
    logistic = evaluate(
        test["Class"].to_numpy(), baseline.predict_proba(test[COLUMNS])[:, 1], base_threshold
    )

    report = {
        "dataset": "ULB credit-card fraud (OpenML 1597)",
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "test_frauds": int(test["Class"].sum()),
        "lightgbm": lightgbm,
        "logistic_regression": logistic,
    }
    _log(config, report, model.best_iteration_)
    (config.data_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2))
    return report


def _log(config: Config, report: dict, best_iteration: int) -> None:
    import mlflow

    from fraudstream.registry import use_experiment

    use_experiment(config, "creditcard-benchmark")
    for name, metrics in (
        ("lightgbm", report["lightgbm"]),
        ("logistic_regression", report["logistic_regression"]),
    ):
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {"model": name, "split": "time-ordered 60/20/20"}
                | ({"best_iteration": best_iteration} if name == "lightgbm" else {})
            )
            mlflow.log_metrics({k: v for k, v in metrics.items() if not np.isnan(v)})
