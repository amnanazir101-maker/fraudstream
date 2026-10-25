"""Model training, explanations, and the MLflow model registry."""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraudstream.features import FEATURES
from fraudstream.metrics import best_f1_threshold, evaluate
from fraudstream.monitoring import make_reference

LGBM_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
}


@dataclass
class ModelBundle:
    """A trained model plus everything needed to use and monitor it."""

    model: lgb.LGBMClassifier
    threshold: float
    reference: dict  # training distribution of the monitored features
    metrics: dict = field(default_factory=dict)
    version: str = "local"
    trained_until_day: int = 0

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Fraud probabilities. Calls the booster directly: the scikit-learn wrapper adds
        milliseconds of overhead per call, which adds up on a stream of small batches."""
        return self.model.booster_.predict(features[FEATURES].to_numpy(dtype=float))

    def reasons(self, features: pd.DataFrame, top: int = 3) -> list[list[dict]]:
        """The features that pushed each score up most (SHAP values from LightGBM)."""
        matrix = features[FEATURES].to_numpy(dtype=float)
        contributions = self.model.booster_.predict(matrix, pred_contrib=True)
        explained = []
        for row, values in zip(contributions[:, :-1], matrix, strict=True):
            best = np.argsort(row)[::-1][:top]
            explained.append(
                [
                    {"feature": FEATURES[i], "value": round(float(values[i]), 3),
                     "impact": round(float(row[i]), 3)}
                    for i in best
                    if row[i] > 0
                ]
            )  # fmt: skip
        return explained


def time_split(frame: pd.DataFrame, valid_days: int, test_days: int, warmup_days: int = 7):
    """Train / validation / test by day, oldest first: the model never sees the future."""
    last = int(frame["day"].max())
    test_start = last - test_days + 1
    valid_start = test_start - valid_days
    usable = frame[frame["day"] >= frame["day"].min() + warmup_days]
    return (
        usable[usable["day"] < valid_start],
        usable[(usable["day"] >= valid_start) & (usable["day"] < test_start)],
        usable[usable["day"] >= test_start],
    )


def fit_lightgbm(train: pd.DataFrame, valid: pd.DataFrame, seed: int = 7) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed)
    model.fit(
        train[FEATURES],
        train["is_fraud"],
        eval_X=(valid[FEATURES],),
        eval_y=(valid["is_fraud"],),
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def train_bundle(train: pd.DataFrame, valid: pd.DataFrame, seed: int = 7) -> ModelBundle:
    """LightGBM with early stopping; alert threshold = best F1 on the validation rows.

    The drift reference is the validation data: the most recent "normal" the model knows.
    """
    model = fit_lightgbm(train, valid, seed)
    threshold = best_f1_threshold(valid["is_fraud"], model.predict_proba(valid[FEATURES])[:, 1])
    return ModelBundle(
        model=model,
        threshold=threshold,
        reference=make_reference(valid),
        trained_until_day=int(max(train["day"].max(), valid["day"].max())),
    )


def train_baseline(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[object, float]:
    """Logistic regression, the usual first model, for comparison."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000),
    )
    model.fit(train[FEATURES], train["is_fraud"])
    threshold = best_f1_threshold(valid["is_fraud"], model.predict_proba(valid[FEATURES])[:, 1])
    return model, threshold


def evaluate_bundle(bundle: ModelBundle, frame: pd.DataFrame) -> dict[str, float]:
    return evaluate(frame["is_fraud"].to_numpy(), bundle.score(frame), bundle.threshold)
