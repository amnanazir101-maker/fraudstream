"""Data-drift detection and the retraining policy.

Two complementary tests run every day on each monitored feature:

- **Population stability index (PSI)** against the reference distribution catches broad
  shifts (0.1 = moderate, 0.2+ = significant).
- **Novelty rate**: the share of today's values outside the range seen in the reference data.
  A brand-new pattern (payments far from home, say) starts small, and PSI is slow to notice
  a small new cluster, but legitimate traffic almost never leaves the training range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Features whose distribution we watch in production.
MONITORED = ["amount", "hour", "terminal_distance", "cust_nb_tx_1d", "amount_to_avg_30d"]
_EPS = 1e-4


def make_reference(features: pd.DataFrame, columns: list[str] = MONITORED, bins: int = 10) -> dict:
    """Decile bin edges, bin proportions and the value range of each monitored feature."""
    reference = {}
    for column in columns:
        values = features[column].dropna().to_numpy(dtype=float)
        inner = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
        edges = np.unique(inner).tolist()
        reference[column] = {
            "edges": edges,
            "proportions": _proportions(values, edges),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return reference


def _proportions(values: np.ndarray, edges: list[float]) -> list[float]:
    counts = np.bincount(np.searchsorted(edges, values, side="left"), minlength=len(edges) + 1)
    return (counts / max(len(values), 1)).tolist()


def psi(expected: list[float], actual: list[float]) -> float:
    e = np.maximum(np.asarray(expected), _EPS)
    a = np.maximum(np.asarray(actual), _EPS)
    return float(np.sum((a - e) * np.log(a / e)))


@dataclass
class DriftReport:
    psi: dict[str, float]
    novelty: dict[str, float]  # share of values outside the reference range
    psi_threshold: float = 0.2
    novelty_threshold: float = 0.005

    @property
    def worst(self) -> tuple[str, float]:
        """The feature with the highest PSI and that PSI."""
        return max(self.psi.items(), key=lambda item: item[1])

    @property
    def drifted(self) -> bool:
        return bool(self.alarms)

    @property
    def alarms(self) -> list[str]:
        """Readable reasons, most severe first."""
        found = [
            f"{column} PSI {value:.2f}"
            for column, value in sorted(self.psi.items(), key=lambda item: -item[1])
            if value >= self.psi_threshold
        ]
        found += [
            f"{share:.1%} of payments have {column} outside the training range"
            for column, share in sorted(self.novelty.items(), key=lambda item: -item[1])
            if share >= self.novelty_threshold
        ]
        return found


def drift_report(
    reference: dict,
    features: pd.DataFrame,
    psi_threshold: float = 0.2,
    novelty_threshold: float = 0.005,
) -> DriftReport:
    scores, novelty = {}, {}
    for column, ref in reference.items():
        values = features[column].dropna().to_numpy(dtype=float)
        if len(values):
            scores[column] = psi(ref["proportions"], _proportions(values, ref["edges"]))
            novelty[column] = float(np.mean((values < ref["min"]) | (values > ref["max"])))
    return DriftReport(scores, novelty, psi_threshold, novelty_threshold)


@dataclass
class RetrainPolicy:
    """When to train a challenger model.

    - Data drift: once enough labelled days exist after the first drifted day, so the new
      model can actually learn the new pattern.
    - Performance: when recall on recent labelled days falls well below the training recall.
    A cooldown stops back-to-back retraining.
    """

    after_label_days: int = 3
    cooldown_days: int = 5
    recall_floor: float = 0.6  # share of the training recall that still counts as healthy

    def decide(
        self,
        today: int,
        trained_on_day: int,
        first_drift_day: int | None,
        last_labelled_day: int | None,
        recent_recall: float | None,
        training_recall: float,
    ) -> str | None:
        if today - trained_on_day < self.cooldown_days:
            return None
        if (
            first_drift_day is not None
            and last_labelled_day is not None
            and last_labelled_day - first_drift_day + 1 >= self.after_label_days
        ):
            return f"data drift since day {first_drift_day}"
        if recent_recall is not None and recent_recall < self.recall_floor * training_recall:
            return f"recall fell to {recent_recall:.0%} (training: {training_recall:.0%})"
        return None
