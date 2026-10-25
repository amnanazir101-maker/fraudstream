import numpy as np
import pandas as pd
import pytest

from fraudstream.monitoring import RetrainPolicy, drift_report, make_reference, psi


def frame(rng: np.random.Generator, n: int, shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": rng.gamma(2, 20, n) + shift,
            "hour": rng.integers(6, 24, n),
            "terminal_distance": rng.uniform(0, 5, n),
            "cust_nb_tx_1d": rng.poisson(2, n) + 1.0,
            "amount_to_avg_30d": rng.lognormal(0, 0.3, n),
        }
    )


def test_psi_is_zero_for_identical_distributions():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0


def test_same_distribution_does_not_drift():
    rng = np.random.default_rng(0)
    reference = make_reference(frame(rng, 20_000))
    assert not drift_report(reference, frame(rng, 4_000)).drifted


def test_a_small_new_pattern_is_caught_by_the_novelty_rate():
    rng = np.random.default_rng(1)
    reference = make_reference(frame(rng, 20_000))
    today = frame(rng, 4_000)
    today.loc[:39, "terminal_distance"] = rng.uniform(20, 90, 40)  # 1%, beyond anything seen
    report = drift_report(reference, today)
    assert report.psi["terminal_distance"] < 0.2  # too small for PSI alone
    assert report.drifted
    assert report.novelty["terminal_distance"] == pytest.approx(0.01, abs=0.002)
    assert report.alarms[0].endswith(
        "of payments have terminal_distance outside the training range"
    )


def test_a_broad_shift_is_caught_by_psi():
    rng = np.random.default_rng(2)
    reference = make_reference(frame(rng, 20_000))
    report = drift_report(reference, frame(rng, 4_000, shift=40.0))
    assert report.drifted
    assert report.alarms[0].startswith("amount PSI")


def test_policy_waits_for_labels_after_drift():
    policy = RetrainPolicy(after_label_days=3, cooldown_days=5)
    args = {"trained_on_day": 90, "recent_recall": 0.8, "training_recall": 0.8}
    assert policy.decide(today=104, first_drift_day=100, last_labelled_day=101, **args) is None
    assert "drift" in policy.decide(today=105, first_drift_day=100, last_labelled_day=102, **args)


def test_policy_retrains_when_recall_collapses_and_respects_cooldown():
    policy = RetrainPolicy(cooldown_days=5)
    common = {"first_drift_day": None, "last_labelled_day": 99, "training_recall": 0.8}
    assert "recall" in policy.decide(today=100, trained_on_day=90, recent_recall=0.3, **common)
    assert policy.decide(today=93, trained_on_day=90, recent_recall=0.3, **common) is None
