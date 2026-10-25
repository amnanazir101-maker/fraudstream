import math

import numpy as np
import pandas as pd
import pytest

from fraudstream.features import FEATURES, FeatureStore, label_events, replay
from fraudstream.simulator import DAY

XY = np.array([[0.0, 0.0], [10.0, 0.0]])  # two customers
TERMINALS = np.array([[3.0, 4.0], [10.0, 1.0]])  # two terminals


def store(delay: int = 1) -> FeatureStore:
    return FeatureStore(XY, TERMINALS, label_delay_days=delay)


def as_dict(row: list[float]) -> dict[str, float]:
    return dict(zip(FEATURES, row, strict=True))


def test_customer_windows_include_the_payment_and_expire():
    s = store()
    first = as_dict(s.transaction(1, 1 * DAY + 100, 0, 0, 10.0))
    assert first["cust_nb_tx_1d"] == 1 and first["cust_avg_amount_1d"] == 10.0
    assert math.isnan(first["amount_to_avg_30d"])  # no earlier spending
    assert first["terminal_distance"] == 5.0 and first["new_terminal"] == 1.0

    second = as_dict(s.transaction(2, 1 * DAY + 200, 0, 0, 30.0))
    assert second["cust_nb_tx_1d"] == 2 and second["cust_avg_amount_1d"] == 20.0
    assert second["amount_to_avg_30d"] == 3.0 and second["new_terminal"] == 0.0

    later = as_dict(s.transaction(3, 3 * DAY, 0, 0, 20.0))  # two days later
    assert later["cust_nb_tx_1d"] == 1 and later["cust_nb_tx_7d"] == 3


def test_terminal_risk_only_uses_labels_that_have_arrived():
    s = store(delay=1)
    s.transaction(1, 0.5 * DAY, 0, 0, 10.0)  # will be labelled fraud
    before = as_dict(s.transaction(2, 1.2 * DAY, 1, 0, 10.0))
    assert before["term_nb_tx_7d"] == 0  # the label is not known yet

    s.label(1, is_fraud=1)  # arrives at 1.5 days
    after = as_dict(s.transaction(3, 1.6 * DAY, 1, 0, 10.0))
    assert after["term_nb_tx_7d"] == 1 and after["term_risk_7d"] == 1.0


def test_replay_matches_a_brute_force_computation():
    rng = np.random.default_rng(0)
    n = 300
    payments = pd.DataFrame(
        {
            "tx_id": np.arange(n),
            "t": np.sort(rng.uniform(0, 20 * DAY, n)),
            "customer_id": rng.integers(0, 2, n),
            "terminal_id": rng.integers(0, 2, n),
            "amount": rng.uniform(1, 100, n).round(2),
            "is_fraud": rng.random(n) < 0.2,
        }
    ).astype({"is_fraud": int})
    features = replay(store(delay=2), payments, label_events(payments, 2))

    for i in rng.choice(n, 25, replace=False):
        p = payments.iloc[i]
        mine = payments[payments["customer_id"] == p["customer_id"]]
        week = mine[(mine["t"] > p["t"] - 7 * DAY) & (mine["t"] <= p["t"])]
        assert features.loc[i, "cust_nb_tx_7d"] == len(week)
        assert features.loc[i, "cust_avg_amount_7d"] == pytest.approx(week["amount"].mean())

        known = payments[
            (payments["terminal_id"] == p["terminal_id"])
            & (payments["t"] + 2 * DAY <= p["t"])  # label already arrived
            & (payments["t"] > p["t"] - 2 * DAY - 7 * DAY)
        ]
        assert features.loc[i, "term_nb_tx_7d"] == len(known)
        expected = known["is_fraud"].mean() if len(known) else 0.0
        assert features.loc[i, "term_risk_7d"] == pytest.approx(expected)
