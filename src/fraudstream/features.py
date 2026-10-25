"""Online feature store: one code path for training (by replaying history) and live scoring.

Every feature uses only what was known at the moment of the payment:

- card-holder windows (1, 7, 30 days) include the current payment and nothing later;
- terminal risk windows use only transactions whose fraud label has already arrived. Labels
  arrive `label_delay_days` after the payment (investigations take time), so the terminal
  windows cover (t - delay - w, t - delay].

Because training features come from the same `FeatureStore` replaying events in time order,
there is no training/serving skew and no look-ahead leakage.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fraudstream.simulator import DAY

WINDOWS = (1, 7, 30)  # days
FEATURES = [
    "amount",
    "hour",
    "is_night",
    "is_weekend",
    *[f"cust_nb_tx_{w}d" for w in WINDOWS],
    *[f"cust_avg_amount_{w}d" for w in WINDOWS],
    "amount_to_avg_30d",
    "terminal_distance",
    "new_terminal",
    *[f"term_nb_tx_{w}d" for w in WINDOWS],
    *[f"term_risk_{w}d" for w in WINDOWS],
]


class Window:
    """Count and sum of the values seen in a sliding time window."""

    __slots__ = ("width", "items", "count", "total")

    def __init__(self, width: float) -> None:
        self.width = width
        self.items: deque[tuple[float, float]] = deque()
        self.count = 0
        self.total = 0.0

    def add(self, t: float, value: float) -> None:
        self.items.append((t, value))
        self.count += 1
        self.total += value

    def expire(self, now: float) -> None:
        cutoff = now - self.width
        items = self.items
        if items and items[0][0] <= cutoff:
            while items and items[0][0] <= cutoff:
                self.total -= items.popleft()[1]
                self.count -= 1
            if not self.count:
                self.total = 0.0  # reset float drift


@dataclass
class FeatureStore:
    customer_xy: np.ndarray
    terminal_xy: np.ndarray
    label_delay_days: int = 3
    customers: dict[int, tuple[Window, ...]] = field(default_factory=dict)
    seen_terminals: dict[int, set[int]] = field(default_factory=dict)
    terminals: dict[int, tuple[Window, ...]] = field(default_factory=dict)
    awaiting_label: dict[int, tuple[int, float]] = field(default_factory=dict)  # tx -> terminal, t
    now: float = 0.0

    def __post_init__(self) -> None:
        # Plain lists: math.dist on Python floats is much faster than on numpy rows.
        self._customer_xy = np.asarray(self.customer_xy, dtype=float).tolist()
        self._terminal_xy = np.asarray(self.terminal_xy, dtype=float).tolist()

    def transaction(
        self, tx_id: int, t: float, customer: int, terminal: int, amount: float
    ) -> list[float]:
        """Features of a payment, computed from past events only; then remember the payment."""
        self.now = max(self.now, t)
        hour = int(t % DAY) // 3600
        is_weekend = float((int(t // DAY) % 7) >= 5)

        windows = self.customers.get(customer)
        if windows is None:
            windows = self.customers[customer] = tuple(Window(w * DAY) for w in WINDOWS)
        for window in windows:
            window.expire(t)
        month = windows[-1]
        ratio = (
            amount / (month.total / month.count) if month.count and month.total > 0 else math.nan
        )
        for window in windows:
            window.add(t, amount)

        seen = self.seen_terminals.setdefault(customer, set())
        new_terminal = float(terminal not in seen)
        seen.add(terminal)
        distance = math.dist(self._customer_xy[customer], self._terminal_xy[terminal])

        term = self.terminals.get(terminal)
        if term is None:
            term = self.terminals[terminal] = tuple(Window(w * DAY) for w in WINDOWS)
        known_until = t - self.label_delay_days * DAY
        for window in term:
            window.expire(known_until)
        self.awaiting_label[tx_id] = (terminal, t)

        return [
            amount,
            float(hour),
            float(hour < 6),
            is_weekend,
            *[float(w.count) for w in windows],
            *[w.total / w.count for w in windows],
            ratio,
            distance,
            new_terminal,
            *[float(w.count) for w in term],
            *[(w.total / w.count if w.count else 0.0) for w in term],
        ]

    def label(self, tx_id: int, is_fraud: int) -> None:
        """A fraud label arrived: the payment now counts towards its terminal's risk."""
        known = self.awaiting_label.pop(tx_id, None)
        if known is None:
            return
        terminal, t = known
        windows = self.terminals.get(terminal)
        if windows is None:
            windows = self.terminals[terminal] = tuple(Window(w * DAY) for w in WINDOWS)
        for window in windows:
            window.add(t, float(is_fraud))


def label_events(transactions: pd.DataFrame, label_delay_days: int) -> pd.DataFrame:
    """When each payment's label becomes known."""
    return pd.DataFrame(
        {
            "t": transactions["t"].to_numpy() + label_delay_days * DAY,
            "tx_id": transactions["tx_id"].to_numpy(),
            "is_fraud": transactions["is_fraud"].to_numpy(),
        }
    )


def replay(store: FeatureStore, transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Feed payments and label arrivals to the store in time order; return payment features.

    A label arriving at the same second as a payment is processed first.
    """
    tx_times = transactions["t"].to_numpy()
    label_times = labels["t"].to_numpy()
    order = np.lexsort(
        (
            np.concatenate([np.ones(len(tx_times)), np.zeros(len(label_times))]),
            np.concatenate([tx_times, label_times]),
        )
    )
    n_tx = len(tx_times)
    # Plain Python lists: indexing them is much faster than indexing numpy arrays.
    tx_ids = transactions["tx_id"].tolist()
    times = tx_times.tolist()
    customers = transactions["customer_id"].tolist()
    terminals = transactions["terminal_id"].tolist()
    amounts = transactions["amount"].tolist()
    label_ids = labels["tx_id"].tolist()
    label_values = labels["is_fraud"].tolist()

    rows: list[list[float] | None] = [None] * n_tx
    add_transaction, add_label = store.transaction, store.label
    for i in order.tolist():
        if i < n_tx:
            rows[i] = add_transaction(tx_ids[i], times[i], customers[i], terminals[i], amounts[i])
        else:
            add_label(label_ids[i - n_tx], label_values[i - n_tx])
    return pd.DataFrame(rows, columns=FEATURES, index=transactions.index)
