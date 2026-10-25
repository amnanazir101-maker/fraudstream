"""Synthetic card-payment world with realistic fraud scenarios and a mid-stream attack.

Follows the transaction simulator of the open handbook "Reproducible Machine Learning for
Credit Card Fraud Detection" (Le Borgne et al., Université Libre de Bruxelles):

1. Any transaction above 220 is fraudulent.
2. Every day two terminals are compromised; all their transactions in the next 28 days are fraud.
3. Every day three cards are compromised; for 14 days a third of their transactions are fraud,
   with amounts multiplied by five.

On top of that, a **card-testing attack** can start on a chosen day (concept drift): every day
a few cards make several small payments at night on terminals far from the card holder's home.
A model trained before the attack has never seen this pattern.

Each day is generated from its own random streams, so any day can be produced on its own,
deterministically, without replaying the days before it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86_400  # seconds
COLUMNS = ["tx_id", "t", "day", "customer_id", "terminal_id", "amount", "is_fraud", "scenario"]
SCENARIOS = {0: "legitimate", 1: "large amount", 2: "compromised terminal",
             3: "compromised card", 4: "card testing"}  # fmt: skip


@dataclass(frozen=True)
class FraudConfig:
    amount_threshold: float = 220.0
    terminals_per_day: int = 2
    terminal_days: int = 28
    cards_per_day: int = 3
    card_days: int = 14
    attack_start_day: int | None = None  # first day of the card-testing attack
    attack_cards_per_day: int = 6
    attack_days_per_card: int = 7
    attack_tx_per_day: tuple[int, int] = (3, 7)


@dataclass
class World:
    """Card holders (home location, spending habits) and payment terminals."""

    customers: pd.DataFrame  # customer_id, x, y, mean_amount, std_amount, mean_tx_per_day
    terminals: pd.DataFrame  # terminal_id, x, y
    radius: float = 5.0  # card holders pay at terminals within this distance of home

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.customers.to_parquet(folder / "customers.parquet", index=False)
        self.terminals.to_parquet(folder / "terminals.parquet", index=False)

    @classmethod
    def load(cls, folder: Path) -> World:
        return cls(
            pd.read_parquet(folder / "customers.parquet"),
            pd.read_parquet(folder / "terminals.parquet"),
        )

    @property
    def customer_xy(self) -> np.ndarray:
        return self.customers[["x", "y"]].to_numpy()

    @property
    def terminal_xy(self) -> np.ndarray:
        return self.terminals[["x", "y"]].to_numpy()


def make_world(n_customers: int = 2000, n_terminals: int = 5000, seed: int = 7) -> World:
    rng = np.random.default_rng([seed, 0])
    mean_amount = rng.uniform(5, 100, n_customers)
    customers = pd.DataFrame(
        {
            "customer_id": np.arange(n_customers),
            "x": rng.uniform(0, 100, n_customers),
            "y": rng.uniform(0, 100, n_customers),
            "mean_amount": mean_amount,
            "std_amount": mean_amount / 2,
            "mean_tx_per_day": rng.uniform(0, 4, n_customers),
        }
    )
    terminals = pd.DataFrame(
        {
            "terminal_id": np.arange(n_terminals),
            "x": rng.uniform(0, 100, n_terminals),
            "y": rng.uniform(0, 100, n_terminals),
        }
    )
    return World(customers, terminals)


class TransactionSimulator:
    def __init__(self, world: World, fraud: FraudConfig | None = None, seed: int = 7) -> None:
        self.world = world
        self.fraud = fraud or FraudConfig()
        self.seed = seed
        # Terminals near each customer's home, stored flat with offsets for vectorized picks.
        homes, spots = world.customer_xy, world.terminal_xy
        nearby = [np.flatnonzero(np.hypot(*(spots - home).T) < world.radius) for home in homes]
        self._lengths = np.array([len(n) for n in nearby])
        self._offsets = np.concatenate([[0], np.cumsum(self._lengths)[:-1]])
        self._flat = np.concatenate(nearby) if nearby else np.array([], dtype=int)

    def days(self, start: int, stop: int) -> Iterator[pd.DataFrame]:
        for day in range(start, stop):
            yield self.day(day)

    def day(self, day: int) -> pd.DataFrame:
        """All transactions of one day, in time order, with fraud labels."""
        frame = self._regular(day)
        self._apply_fraud(day, frame)
        if self.fraud.attack_start_day is not None and day >= self.fraud.attack_start_day:
            frame = pd.concat([frame, self._card_testing(day)], ignore_index=True)
        frame = frame.sort_values("t", kind="stable", ignore_index=True)
        frame["tx_id"] = day * 1_000_000 + np.arange(len(frame))
        frame["day"] = day
        return frame[COLUMNS]

    # --- regular spending ---------------------------------------------------------------
    def _regular(self, day: int) -> pd.DataFrame:
        rng = np.random.default_rng([self.seed, day, 0])
        customers = self.world.customers
        counts = rng.poisson(customers["mean_tx_per_day"].to_numpy())
        counts[self._lengths == 0] = 0  # nobody to pay nearby
        ids = np.repeat(customers["customer_id"].to_numpy(), counts)

        seconds = rng.normal(DAY / 2, 20_000, len(ids))
        bad = (seconds < 0) | (seconds >= DAY)
        while bad.any():  # redraw times outside the day
            seconds[bad] = rng.normal(DAY / 2, 20_000, bad.sum())
            bad = (seconds < 0) | (seconds >= DAY)

        mean = customers["mean_amount"].to_numpy()[ids]
        amount = rng.normal(mean, customers["std_amount"].to_numpy()[ids])
        negative = amount < 0
        amount[negative] = rng.uniform(0, 2 * mean[negative])

        pick = (rng.random(len(ids)) * self._lengths[ids]).astype(int)
        terminals = self._flat[self._offsets[ids] + pick]
        return pd.DataFrame(
            {
                "t": day * DAY + seconds,
                "customer_id": ids,
                "terminal_id": terminals,
                "amount": np.round(amount, 2),
                "is_fraud": 0,
                "scenario": 0,
            }
        )

    # --- fraud scenarios ------------------------------------------------------------------
    def _picks(self, day: int, stream: int, population: int, size: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, day, stream])
        return rng.choice(population, size=min(size, population), replace=False)

    def _active(self, day: int, stream: int, population: int, size: int, span: int) -> set[int]:
        """Everything picked on one of the last `span` days (including today)."""
        active: set[int] = set()
        for start in range(max(0, day - span + 1), day + 1):
            active.update(self._picks(start, stream, population, size).tolist())
        return active

    def _apply_fraud(self, day: int, frame: pd.DataFrame) -> None:
        fraud = self.fraud
        n_terminals, n_customers = len(self.world.terminals), len(self.world.customers)

        large = frame["amount"] > fraud.amount_threshold
        frame.loc[large, ["is_fraud", "scenario"]] = [1, 1]

        terminals = self._active(day, 1, n_terminals, fraud.terminals_per_day, fraud.terminal_days)
        hit = frame["terminal_id"].isin(terminals) & (frame["is_fraud"] == 0)
        frame.loc[hit, ["is_fraud", "scenario"]] = [1, 2]

        cards = self._active(day, 2, n_customers, fraud.cards_per_day, fraud.card_days)
        rng = np.random.default_rng([self.seed, day, 3])
        chosen = frame["customer_id"].isin(cards) & (rng.random(len(frame)) < 1 / 3)
        chosen &= frame["is_fraud"] == 0
        frame.loc[chosen, "amount"] = (frame.loc[chosen, "amount"] * 5).round(2)
        frame.loc[chosen, ["is_fraud", "scenario"]] = [1, 3]

    def _card_testing(self, day: int) -> pd.DataFrame:
        fraud = self.fraud
        cards = sorted(
            card
            for start in range(
                max(fraud.attack_start_day, day - fraud.attack_days_per_card + 1), day + 1
            )
            for card in self._picks(start, 4, len(self.world.customers), fraud.attack_cards_per_day)
        )
        rng = np.random.default_rng([self.seed, day, 5])
        low, high = fraud.attack_tx_per_day
        counts = rng.integers(low, high + 1, len(cards))
        ids = np.repeat(np.array(cards, dtype=int), counts)
        return pd.DataFrame(
            {
                "t": day * DAY + rng.uniform(0, 6 * 3600, len(ids)),  # at night
                "customer_id": ids,
                "terminal_id": rng.integers(0, len(self.world.terminals), len(ids)),  # anywhere
                "amount": np.round(rng.uniform(1, 10, len(ids)), 2),  # small test payments
                "is_fraud": 1,
                "scenario": 4,
            }
        )
