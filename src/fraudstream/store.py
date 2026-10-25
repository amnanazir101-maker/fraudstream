"""SQLite event log shared by the stream processor (writer) and the dashboard (reader)."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    tx_id INTEGER PRIMARY KEY, t REAL, day INTEGER, customer_id INTEGER, terminal_id INTEGER,
    amount REAL, score REAL, model_version TEXT, reasons TEXT, is_fraud INTEGER
);
CREATE TABLE IF NOT EXISTS days (
    day INTEGER PRIMARY KEY, transactions INTEGER, alerts INTEGER, model_version TEXT,
    max_psi REAL, drift_feature TEXT, max_novelty REAL, drifted INTEGER, drift_reason TEXT,
    frauds INTEGER, caught INTEGER, caught_frozen INTEGER,
    precision REAL, recall REAL, recall_frozen REAL, pr_auc REAL, pr_auc_frozen REAL,
    attack_frauds INTEGER, attack_caught INTEGER, attack_caught_frozen INTEGER
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, day INTEGER, kind TEXT, message TEXT
);
CREATE TABLE IF NOT EXISTS status (key TEXT PRIMARY KEY, value TEXT);
"""


class EventLog:
    def __init__(
        self,
        path: Path,
        reset: bool = False,
        on_event: Callable[[int, str, str], None] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            for suffix in ("", "-wal", "-shm"):
                Path(f"{path}{suffix}").unlink(missing_ok=True)
        self.path = path
        self.on_event = on_event
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")  # the dashboard reads while we write
        self._db.executescript(_SCHEMA)

    def add_alerts(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self._lock, self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO alerts VALUES (:tx_id, :t, :day, :customer_id, "
                ":terminal_id, :amount, :score, :model_version, :reasons, NULL)",
                [{**row, "reasons": json.dumps(row["reasons"])} for row in rows],
            )

    def label_alerts(self, labels: list[tuple[int, int]]) -> None:
        with self._lock, self._db:
            self._db.executemany(
                "UPDATE alerts SET is_fraud = ? WHERE tx_id = ?",
                [(is_fraud, tx_id) for tx_id, is_fraud in labels],
            )

    def upsert_day(self, day: int, **values: object) -> None:
        columns = ", ".join(["day", *values])
        marks = ", ".join(["?"] * (len(values) + 1))
        updates = ", ".join(f"{key} = excluded.{key}" for key in values)
        with self._lock, self._db:
            self._db.execute(
                f"INSERT INTO days ({columns}) VALUES ({marks}) "
                f"ON CONFLICT(day) DO UPDATE SET {updates}",
                [day, *values.values()],
            )

    def add_event(self, day: int, kind: str, message: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO events (day, kind, message) VALUES (?, ?, ?)", (day, kind, message)
            )
        if self.on_event:
            self.on_event(day, kind, message)

    def set_status(self, **values: object) -> None:
        with self._lock, self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO status VALUES (?, ?)",
                [(key, json.dumps(value)) for key, value in values.items()],
            )

    def close(self) -> None:
        self._db.close()


def read_table(path: Path, query: str) -> pd.DataFrame:
    """Read-only access for the dashboard (a fresh connection per read)."""
    if not path.exists():
        return pd.DataFrame()
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=10)
    try:
        return pd.read_sql_query(query, db)
    finally:
        db.close()


def read_status(path: Path) -> dict:
    table = read_table(path, "SELECT key, value FROM status")
    return {row.key: json.loads(row.value) for row in table.itertuples()} if len(table) else {}
