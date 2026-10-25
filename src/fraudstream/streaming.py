"""Live scoring: the event stream and the processor that scores, monitors and retrains.

Events are plain dicts, so the same processor runs on the local simulator or on Kafka:

- {"type": "transaction", "tx_id", "t", "customer_id", "terminal_id", "amount"}
- {"type": "label", "tx_id", "t", "is_fraud", "scenario"}  (arrives days after the payment)
"""

from __future__ import annotations

import logging
import pickle
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraudstream.config import Config
from fraudstream.features import FEATURES, FeatureStore
from fraudstream.metrics import alert_metrics
from fraudstream.model import ModelBundle, evaluate_bundle, train_bundle
from fraudstream.monitoring import RetrainPolicy, drift_report
from fraudstream.registry import Registry
from fraudstream.simulator import DAY, FraudConfig, TransactionSimulator, World
from fraudstream.store import EventLog, read_status, read_table

log = logging.getLogger(__name__)
ATTACK = 4  # simulator scenario id of the card-testing attack
PAYMENT_FIELDS = ("tx_id", "t", "customer_id", "terminal_id", "amount")


def live_events(config: Config) -> Iterator[dict]:
    """Payments of the live days plus label arrivals, in time order (labels first on ties).

    After the last day, the remaining labels keep arriving so every day can be evaluated.
    """
    world = World.load(config.data_dir)
    simulator = TransactionSimulator(
        world, FraudConfig(attack_start_day=config.attack_day), seed=config.seed
    )
    pending = pd.read_parquet(config.data_dir / "pending_labels.parquet")
    labels = deque(pending[["t", "tx_id", "is_fraud", "scenario"]].itertuples(index=False))
    delay = config.label_delay_days * DAY

    first, last = config.history_days, config.history_days + config.live_days
    for day in range(first, last):
        payments = simulator.day(day)
        due = []
        while labels and labels[0].t < (day + 1) * DAY:
            due.append(labels.popleft())
        labels.extend(
            payments.assign(t=payments["t"] + delay)[["t", "tx_id", "is_fraud", "scenario"]]
            .itertuples(index=False)
        )  # fmt: skip
        merged = [(row.t, 0, row) for row in due] + [
            (row.t, 1, row) for row in payments.itertuples(index=False)
        ]
        merged.sort(key=lambda item: (item[0], item[1]))
        for _, kind, row in merged:
            yield _label(row) if kind == 0 else _payment(row)
    while labels:
        yield _label(labels.popleft())


def _payment(row) -> dict:
    return {"type": "transaction", "tx_id": int(row.tx_id), "t": float(row.t),
            "customer_id": int(row.customer_id), "terminal_id": int(row.terminal_id),
            "amount": float(row.amount)}  # fmt: skip


def _label(row) -> dict:
    return {"type": "label", "tx_id": int(row.tx_id), "t": float(row.t),
            "is_fraud": int(row.is_fraud), "scenario": int(row.scenario)}  # fmt: skip


def paced(events: Iterable[dict], day_seconds: float) -> Iterator[dict]:
    """Replay events in (sped-up) real time: one simulated day every `day_seconds`.

    If the consumer pauses (for example to retrain), the clock resumes from where it was
    instead of rushing to catch up.
    """
    start_real, start_sim = time.monotonic(), None
    for event in events:
        if day_seconds > 0:
            start_sim = event["t"] if start_sim is None else start_sim
            due = (event["t"] - start_sim) / DAY * day_seconds
            ahead = due - (time.monotonic() - start_real)
            if ahead > 0.02:
                time.sleep(ahead)
            elif ahead < -1.0:
                start_real = time.monotonic() - due
        yield event


class StreamProcessor:
    """Scores payments, tracks drift and performance, and retrains in the background.

    A frozen copy of the first model scores every payment too ("shadow" model), so the
    dashboard can show what would have happened without retraining.
    """

    def __init__(
        self,
        config: Config,
        champion: ModelBundle,
        store: FeatureStore,
        events_log: EventLog,
        history: pd.DataFrame,
        registry: Registry | None = None,
        batch_size: int = 500,
        wait_for_retraining: bool = False,
    ) -> None:
        self.config = config
        self.wait_for_retraining = wait_for_retraining  # train in the foreground (tests, fast runs)
        self.champion = champion
        self.frozen = champion
        self.reference = champion.reference
        self.store = store
        self.log = events_log
        self.history = history[history["day"] >= history["day"].min() + 7]
        self.registry = registry
        self.batch_size = batch_size
        self.policy = RetrainPolicy(config.retrain_after_label_days, config.retrain_cooldown_days)

        self.deployed_day = config.history_days
        self.first_drift_day: int | None = None
        self.last_labelled_day: int | None = None
        self.processed = 0
        self.retrain_thread: threading.Thread | None = None
        self.on_day_closed: Callable[[int, dict], None] | None = None

        self._lock = threading.Lock()
        self._day: int | None = None
        self._batch: list[tuple[dict, list[float]]] = []
        self._batch_ids: set[int] = set()
        self._day_rows: list[list[float]] = []
        self._counts: dict[int, int] = defaultdict(int)
        self._alerts: dict[int, int] = defaultdict(int)
        self._awaiting: dict[int, tuple] = {}
        self._outcomes: dict[int, list[tuple]] = defaultdict(list)
        self._recall: dict[int, float] = {}
        self._labelled: list[tuple[list[float], int, int]] = []  # features, label, day
        self._alert_labels: list[tuple[int, int]] = []

    # --- events ------------------------------------------------------------------------
    def handle(self, event: dict) -> None:
        if event["type"] == "transaction":
            self._on_transaction(event)
        elif event["type"] == "label":
            self._on_label(event)

    def _on_transaction(self, event: dict) -> None:
        day = int(event["t"] // DAY)
        if self._day is None:
            self._day = day
        elif day > self._day:
            self._close_day()
            self._day = day
            self._maybe_retrain(day)
        features = self.store.transaction(
            event["tx_id"], event["t"], event["customer_id"], event["terminal_id"], event["amount"]
        )
        self._batch.append((event, features))
        self._batch_ids.add(event["tx_id"])
        self._day_rows.append(features)
        if len(self._batch) >= self.batch_size:
            self._flush()

    def _on_label(self, event: dict) -> None:
        self.store.label(event["tx_id"], event["is_fraud"])
        if event["tx_id"] in self._batch_ids:  # only possible with a zero label delay
            self._flush()
        known = self._awaiting.pop(event["tx_id"], None)
        if known is None:  # a payment from before go-live
            return
        day, row, score, alert, frozen_score, frozen_alert = known
        label = int(event["is_fraud"])
        self._outcomes[day].append(
            (label, score, alert, frozen_score, frozen_alert, event["scenario"] == ATTACK)
        )
        self._labelled.append((row, label, day))
        if alert:
            self._alert_labels.append((event["tx_id"], label))
        if len(self._outcomes[day]) == self._counts[day]:
            self._finish_day(day)

    # --- scoring -------------------------------------------------------------------------
    def _flush(self) -> None:
        if not self._batch:
            return
        events, rows = zip(*self._batch, strict=True)
        self._batch = []
        self._batch_ids = set()
        features = pd.DataFrame(list(rows), columns=FEATURES)
        with self._lock:
            champion = self.champion
        scores = champion.score(features)
        alerts = scores >= champion.threshold
        if self.frozen is champion:
            frozen_scores, frozen_alerts = scores, alerts
        else:
            frozen_scores = self.frozen.score(features)
            frozen_alerts = frozen_scores >= self.frozen.threshold

        flagged = np.flatnonzero(alerts)
        reasons = champion.reasons(features.iloc[flagged]) if len(flagged) else []
        self.log.add_alerts(
            [
                {**{k: events[i][k] for k in PAYMENT_FIELDS},
                 "day": int(events[i]["t"] // DAY), "score": float(scores[i]),
                 "model_version": champion.version, "reasons": reason}
                for i, reason in zip(flagged, reasons, strict=True)
            ]
        )  # fmt: skip
        for i, event in enumerate(events):
            day = int(event["t"] // DAY)
            self._counts[day] += 1
            self._alerts[day] += int(alerts[i])
            self._awaiting[event["tx_id"]] = (
                day, rows[i], float(scores[i]), bool(alerts[i]),
                float(frozen_scores[i]), bool(frozen_alerts[i]),
            )  # fmt: skip
        self.processed += len(events)

    # --- monitoring ----------------------------------------------------------------------
    def _close_day(self) -> None:
        self._flush()
        day = self._day
        features = pd.DataFrame(self._day_rows, columns=FEATURES)
        self._day_rows = []
        with self._lock:
            reference, version = self.reference, self.champion.version
        report = drift_report(reference, features, self.config.drift_threshold)
        feature, value = report.worst
        reason = report.alarms[0] if report.drifted else ""
        self.log.upsert_day(
            day,
            transactions=self._counts[day],
            alerts=self._alerts[day],
            model_version=version,
            max_psi=round(value, 4),
            drift_feature=feature,
            max_novelty=round(max(report.novelty.values(), default=0.0), 5),
            drifted=int(report.drifted),
            drift_reason=reason,
        )
        if report.drifted and self.first_drift_day is None:
            self.first_drift_day = day
            self.log.add_event(day, "drift", f"Data drift detected: {reason}")
        if self._alert_labels:
            self.log.label_alerts(self._alert_labels)
            self._alert_labels = []
        self._status(day)
        if self.on_day_closed:
            self.on_day_closed(
                day,
                {"transactions": self._counts[day], "alerts": self._alerts[day],
                 "model_version": version, "drift": reason},
            )  # fmt: skip

    def _finish_day(self, day: int) -> None:
        """All labels of a day are in: record how well each model did."""
        rows = np.array(self._outcomes.pop(day), dtype=float)
        labels, scores, alerts, frozen_scores, frozen_alerts, attack = rows.T
        live, frozen = alert_metrics(labels, alerts), alert_metrics(labels, frozen_alerts)
        attack = attack.astype(bool)
        both = 0 < labels.sum() < len(labels)
        self.log.upsert_day(
            day,
            frauds=live["frauds"],
            caught=live["caught"],
            caught_frozen=frozen["caught"],
            precision=round(live["precision"], 4),
            recall=round(live["recall"], 4),
            recall_frozen=round(frozen["recall"], 4),
            pr_auc=round(average_precision_score(labels, scores), 4) if both else None,
            pr_auc_frozen=round(average_precision_score(labels, frozen_scores), 4)
            if both
            else None,
            attack_frauds=int(attack.sum()),
            attack_caught=int((attack & alerts.astype(bool)).sum()),
            attack_caught_frozen=int((attack & frozen_alerts.astype(bool)).sum()),
        )
        if live["frauds"]:
            self._recall[day] = live["recall"]
        self.last_labelled_day = max(day, self.last_labelled_day or day)

    def _status(self, day: int) -> None:
        self.log.set_status(
            day=day,
            processed=self.processed,
            model_version=self.champion.version,
            retraining=bool(self.retrain_thread and self.retrain_thread.is_alive()),
            attack_day=self.config.attack_day,
            first_day=self.config.history_days,
            last_day=self.config.history_days + self.config.live_days - 1,
        )

    # --- retraining ----------------------------------------------------------------------
    def _maybe_retrain(self, today: int) -> None:
        if self.retrain_thread and self.retrain_thread.is_alive():
            return
        # Judge the current champion only on days it scored itself.
        served = [d for d in sorted(self._recall) if d >= self.deployed_day]
        recent = [self._recall[d] for d in served[-3:]]
        reason = self.policy.decide(
            today=today,
            trained_on_day=self.deployed_day,
            first_drift_day=self.first_drift_day,
            last_labelled_day=self.last_labelled_day,
            recent_recall=float(np.mean(recent)) if recent else None,
            training_recall=self.champion.metrics.get("recall", 0.0),
        )
        if reason is None:
            return
        self.log.add_event(today, "retrain", f"Retraining started: {reason}")
        data = self._training_data()
        self.retrain_thread = threading.Thread(
            target=self._retrain, args=(data, today, reason), daemon=True
        )
        self.retrain_thread.start()
        self._status(today)
        if self.wait_for_retraining:
            self.retrain_thread.join()
            self._status(today)

    def _training_data(self) -> pd.DataFrame:
        rows, labels, days = zip(*self._labelled, strict=True) if self._labelled else ((), (), ())
        live = pd.DataFrame(list(rows), columns=FEATURES).assign(
            is_fraud=list(labels), day=list(days)
        )
        data = pd.concat([self.history[[*FEATURES, "is_fraud", "day"]], live], ignore_index=True)
        # Only days whose labels have all arrived: a half-labelled day would bias training
        # (only early payments) and make a useless holdout.
        last = self.last_labelled_day if self.last_labelled_day is not None else data["day"].max()
        data = data[data["day"] <= last]
        return data[data["day"] > last - self.config.retrain_window_days]

    def _retrain(self, data: pd.DataFrame, today: int, reason: str) -> None:
        try:
            # The newest labelled day is the holdout for the champion-vs-challenger test. The
            # new fraud pattern only exists in the last few days, so those days must be in
            # the training data: validation for early stopping is a random 15% instead.
            last = int(data["day"].max())
            holdout = data[data["day"] == last]
            rest = data[data["day"] < last]
            valid = rest.sample(frac=0.15, random_state=self.config.seed)
            train = rest.drop(valid.index)
            challenger = train_bundle(train, valid, self.config.seed)
            with self._lock:
                champion = self.champion
            challenger.metrics = evaluate_bundle(challenger, holdout)
            current = evaluate_bundle(champion, holdout)
            if np.isnan(challenger.metrics["pr_auc"]) or np.isnan(current["pr_auc"]):
                # No fraud on the holdout day: compare on the validation rows instead.
                challenger.metrics = evaluate_bundle(challenger, valid)
                current = evaluate_bundle(champion, valid)
            better = challenger.metrics["pr_auc"] > current["pr_auc"]
            if self.registry:
                version = self.registry.log(
                    challenger,
                    holdout,
                    run_name=f"retrain-day-{today}",
                    params={"trigger": reason,
                            "train_days": f"{train['day'].min()}-{train['day'].max()}",
                            "threshold": round(challenger.threshold, 4)},
                    metrics={**{f"holdout_{k}": v for k, v in challenger.metrics.items()},
                             "champion_holdout_pr_auc": current["pr_auc"]},
                    tags={"trigger": reason, "promoted": str(better)},
                )  # fmt: skip
            else:
                version = f"{int(champion.version) + 1}" if champion.version.isdigit() else "2"
                challenger.version = version
            summary = (
                f"holdout PR-AUC {challenger.metrics['pr_auc']:.2f} vs champion "
                f"{current['pr_auc']:.2f}"
            )
            if better:
                if self.registry:
                    self.registry.promote(version)
                with self._lock:
                    self.champion = challenger
                    self.reference = challenger.reference
                    self.deployed_day = today
                    self.first_drift_day = None
                self.log.add_event(today, "promoted", f"Model v{version} promoted: {summary}")
            else:
                with self._lock:
                    self.deployed_day = today  # wait a cooldown before trying again
                self.log.add_event(today, "rejected", f"Challenger v{version} rejected: {summary}")
        except Exception as exc:  # a failed retrain must never stop live scoring
            log.exception("Retraining failed")
            self.log.add_event(today, "error", f"Retraining failed: {exc}")

    # --- end of stream -------------------------------------------------------------------
    def close(self) -> None:
        if self._day is not None:
            self._close_day()
        if self.retrain_thread:
            self.retrain_thread.join()
        self._status(self._day or 0)


def summarize(path: Path) -> dict:
    """Recall of the adaptive system vs the frozen first model, before and after the attack."""
    days = read_table(path, "SELECT * FROM days ORDER BY day")
    events = read_table(path, "SELECT * FROM events ORDER BY id")
    status = read_status(path)
    if days.empty or not status:
        return {}
    first, attack, last = status["first_day"], status["attack_day"], status["last_day"]
    promoted = events.loc[(events["kind"] == "promoted") & (events["day"] >= attack), "day"]
    retrained = int(promoted.min()) if len(promoted) else None
    periods = {"before the attack": (first, attack - 1)}
    if retrained:
        periods["attack, before retraining"] = (attack, retrained - 1)
        periods["after retraining"] = (retrained, last)
    else:
        periods["during the attack"] = (attack, last)

    def ratio(part: pd.Series, whole: pd.Series) -> float | None:
        return round(float(part.sum() / whole.sum()), 4) if whole.sum() else None

    summary = {"attack_day": attack, "retrained_day": retrained, "periods": {}}
    for name, (start, stop) in periods.items():
        rows = days[(days["day"] >= start) & (days["day"] <= stop)]
        summary["periods"][name] = {
            "days": f"{start}-{stop}",
            "recall": ratio(rows["caught"], rows["frauds"]),
            "recall_frozen": ratio(rows["caught_frozen"], rows["frauds"]),
            "precision": ratio(rows["caught"], rows["alerts"]),
            "attack_recall": ratio(rows["attack_caught"], rows["attack_frauds"]),
            "attack_recall_frozen": ratio(rows["attack_caught_frozen"], rows["attack_frauds"]),
        }
    drift = events.loc[(events["kind"] == "drift") & (events["day"] >= attack), "day"]
    summary["drift_detected_day"] = int(drift.min()) if len(drift) else None
    summary["false_drift_alarms"] = int(((days["day"] < attack) & (days["drifted"] == 1)).sum())
    return summary


def load_processor(
    config: Config, events_log: EventLog, wait_for_retraining: bool = False
) -> StreamProcessor:
    """A processor resuming from the end of the training history with the champion model."""
    registry = Registry(config)
    with open(config.data_dir / "feature_store.pkl", "rb") as file:
        store = pickle.load(file)  # our own file, written by `fraudstream train`
    history = pd.read_parquet(config.data_dir / "features.parquet")
    return StreamProcessor(
        config, registry.champion(), store, events_log, history, registry,
        wait_for_retraining=wait_for_retraining,
    )  # fmt: skip
