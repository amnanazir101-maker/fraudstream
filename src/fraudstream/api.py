"""Scoring API: send a payment, get a fraud probability, an alert decision and the reasons.

fraudstream serve        # then open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import pickle
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from fraudstream.config import Config
from fraudstream.features import FEATURES, FeatureStore
from fraudstream.model import ModelBundle

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)  # day 0 of the simulated world


class Payment(BaseModel):
    customer_id: int = Field(ge=0, examples=[42])
    terminal_id: int = Field(ge=0, examples=[1337])
    amount: float = Field(gt=0, examples=[57.3])
    timestamp: datetime | None = Field(
        default=None, description="Defaults to the current time of the simulated world."
    )
    transaction_id: int | None = None


class Reason(BaseModel):
    feature: str
    value: float
    impact: float


class Score(BaseModel):
    transaction_id: int
    fraud_probability: float
    alert: bool
    threshold: float
    model_version: str
    reasons: list[Reason]


class Feedback(BaseModel):
    transaction_id: int
    is_fraud: bool


class ScoringService:
    """The champion model plus a live feature store, safe to call from many requests."""

    def __init__(self, bundle: ModelBundle, store: FeatureStore) -> None:
        self.bundle = bundle
        self.store = store
        self._lock = threading.Lock()
        self._next_id = 9_000_000_000

    @classmethod
    def from_config(cls, config: Config) -> ScoringService:
        from fraudstream.registry import Registry

        with open(config.data_dir / "feature_store.pkl", "rb") as file:
            store = pickle.load(file)  # our own file, written by `fraudstream train`
        return cls(Registry(config).champion(), store)

    def now(self) -> datetime:
        return EPOCH + timedelta(seconds=self.store.now)

    def score(self, payment: Payment) -> Score:
        if payment.customer_id >= len(self.store.customer_xy):
            raise HTTPException(404, f"Unknown customer {payment.customer_id}")
        if payment.terminal_id >= len(self.store.terminal_xy):
            raise HTTPException(404, f"Unknown terminal {payment.terminal_id}")
        with self._lock:
            t = (payment.timestamp - EPOCH).total_seconds() if payment.timestamp else self.store.now
            tx_id = payment.transaction_id or self._new_id()
            row = self.store.transaction(
                tx_id, t, payment.customer_id, payment.terminal_id, payment.amount
            )
        features = pd.DataFrame([row], columns=FEATURES)
        probability = float(self.bundle.score(features)[0])
        alert = probability >= self.bundle.threshold
        return Score(
            transaction_id=tx_id,
            fraud_probability=round(probability, 5),
            alert=alert,
            threshold=round(self.bundle.threshold, 5),
            model_version=self.bundle.version,
            reasons=self.bundle.reasons(features)[0] if alert else [],
        )

    def feedback(self, feedback: Feedback) -> None:
        with self._lock:
            if feedback.transaction_id not in self.store.awaiting_label:
                raise HTTPException(404, f"Unknown transaction {feedback.transaction_id}")
            self.store.label(feedback.transaction_id, int(feedback.is_fraud))

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id


def create_app(service_factory: Callable[[], ScoringService] | None = None) -> FastAPI:
    metrics = CollectorRegistry()
    scored = Counter("fraudstream_scored_total", "Payments scored", registry=metrics)
    alerts = Counter("fraudstream_alerts_total", "Fraud alerts raised", registry=metrics)
    latency = Histogram("fraudstream_score_seconds", "Scoring latency", registry=metrics)
    version = Gauge("fraudstream_model_version", "Champion model version", registry=metrics)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        factory = service_factory or (lambda: ScoringService.from_config(Config.from_env()))
        app.state.service = factory()
        if app.state.service.bundle.version.isdigit():
            version.set(int(app.state.service.bundle.version))
        yield

    app = FastAPI(
        title="FraudStream scoring API",
        description="Real-time card-fraud scoring with explanations.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict:
        service: ScoringService = request.app.state.service
        return {
            "status": "ok",
            "model_version": service.bundle.version,
            "simulated_time": service.now().isoformat(),
        }

    @app.post("/v1/score", response_model=Score)
    def score(payment: Payment, request: Request) -> Score:
        started = time.perf_counter()
        result = request.app.state.service.score(payment)
        latency.observe(time.perf_counter() - started)
        scored.inc()
        if result.alert:
            alerts.inc()
        return result

    @app.post("/v1/feedback")
    def feedback(body: Feedback, request: Request) -> dict:
        request.app.state.service.feedback(body)
        return {"status": "recorded"}

    @app.get("/metrics")
    def prometheus() -> Response:
        return Response(generate_latest(metrics), media_type=CONTENT_TYPE_LATEST)

    return app
