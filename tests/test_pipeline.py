"""Training, registry, live stream with retraining, the scoring API and the dashboard."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fraudstream.api import ScoringService, create_app
from fraudstream.features import FEATURES
from fraudstream.registry import Registry
from fraudstream.store import EventLog, read_table
from fraudstream.streaming import live_events, load_processor, summarize

ROOT = Path(__file__).parents[1]


def test_training_reports_both_models_and_registers_the_champion(trained):
    report, config = trained["report"], trained["config"]
    for model in ("lightgbm", "logistic_regression"):
        assert 0 < report[model]["pr_auc"] <= 1
    assert report["lightgbm"]["pr_auc"] > report["logistic_regression"]["pr_auc"]

    champion = Registry(config).champion()
    assert champion.version == report["model_version"]
    assert 0 < champion.threshold < 1
    sample = trained["table"].tail(200)
    scores = champion.score(sample)
    assert scores.shape == (200,) and ((scores >= 0) & (scores <= 1)).all()


def test_explanations_name_real_features(trained):
    champion = Registry(trained["config"]).champion()
    frauds = trained["table"].query("is_fraud == 1").head(5)
    reasons = champion.reasons(frauds)
    assert len(reasons) == 5
    assert all(r["feature"] in FEATURES and r["impact"] > 0 for row in reasons for r in row)


def test_api_scores_payments_and_takes_feedback(trained):
    service = ScoringService.from_config(trained["config"])
    with TestClient(create_app(lambda: service)) as client:
        assert client.get("/health").json()["status"] == "ok"

        normal = client.post("/v1/score", json={"customer_id": 1, "terminal_id": 2, "amount": 20})
        assert normal.status_code == 200
        assert 0 <= normal.json()["fraud_probability"] <= 1

        huge = client.post("/v1/score", json={"customer_id": 1, "terminal_id": 2, "amount": 4000})
        assert huge.json()["alert"] is True
        assert huge.json()["reasons"][0]["feature"] in FEATURES

        tx = huge.json()["transaction_id"]
        assert (
            client.post("/v1/feedback", json={"transaction_id": tx, "is_fraud": True}).status_code
            == 200
        )
        assert (
            client.post(
                "/v1/score", json={"customer_id": 10**6, "terminal_id": 2, "amount": 5}
            ).status_code
            == 404
        )
        assert "fraudstream_scored_total 2.0" in client.get("/metrics").text


@pytest.fixture(scope="module")
def streamed(trained) -> Path:
    """Run the whole live stream once; tests read its event log."""
    config = trained["config"]
    path = config.data_dir / "stream.db"
    events_log = EventLog(path, reset=True)
    processor = load_processor(config, events_log, wait_for_retraining=True)
    for event in live_events(config):
        processor.handle(event)
    processor.close()
    events_log.close()
    return path


def test_stream_detects_the_attack_and_retrains(trained, streamed):
    config, path = trained["config"], streamed
    summary = summarize(path)
    assert summary["false_drift_alarms"] == 0
    assert summary["drift_detected_day"] == config.attack_day  # flagged on the first day
    assert summary["retrained_day"] is not None
    after = summary["periods"]["after retraining"]
    assert after["attack_recall"] > after["attack_recall_frozen"] + 0.3

    days = read_table(path, "SELECT * FROM days ORDER BY day")
    assert len(days) == config.live_days
    assert days["frauds"].notna().all()  # every day was evaluated once its labels arrived
    alerts = read_table(path, "SELECT * FROM alerts")
    assert len(alerts) and alerts["is_fraud"].notna().any()
    live_after = days.loc[days["day"] >= summary["retrained_day"], "model_version"]
    assert (live_after != "1").all()  # the retrained model served from then on


def test_dashboard_renders_the_stream(trained, streamed, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("FRAUDSTREAM_DATA_DIR", str(trained["config"].data_dir))
    app = testing.AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=60).run()
    assert not app.exception
    assert [metric.label for metric in app.metric][:2] == ["Simulated day", "Payments scored"]
    assert any("promoted" in block.value for block in app.markdown)

    app.toggle[0].set_value(True).run()  # the card-testing-only view renders too
    assert not app.exception
