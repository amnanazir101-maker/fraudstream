"""Round trip through a real Kafka broker. Runs in CI against a Kafka service container."""

import os
import time
import uuid

import pytest

pytestmark = [
    pytest.mark.kafka,
    pytest.mark.skipif(not os.getenv("KAFKA_BOOTSTRAP"), reason="set KAFKA_BOOTSTRAP to run"),
]


def wait_for_broker(bootstrap: str, seconds: float = 90) -> None:
    from confluent_kafka.admin import AdminClient

    admin = AdminClient({"bootstrap.servers": bootstrap})
    deadline = time.monotonic() + seconds
    while True:
        try:
            admin.list_topics(timeout=5)
            return
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)


def test_events_keep_their_order_through_kafka():
    from fraudstream.kafka import consume, publish

    bootstrap = os.environ["KAFKA_BOOTSTRAP"]
    wait_for_broker(bootstrap)
    topic = f"fraudstream-test-{uuid.uuid4().hex[:8]}"
    events = [
        {"type": "transaction", "tx_id": i, "t": float(i), "customer_id": i % 7,
         "terminal_id": i % 11, "amount": 10.0 + i}
        for i in range(200)
    ]  # fmt: skip
    assert publish(events, bootstrap, topic) == 200
    received = list(
        consume(bootstrap, topic, group=f"test-{uuid.uuid4().hex[:6]}", idle_timeout=60)
    )
    assert received == events
