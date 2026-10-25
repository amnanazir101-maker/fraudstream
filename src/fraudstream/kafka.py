"""Kafka transport: the simulator publishes payment and label events, the scorer consumes them.

All events share one single-partition topic so that their time order is preserved.

    fraudstream produce --bootstrap localhost:9092
    fraudstream consume --bootstrap localhost:9092
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator

TOPIC = "fraudstream.events"
END = {"type": "end"}


def ensure_topic(bootstrap: str, topic: str = TOPIC) -> None:
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap})
    if topic not in admin.list_topics(timeout=30).topics:
        futures = admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])
        for future in futures.values():
            try:
                future.result(timeout=30)
            except Exception as exc:  # created by a concurrent producer
                if "TOPIC_ALREADY_EXISTS" not in str(exc):
                    raise


def publish(events: Iterable[dict], bootstrap: str, topic: str = TOPIC) -> int:
    """Send events in order, then an end marker. Returns the number of events sent."""
    from confluent_kafka import Producer

    ensure_topic(bootstrap, topic)
    producer = Producer({"bootstrap.servers": bootstrap, "linger.ms": 20})
    sent = 0
    for event in events:
        producer.produce(topic, value=json.dumps(event).encode())
        producer.poll(0)
        sent += 1
    producer.produce(topic, value=json.dumps(END).encode())
    producer.flush(30)
    return sent


def consume(
    bootstrap: str,
    topic: str = TOPIC,
    group: str = "fraudstream-scorer",
    idle_timeout: float | None = None,
) -> Iterator[dict]:
    """Yield events until the end marker (or `idle_timeout` seconds without a message)."""
    from confluent_kafka import Consumer

    ensure_topic(bootstrap, topic)
    consumer = Consumer(
        {"bootstrap.servers": bootstrap, "group.id": group, "auto.offset.reset": "earliest"}
    )
    consumer.subscribe([topic])
    idle = 0.0
    try:
        while True:
            message = consumer.poll(1.0)
            if message is None:
                idle += 1.0
                if idle_timeout is not None and idle >= idle_timeout:
                    return
                continue
            if message.error():
                raise RuntimeError(message.error())
            idle = 0.0
            event = json.loads(message.value())
            if event == END:
                return
            yield event
    finally:
        consumer.close()


def run(handle: Callable[[dict], None], bootstrap: str) -> int:
    count = 0
    for event in consume(bootstrap):
        handle(event)
        count += 1
    return count
