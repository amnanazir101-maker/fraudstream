"""The `fraudstream` command: simulate, train, stream, serve, benchmark, produce, consume, demo."""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings

from rich.console import Console
from rich.table import Table

from fraudstream.config import Config

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fraudstream",
        description="Real-time card-fraud detection with drift monitoring and retraining.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("simulate", help="create the simulated world and its payment history")
    commands.add_parser("train", help="build features, train and register the first model")
    stream = commands.add_parser("stream", help="score the live stream, monitor and retrain")
    stream.add_argument(
        "--day-seconds", type=float, default=3.0,
        help="real seconds per simulated day (0 = as fast as possible)",
    )  # fmt: skip
    serve = commands.add_parser("serve", help="start the scoring API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    commands.add_parser("benchmark", help="evaluate on the real ULB credit-card dataset")
    produce = commands.add_parser("produce", help="publish the live stream to Kafka")
    produce.add_argument("--bootstrap", default="localhost:9092")
    produce.add_argument("--day-seconds", type=float, default=3.0)
    consume = commands.add_parser("consume", help="score events from Kafka")
    consume.add_argument("--bootstrap", default="localhost:9092")
    demo = commands.add_parser("demo", help="simulate, train and stream in one go")
    demo.add_argument("--day-seconds", type=float, default=3.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    warnings.filterwarnings("ignore")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    config = Config.from_env()

    if args.command in ("simulate", "demo"):
        simulate(config)
    if args.command in ("train", "demo"):
        train(config)
    if args.command in ("stream", "demo"):
        stream_local(config, args.day_seconds)
    if args.command == "serve":
        import uvicorn

        from fraudstream.api import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)
    if args.command == "benchmark":
        benchmark(config)
    if args.command == "produce":
        from fraudstream.kafka import publish
        from fraudstream.streaming import live_events, paced

        sent = publish(paced(live_events(config), args.day_seconds), args.bootstrap)
        console.print(f"Published {sent:,} events to Kafka.")
    if args.command == "consume":
        stream_kafka(config, args.bootstrap)
    return 0


def simulate(config: Config) -> None:
    from fraudstream.pipeline import simulate_history

    started = time.perf_counter()
    with console.status("Simulating card holders, terminals and payments..."):
        history = simulate_history(config)
    console.print(
        f"[green]>[/green] Simulated {config.customers:,} card holders, {config.terminals:,} "
        f"terminals and {len(history):,} payments over {config.history_days} days "
        f"({history['is_fraud'].mean():.2%} fraud) in {time.perf_counter() - started:.0f} s."
    )


def train(config: Config) -> None:
    from fraudstream.pipeline import build_features, train_initial

    with console.status("Building point-in-time features..."):
        table = build_features(config)
    console.print(f"[green]>[/green] Built features for {len(table):,} payments.")
    with console.status("Training LightGBM and a logistic-regression baseline..."):
        report = train_initial(config, table)
    results = Table(title=f"Test days {report['split']['test_days']} (never seen in training)")
    results.add_column("Model")
    for metric in ("PR-AUC", "ROC-AUC", "Precision", "Recall"):
        results.add_column(metric, justify="right")
    for name, key in (("LightGBM", "lightgbm"), ("Logistic regression", "logistic_regression")):
        m = report[key]
        results.add_row(name, f"{m['pr_auc']:.3f}", f"{m['roc_auc']:.3f}",
                        f"{m['precision']:.1%}", f"{m['recall']:.1%}")  # fmt: skip
    console.print(results)
    console.print(
        f"[green]>[/green] Registered model version {report['model_version']} in MLflow as "
        f"'{config.model_name}@champion'."
    )


def _print_day(day: int, info: dict) -> None:
    drift = f"[red]drift: {info['drift']}[/red]" if info["drift"] else "no drift"
    console.print(
        f"Day {day} | {info['transactions']:,} payments | {info['alerts']} alerts | "
        f"model v{info['model_version']} | {drift}"
    )


def _print_event(day: int, kind: str, message: str) -> None:
    color = {"drift": "red", "retrain": "yellow", "promoted": "green"}.get(kind, "white")
    console.print(f"[bold {color}]  * Day {day}: {message}[/bold {color}]")


def stream_local(config: Config, day_seconds: float) -> None:
    from fraudstream.store import EventLog
    from fraudstream.streaming import live_events, load_processor, paced

    events_log = EventLog(config.data_dir / "stream.db", reset=True, on_event=_print_event)
    # A simulated day lasts a few seconds here, while retraining takes tens of seconds, so the
    # simulated clock pauses during retraining (in production a retrain is a tiny part of a day).
    processor = load_processor(config, events_log, wait_for_retraining=True)
    processor.on_day_closed = _print_day
    console.print(
        f"Streaming days {config.history_days}-{config.history_days + config.live_days - 1} "
        f"from the go-live model v{processor.champion.version}; a card-testing attack starts "
        f"on day {config.attack_day}. Open the dashboard with: streamlit run dashboard.py"
    )
    for event in paced(live_events(config), day_seconds):
        processor.handle(event)
    processor.close()
    _print_summary(config)


def stream_kafka(config: Config, bootstrap: str) -> None:
    from fraudstream.kafka import run
    from fraudstream.store import EventLog
    from fraudstream.streaming import load_processor

    events_log = EventLog(config.data_dir / "stream.db", reset=True, on_event=_print_event)
    processor = load_processor(config, events_log, wait_for_retraining=True)  # Kafka buffers
    processor.on_day_closed = _print_day
    console.print(f"Waiting for events on Kafka ({bootstrap})...")
    run(processor.handle, bootstrap)
    processor.close()
    _print_summary(config)


def _print_summary(config: Config) -> None:
    import json

    from fraudstream.streaming import summarize

    summary = summarize(config.data_dir / "stream.db")
    if not summary:
        return
    (config.data_dir / "stream_summary.json").write_text(json.dumps(summary, indent=2))
    table = Table(title="Share of fraud caught: self-retraining system vs frozen first model")
    table.add_column("Period")
    table.add_column("Days")
    for column in ("All fraud\nretraining", "All fraud\nfrozen", "Card testing\nretraining",
                   "Card testing\nfrozen"):  # fmt: skip
        table.add_column(column, justify="right")

    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value:.0%}"

    for name, p in summary["periods"].items():
        table.add_row(name, p["days"], pct(p["recall"]), pct(p["recall_frozen"]),
                      pct(p["attack_recall"]), pct(p["attack_recall_frozen"]))  # fmt: skip
    console.print(table)
    console.print(
        f"Attack started on day {summary['attack_day']}, drift flagged on day "
        f"{summary['drift_detected_day']}, retrained model live from day "
        f"{summary['retrained_day']}; "
        f"{summary['false_drift_alarms']} false drift alarms before the attack."
    )


def benchmark(config: Config) -> None:
    from fraudstream.benchmark import run_benchmark

    with console.status("Downloading (73 MB, first run only) and training on real data..."):
        report = run_benchmark(config)
    table = Table(title=f"{report['dataset']}: test set with {report['test_frauds']} frauds")
    table.add_column("Model")
    for metric in ("PR-AUC", "ROC-AUC", "Precision", "Recall"):
        table.add_column(metric, justify="right")
    for name, key in (("LightGBM", "lightgbm"), ("Logistic regression", "logistic_regression")):
        m = report[key]
        table.add_row(name, f"{m['pr_auc']:.3f}", f"{m['roc_auc']:.3f}",
                      f"{m['precision']:.1%}", f"{m['recall']:.1%}")  # fmt: skip
    console.print(table)


if __name__ == "__main__":
    raise SystemExit(main())
