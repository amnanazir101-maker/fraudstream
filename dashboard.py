"""FraudStream live monitoring dashboard. Run with: streamlit run dashboard.py

Reads the event log written by `fraudstream stream` and refreshes every two seconds.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from fraudstream.charts import TOKENS, drift_charts, recall_chart, style
from fraudstream.config import Config
from fraudstream.store import read_status, read_table

st.set_page_config(page_title="FraudStream", page_icon="🛡️", layout="wide")
DB = Config.from_env().data_dir / "stream.db"
ICONS = {"drift": "⚠️", "retrain": "🔄", "promoted": "✅", "rejected": "↩️", "error": "❌"}


def tokens() -> dict[str, str]:
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    return TOKENS["dark" if theme == "dark" else "light"]


st.title("🛡️ FraudStream")
st.caption(
    "Live card-fraud scoring with drift monitoring and automatic retraining. "
    "Start a stream with `fraudstream stream`; this page refreshes every two seconds."
)
attack_only = st.toggle("Card-testing attack only", help="Show only the fraud of the new attack.")


@st.fragment(run_every="2s")
def live() -> None:
    status = read_status(DB)
    days = read_table(DB, "SELECT * FROM days ORDER BY day")
    if not status or days.empty:
        st.info("No stream data yet. Run `fraudstream stream` in a terminal.")
        return
    events = read_table(DB, "SELECT * FROM events ORDER BY id")
    if status["day"] >= status["attack_day"]:  # mark the attack once the stream reaches it
        attack = pd.DataFrame({"day": [status["attack_day"]], "kind": ["attack"], "message": [""]})
        events = pd.concat([attack, events], ignore_index=True)
    alerts = read_table(DB, "SELECT * FROM alerts ORDER BY t DESC LIMIT 200")
    t = tokens()

    labelled = days.dropna(subset=["frauds"]).tail(3)
    recent = (
        labelled["caught"].sum() / labelled["frauds"].sum() if labelled["frauds"].sum() else None
    )
    tiles = st.columns(5)
    tiles[0].metric("Simulated day", status["day"])
    tiles[1].metric("Payments scored", f"{status['processed']:,}")
    tiles[2].metric("Alerts raised", f"{int(days['alerts'].sum()):,}")
    tiles[3].metric(
        "Fraud caught, last 3 labelled days", "-" if recent is None else f"{recent:.0%}"
    )
    tiles[4].metric(
        "Live model", f"v{status['model_version']}",
        "retraining..." if status.get("retraining") else None, delta_color="off",
    )  # fmt: skip

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Fraud caught per day")
        st.caption("Labels arrive three days after each payment, so the last days fill in later.")
        st.altair_chart(style(recall_chart(days, events, attack_only, t), t), width="stretch")
    with right:
        st.subheader("Data drift")
        st.caption(
            "Two daily tests against the training data: payments outside anything seen in "
            "training (catches new patterns early) and the stability index (broad shifts)."
        )
        for chart in drift_charts(days, t):
            st.altair_chart(style(chart, t), width="stretch")

    log, feed = st.columns([2, 3])
    with log:
        st.subheader("What the system did")
        for row in events[events["kind"] != "attack"].itertuples():
            st.markdown(f"{ICONS.get(row.kind, '•')} **Day {row.day}:** {row.message}")
    with feed:
        st.subheader("Latest alerts")
        if alerts.empty:
            st.write("No alerts yet.")
        else:
            alerts["outcome"] = (
                alerts["is_fraud"].map({1: "Fraud", 0: "Legitimate"}).fillna("Pending")
            )
            alerts["why"] = alerts["reasons"].map(
                lambda text: ", ".join(r["feature"] for r in json.loads(text))
            )
            st.dataframe(
                alerts[["day", "customer_id", "terminal_id", "amount", "score", "why", "outcome"]],
                column_config={
                    "day": "Day", "customer_id": "Card", "terminal_id": "Terminal",
                    "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
                    "score": st.column_config.ProgressColumn(
                        "Fraud score", min_value=0, max_value=1
                    ),
                    "why": "Top reasons", "outcome": "Investigation",
                },
                hide_index=True, height=300, width="stretch",
            )  # fmt: skip

    with st.expander("Daily numbers (table view)"):
        st.dataframe(days, hide_index=True, width="stretch")


live()
