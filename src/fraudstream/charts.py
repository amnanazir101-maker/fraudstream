"""Altair charts for the dashboard (kept here so they can be tested and rendered on their own).

Colors come from a validated reference palette: one accent hue for the system the story is
about and gray for the frozen comparison model (an emphasis chart, not a categorical one).
The status red is reserved for drift alarms.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

TOKENS = {
    "light": {"accent": "#2a78d6", "context": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "muted": "#898781", "ink": "#0b0b0b", "surface": "#fcfcfb", "critical": "#d03b3b"},
    "dark": {"accent": "#3987e5", "context": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "muted": "#898781", "ink": "#f0efec", "surface": "#1a1a19", "critical": "#d03b3b"},
}  # fmt: skip
ADAPTIVE, FROZEN = "Self-retraining system", "Frozen first model"


def style(chart: alt.TopLevelMixin, t: dict[str, str]) -> alt.TopLevelMixin:
    """Recessive hairline axes and grid; text in ink tokens, never in series colors."""
    return (
        chart.configure_axis(
            gridColor=t["grid"], gridWidth=1, domainColor=t["axis"], tickColor=t["axis"],
            labelColor=t["muted"], titleColor=t["muted"], labelFontSize=12, titleFontSize=12,
        )
        .configure_view(strokeWidth=0)
        .configure_legend(labelColor=t["ink"], titleColor=t["muted"], orient="top")
    )  # fmt: skip


def recall_chart(
    days: pd.DataFrame, events: pd.DataFrame, attack_only: bool, t: dict[str, str]
) -> alt.LayerChart:
    """Share of fraud caught per day, live system vs the frozen first model."""
    caught, frauds = ("attack_caught", "attack_frauds") if attack_only else ("caught", "frauds")
    rows = days.dropna(subset=["frauds"])
    rows = rows[rows[frauds] > 0]
    # The frozen model first, so the live system's line is drawn on top where they coincide.
    data = pd.concat(
        [
            pd.DataFrame({"day": rows["day"], "model": FROZEN, "frauds": rows[frauds],
                          "recall": rows[f"{caught}_frozen"] / rows[frauds]}),
            pd.DataFrame({"day": rows["day"], "model": ADAPTIVE,
                          "recall": rows[caught] / rows[frauds], "frauds": rows[frauds]}),
        ]
    )  # fmt: skip
    color = alt.Color(
        "model:N",
        scale=alt.Scale(domain=[ADAPTIVE, FROZEN], range=[t["accent"], t["context"]]),
        legend=alt.Legend(title=None, orient="bottom", direction="horizontal"),
    )
    first, last = int(days["day"].min()), int(days["day"].max())
    x = alt.X(
        "day:Q",
        title="Simulated day",
        scale=alt.Scale(domain=[first, last], nice=False),
        axis=alt.Axis(format="d", tickMinStep=1),
    )
    y = alt.Y("recall:Q", title="Fraud caught", scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(format="%", tickCount=5))  # fmt: skip
    hover = alt.selection_point(fields=["day"], nearest=True, on="pointerover", empty=False)

    lines = (
        alt.Chart(data)
        .mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round")
        .encode(x=x, y=y, color=color)
    )
    ends = (
        alt.Chart(data[data["day"] == data["day"].max()])
        .mark_text(align="left", dx=8, fontSize=12, color=t["ink"])
        .encode(x=x, y=y, text=alt.Text("recall:Q", format=".0%"))
    )
    rule = (
        alt.Chart(data)
        .mark_rule(color=t["muted"], strokeWidth=1)
        .encode(x=x, opacity=alt.condition(hover, alt.value(1), alt.value(0)))
        .add_params(hover)
    )
    points = (
        alt.Chart(data)
        .mark_point(filled=True, size=70, stroke=t["surface"], strokeWidth=2)
        .encode(
            x=x, y=y, color=color,
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip("day:Q", title="Day"), alt.Tooltip("model:N", title="Model"),
                     alt.Tooltip("recall:Q", title="Fraud caught", format=".0%"),
                     alt.Tooltip("frauds:Q", title="Frauds that day")],
        )
    )  # fmt: skip
    layers = [lines, rule, points, ends]
    markers = event_markers(events, x, t)
    if markers is not None:
        layers.insert(0, markers)
    return alt.layer(*layers).properties(height=280)


def event_markers(events: pd.DataFrame, x: alt.X, t: dict[str, str]) -> alt.LayerChart | None:
    """Thin vertical rules where the attack started and where a new model went live.

    The attack label sits left of its rule and the model label right of its rule, so the two
    never collide even when the events are a few days apart.
    """
    marks = events[events["kind"].isin(["attack", "promoted"])].copy() if len(events) else events
    if marks.empty:
        return None
    marks["label"] = marks["kind"].map({"attack": "Attack starts", "promoted": "New model live"})
    base = alt.Chart(marks).encode(x=x)
    layers = [base.mark_rule(color=t["muted"], strokeWidth=1)]
    for kind, align, dx in (("attack", "right", -4), ("promoted", "left", 4)):
        layers.append(
            base.transform_filter(alt.datum.kind == kind)
            .mark_text(align=align, dx=dx, y=10, fontSize=11, color=t["muted"])
            .encode(text="label:N")
        )
    return alt.layer(*layers)


def drift_charts(
    days: pd.DataFrame,
    t: dict[str, str],
    novelty_threshold: float = 0.005,
    psi_threshold: float = 0.2,
) -> list[alt.LayerChart]:
    """The two drift tests as small multiples, each against its own alarm line.

    First: share of payments outside the training range. Second: worst-feature PSI.
    Bars above their alarm line are drawn in the status red.
    """
    data = days.assign(reason=days["drift_reason"].fillna("").replace("", "-"))
    tooltip = [alt.Tooltip("day:O", title="Day"),
               alt.Tooltip("max_novelty:Q", title="Outside training range", format=".1%"),
               alt.Tooltip("max_psi:Q", title="Worst PSI", format=".2f"),
               alt.Tooltip("drift_feature:N", title="Most-shifted feature"),
               alt.Tooltip("reason:N", title="Alarm")]  # fmt: skip
    x = alt.X("day:O", title="Simulated day", axis=alt.Axis(labelAngle=0, labelOverlap=True))

    def panel(field: str, title: str, threshold: float, fmt: str, label: str) -> alt.LayerChart:
        top = max(float(data[field].max()), threshold) * 1.15
        bars = (
            alt.Chart(data)
            .mark_bar(size=12, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=x,
                y=alt.Y(f"{field}:Q", title=title, scale=alt.Scale(domain=[0, top]),
                        axis=alt.Axis(format=fmt, tickCount=4)),
                color=alt.condition(
                    alt.datum[field] >= threshold, alt.value(t["critical"]), alt.value(t["accent"])
                ),
                tooltip=tooltip,
            )
        )  # fmt: skip
        limit = pd.DataFrame({"y": [threshold], "day": [data["day"].max()], "label": [label]})
        rule = alt.Chart(limit).mark_rule(color=t["muted"], strokeWidth=1).encode(y="y:Q")
        text = (
            alt.Chart(limit)
            .mark_text(align="right", dy=-6, fontSize=11, color=t["ink"])
            .encode(x=x, y="y:Q", text="label:N")
        )
        return alt.layer(bars, rule, text).properties(height=120)

    novelty = panel(
        "max_novelty", "Outside training range", novelty_threshold, ".1%",
        f"alarm at {novelty_threshold:.1%}",
    )  # fmt: skip
    psi = panel("max_psi", "Worst-feature PSI", psi_threshold, ".2f", f"alarm at {psi_threshold}")
    return [novelty, psi]
