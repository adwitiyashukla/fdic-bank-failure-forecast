import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DEMO = Path(__file__).resolve().parent / "demo"

SURFACE = "#1a1a19"
INK = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
BLUE = "#3987e5"
ORANGE = "#d95926"
AQUA = "#199e70"

MODEL_NAMES = {"gbm": "gradient boosting", "logit": "logistic model", "texas": "Texas ratio"}
MODEL_COLORS = alt.Scale(domain=list(MODEL_NAMES.values()), range=[BLUE, ORANGE, AQUA])
DESIGN_NAMES = {
    "crisis": "Crisis test: trained through 2005, scored on reports from 2006Q4 to 2011Q4",
    "recent": "Recent test: trained through 2013, scored on reports from 2014Q4 to 2025Q2",
}
METRIC_COLUMNS = {
    "model": "model",
    "auc": "AUC",
    "auc_low": "AUC low",
    "auc_high": "AUC high",
    "average_precision": "average precision",
    "recall_top_0.01": "caught in top 1%",
    "recall_top_0.05": "caught in top 5%",
    "precision_top_0.05": "precision in top 5%",
    "median_warning_quarters": "median warning quarters",
}

st.set_page_config(page_title="FDIC bank failure forecast", layout="wide")


@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(DEMO / f"{name}.csv", keep_default_na=False, na_values=[""])


@st.cache_data(show_spinner=False)
def summary() -> dict:
    return json.loads((DEMO / "panel_summary.json").read_text())


def styled(chart: alt.Chart, height: int = 320) -> alt.Chart:
    return (
        chart.properties(height=height, background=SURFACE)
        .configure_axis(
            labelColor=INK,
            titleColor=MUTED,
            gridColor=GRID,
            domainColor=BASELINE,
            tickColor=BASELINE,
            labelLimit=320,
        )
        .configure_legend(labelColor=INK, titleColor=MUTED, orient="top")
        .configure_view(strokeOpacity=0)
    )


def show(chart: alt.Chart, height: int = 320) -> None:
    st.altair_chart(styled(chart, height), width="stretch")


def grid(frame: pd.DataFrame, columns: dict[str, str]) -> None:
    st.dataframe(frame[list(columns)].rename(columns=columns), hide_index=True, width="stretch")


def named_models(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["model"] = out["model"].map(MODEL_NAMES)
    return out


def percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def model_lines(frame: pd.DataFrame, x: alt.X, y: alt.Y, tooltip: list) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2)
        .encode(
            x=x, y=y, color=alt.Color("model:N", scale=MODEL_COLORS, title=None), tooltip=tooltip
        )
    )


def overview() -> None:
    info = summary()
    metrics = table("design_metrics").set_index(["design", "model"])
    crisis = metrics.loc[("crisis", "gbm")]
    recent = metrics.loc[("recent", "gbm")]
    case = table("case2023")
    svb = case[(case["CERT"] == 24735) & (case["as_of"] == case["as_of"].max())]
    now = table("current_summary").iloc[0]
    st.markdown(
        f"Every FDIC insured bank, every quarter from {info['first_quarter']} to "
        f"{info['last_quarter']}. That is {info['bank_quarters']:,} bank quarters and "
        f"{info['events_matched']:,} failures. Each model gives every bank a probability of "
        "failing within the next 12 months, using only what its call report showed that quarter."
    )
    cols = st.columns(4)
    cols[0].metric("Crisis test AUC", f"{crisis['auc']:.3f}", border=True)
    cols[1].metric(
        "Crisis failures caught in the top 5%", percent(crisis["recall_top_0.05"]), border=True
    )
    cols[2].metric("Recent test AUC", f"{recent['auc']:.3f}", border=True)
    if not svb.empty:
        row = svb.iloc[0]
        cols[3].metric(
            f"SVB rank at {row['as_of']}",
            f"{int(row['gbm_rank']):,} of {int(row['banks_ranked']):,}",
            border=True,
        )
    st.caption(
        f"Latest quarter {now['as_of']}: {int(now['banks']):,} banks, "
        f"{now['expected_failures']:.1f} failures expected over the next year "
        f"(90% range {now['interval_low']:.0f} to {now['interval_high']:.0f})."
    )
    rolling = named_models(table("rolling"))
    st.subheader("Out of time AUC, retrained every year")
    show(
        model_lines(
            rolling,
            alt.X("year:O", title=None),
            alt.Y("auc:Q", title="AUC", scale=alt.Scale(domain=[0.3, 1.0])),
            ["year:O", "model:N", alt.Tooltip("auc:Q", format=".3f"), "failed_banks:Q"],
        )
    )
    counts = rolling[rolling["model"] == MODEL_NAMES["gbm"]]
    bars = (
        alt.Chart(counts)
        .mark_bar(color=BLUE, cornerRadiusEnd=4)
        .encode(
            x=alt.X("year:O", title=None),
            y=alt.Y("failed_banks:Q", title="failed banks"),
            tooltip=["year:O", "failed_banks:Q", "positive_rows:Q"],
        )
    )
    st.markdown("**Banks that failed within a year of each year's reports**")
    show(bars, 220)


def backtests() -> None:
    metrics = named_models(table("design_metrics"))
    choice = st.radio("Test", list(DESIGN_NAMES), format_func=DESIGN_NAMES.get)
    picked = metrics[metrics["design"] == choice]
    first = picked.iloc[0]
    st.write(
        f"{int(first['rows']):,} bank quarters, {int(first['failed_banks'])} banks that failed, "
        f"{int(first['positive_rows']):,} bank quarters within a year of a failure."
    )
    grid(picked.round(3), METRIC_COLUMNS)
    left, right = st.columns(2)
    with left:
        st.markdown("**Share of failures caught when flagging the riskiest x% of banks**")
        curve = named_models(table("design_curve"))
        curve = curve[curve["design"] == choice]
        show(
            model_lines(
                curve,
                alt.X("fraction:Q", title="share of banks flagged", axis=alt.Axis(format="%")),
                alt.Y("recall:Q", title="failures caught", axis=alt.Axis(format="%")),
                [
                    "model:N",
                    alt.Tooltip("fraction:Q", format=".1%"),
                    alt.Tooltip("recall:Q", format=".1%"),
                ],
            )
        )
    with right:
        st.markdown("**Failures the boosting model expected vs failures that happened**")
        quarterly = table("design_quarterly")
        quarterly = quarterly[quarterly["design"] == choice]
        long = quarterly.melt(
            id_vars=["quarter"],
            value_vars=["observed", "gbm"],
            var_name="series",
            value_name="count",
        )
        long["series"] = long["series"].map({"observed": "happened", "gbm": "expected"})
        chart = (
            alt.Chart(long)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("quarter:O", title=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("count:Q", title="failures within 12 months"),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(domain=["expected", "happened"], range=[BLUE, ORANGE]),
                    title=None,
                ),
                tooltip=["quarter:O", "series:N", alt.Tooltip("count:Q", format=".1f")],
            )
        )
        show(chart)
    st.markdown("**What the boosting model leans on (AUC lost when a group is shuffled)**")
    importance = table("design_importance")
    importance = importance[importance["design"] == choice]
    bars = (
        alt.Chart(importance)
        .mark_bar(color=BLUE, cornerRadiusEnd=4)
        .encode(
            x=alt.X("auc_drop:Q", title="AUC drop"),
            y=alt.Y("group:N", sort="-x", title=None),
            tooltip=["group:N", "features:Q", alt.Tooltip("auc_drop:Q", format=".4f")],
        )
    )
    show(bars, 280)


def failed_banks() -> None:
    paths = table("failed_paths")
    paths["label"] = paths["name"] + " (" + paths["fail_date"] + ")"
    labels = list(paths.sort_values("fail_date", ascending=False)["label"].unique())
    default = next((i for i, text in enumerate(labels) if text.startswith("SILICON VALLEY")), 0)
    picked = st.selectbox("Failed bank", labels, index=default)
    one = paths[paths["label"] == picked]
    long = one.melt(
        id_vars=["quarter", "quarters_to_event"],
        value_vars=["gbm_percentile", "logit_percentile", "texas_percentile"],
        var_name="model",
        value_name="percentile",
    )
    long["model"] = long["model"].str.replace("_percentile", "").map(MODEL_NAMES)
    show(
        model_lines(
            long,
            alt.X("quarter:O", title="report quarter"),
            alt.Y(
                "percentile:Q",
                title="riskier than this share of banks",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            [
                "quarter:O",
                "quarters_to_event:Q",
                "model:N",
                alt.Tooltip("percentile:Q", format=".1%"),
            ],
        )
    )
    st.caption(
        "Scores come from the yearly retrained models, so each quarter is scored by a model "
        "that never saw this failure."
    )
    st.markdown("**Warning time in the backtests (boosting model, riskiest 5% flagged)**")
    grid(
        table("design_warnings"),
        {
            "design": "test",
            "name": "bank",
            "place": "place",
            "fail_date": "failed",
            "warning_quarters": "quarters flagged without a break",
            "first_flag_quarters": "quarters since first flag",
        },
    )


def case_2023() -> None:
    case = table("case2023").sort_values(["as_of", "gbm_rank"], ascending=[False, True])
    st.write(
        "Where the models ranked the banks that failed in 2023 and 2024, using models built "
        "only from data available at the time. Rank 1 is the riskiest bank."
    )
    grid(
        case.round(1),
        {
            "as_of": "as of",
            "name": "bank",
            "fail_date": "failed",
            "banks_ranked": "banks",
            "gbm_rank": "boosting",
            "gbm_credit_rank": "boosting, credit only",
            "logit_rank": "logistic",
            "texas_rank": "Texas ratio",
            "run_rank": "run screen",
            "uninsured_deposits": "uninsured deposits %",
            "unrealized_loss_to_equity": "unrealized loss % of equity",
        },
    )
    screen = table("run_screen")
    screen["name"] = screen["name"].fillna("")
    screen["unrealized_loss_to_equity"] = screen["unrealized_loss_to_equity"].clip(-50, 300)
    others = screen[screen["name"] == ""]
    failed = screen[screen["name"] != ""]
    x = alt.X("uninsured_deposits:Q", title="uninsured deposits, % of domestic deposits")
    y = alt.Y("unrealized_loss_to_equity:Q", title="unrealized securities loss, % of equity")
    base = (
        alt.Chart(others)
        .mark_circle(size=14, color=MUTED, opacity=0.35)
        .encode(
            x=x,
            y=y,
            tooltip=[
                "size:N",
                alt.Tooltip("uninsured_deposits:Q", format=".1f"),
                alt.Tooltip("unrealized_loss_to_equity:Q", format=".1f"),
            ],
        )
    )
    marks = (
        alt.Chart(failed)
        .mark_circle(size=90, color=ORANGE, stroke=SURFACE, strokeWidth=2)
        .encode(x=x, y=y, tooltip=["name:N", "run_rank:Q", "gbm_rank:Q"])
    )
    text = (
        alt.Chart(failed)
        .mark_text(align="right", dx=-9, color=INK, fontSize=11)
        .encode(x=x, y=y, text="name:N")
    )
    st.markdown(f"**Run risk screen at {case['as_of'].max()}, every dot is a bank**")
    show(base + marks + text, 440)
    st.caption("Orange dots are banks that failed within two years. Losses are capped at 300%.")


def stress() -> None:
    scenarios = table("stress_scenarios")
    quarters = int(scenarios["horizon_quarters"].iloc[0])
    order = list(scenarios["label"])
    st.write(
        f"A second boosting model also sees how unemployment, house prices and commercial real "
        f"estate prices move over the next {quarters} quarters. Today's banks are run through "
        "the Fed's 2026 scenarios and through what really happened from 2008 on."
    )
    y = alt.Y("label:N", sort=order, title=None)
    bars = (
        alt.Chart(scenarios)
        .mark_bar(color=BLUE, cornerRadiusEnd=4)
        .encode(
            x=alt.X("expected_failures:Q", title=f"expected failures within {quarters} quarters"),
            y=y,
            tooltip=[
                "label:N",
                alt.Tooltip("expected_failures:Q", format=".1f"),
                "interval_low:Q",
                "interval_high:Q",
            ],
        )
    )
    ranges = (
        alt.Chart(scenarios)
        .mark_rule(color=INK, strokeWidth=2)
        .encode(x="interval_low:Q", x2="interval_high:Q", y=y)
    )
    show(bars + ranges, 200)
    grid(
        scenarios.round(1),
        {
            "label": "scenario",
            "path_unemployment_rise": "unemployment rise, points",
            "path_house_price_change": "house prices %",
            "path_cre_price_change": "commercial real estate prices %",
            "expected_failures": "expected failures",
            "interval_low": "90% low",
            "interval_high": "90% high",
            "banks_above_1pct": "banks above 1%",
        },
    )
    history = table("stress_backtest")
    long = history.melt(
        id_vars=["quarter"],
        value_vars=["observed", "with_path", "without_path"],
        var_name="series",
        value_name="count",
    )
    names = {
        "observed": "happened",
        "with_path": "model given the real path",
        "without_path": "model without it",
    }
    long["series"] = long["series"].map(names)
    chart = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("quarter:O", title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("count:Q", title=f"banks failing within {quarters} quarters"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(names.values()), range=[ORANGE, BLUE, AQUA]),
                title=None,
            ),
            tooltip=["quarter:O", "series:N", alt.Tooltip("count:Q", format=".1f")],
        )
    )
    first, last = history["quarter"].iloc[0], history["quarter"].iloc[-1]
    st.markdown(f"**Backtest: trained through 2004, scored on reports from {first} to {last}**")
    show(chart)


def today() -> None:
    now = table("current_summary").iloc[0]
    cols = st.columns(3)
    cols[0].metric(f"Banks at {now['as_of']}", f"{int(now['banks']):,}", border=True)
    cols[1].metric(
        "Expected failures, next 12 months", f"{now['expected_failures']:.1f}", border=True
    )
    cols[2].metric("Banks above 1%", f"{int(now['banks_above_0.01'])}", border=True)
    st.caption("No bank that is still open is named anywhere in this app, only totals.")
    histogram = table("current_histogram")
    x = alt.X("band:N", sort=list(histogram["band"]), title="12 month failure probability")
    top = float(histogram["banks"].max()) * 1.15
    y = alt.Y("banks:Q", title="banks", scale=alt.Scale(domain=[0, top]))
    bars = (
        alt.Chart(histogram)
        .mark_bar(color=BLUE, cornerRadiusEnd=4)
        .encode(x=x, y=y, tooltip=["band:N", "banks:Q"])
    )
    labels = (
        alt.Chart(histogram)
        .mark_text(dy=-8, color=INK, fontSize=12)
        .encode(x=x, y=y, text="banks:Q")
    )
    show(bars + labels, 260)
    st.markdown("**By asset size**")
    grid(
        table("current_size").round(2),
        {
            "size": "assets",
            "banks": "banks",
            "expected_failures": "expected failures",
            "above_1pct": "banks above 1%",
            "assets_billions": "total assets, $B",
        },
    )


st.title("Which US banks fail within a year?")
if not (DEMO / "panel_summary.json").is_file():
    st.error("Demo tables not found. Run the pipeline and then python -m bankfail export.")
    st.stop()
tabs = st.tabs(["Overview", "Backtests", "Failed banks", "2023", "Stress", "Today"])
pages = [overview, backtests, failed_banks, case_2023, stress, today]
for tab, page in zip(tabs, pages, strict=True):
    with tab:
        page()
