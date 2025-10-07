"""Streamlit entry point for the Spending Analyser app."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from src import features, insights, summarize, synth, utils, viz


@st.cache_data(show_spinner=False)
def _load_transactions(rows: int, seed: int) -> pd.DataFrame:
    return synth.generate_sample_transactions(rows=rows, seed=seed)


@st.cache_data(show_spinner=False)
def _load_full_dataset(seed: int) -> pd.DataFrame:
    return synth.generate_transactions(rows=synth.DEFAULT_DATASET_ROWS, seed=seed)


def main() -> None:
    """Render the Spending Analyser Streamlit application."""

    st.set_page_config(
        page_title="Spending Analyser",
        page_icon="💸",
        layout="wide",
    )

    st.title("Spending Analyser")
    st.caption("Stage 2: richer insights, KPIs, and anomaly surfacing")

    sidebar = st.sidebar
    sidebar.header("Synthetic data controls")
    seed = int(
        sidebar.number_input(
            "Random seed", value=synth.DEFAULT_SEED, min_value=0, step=1, help="Deterministic RNG seed"
        )
    )
    sample_rows = int(
        sidebar.slider(
            "Sample rows",
            min_value=50,
            max_value=1000,
            value=synth.DEFAULT_SAMPLE_ROWS,
            step=50,
            help="Number of rows to preview in the dashboard",
        )
    )

    transactions = _load_transactions(sample_rows, seed)
    enriched = features.add_engineered_features(transactions)
    insight_payload = insights.calculate_kpis(enriched)
    summary = summarize.summarize_spending(enriched)

    dataset_meta = (
        f"{len(enriched):,} transactions | {enriched['posted_date'].min().date()}"
        f" → {enriched['posted_date'].max().date()}"
    )
    st.caption(f"Sample preview: {dataset_meta}")

    st.subheader("Key metrics")
    summary_metrics = insight_payload["summary"]
    summary_cards = [
        ("Total spend", utils.format_currency(summary_metrics["total_spend"])),
        ("Total income", utils.format_currency(summary_metrics["total_income"])),
        ("Net cashflow", utils.format_currency(summary_metrics["net_cashflow"])),
        ("Avg. daily spend", utils.format_currency(summary_metrics["avg_daily_spend"])),
        (
            "Recurring share",
            f"{summary_metrics['recurring_share_pct']:.1f}%",
        ),
    ]
    metric_columns = st.columns(len(summary_cards))
    for column, (label, value) in zip(metric_columns, summary_cards, strict=False):
        column.metric(label=label, value=value)

    st.subheader("Monthly balance")
    monthly_balance = pd.DataFrame(insight_payload["monthly_balance"])
    if not monthly_balance.empty:
        st.dataframe(
            monthly_balance.set_index("month"),
            width="stretch",
            use_container_width=True,
        )
    else:
        st.write("No historic balance available yet.")

    st.subheader("Top categories & merchants")

    window_labels = {
        "current_month": "Current month",
        "last_3_months": "Last 3 months",
        "last_6_months": "Last 6 months",
        "last_12_months": "Last 12 months",
    }

    def _render_breakdown(
        breakdown: dict[str, list[insights.BreakdownEntry]]
    ) -> None:
        keys = list(breakdown.keys())
        tabs = st.tabs([window_labels.get(key, key.replace("_", " ").title()) for key in keys])
        for tab, key in zip(tabs, keys, strict=False):
            with tab:
                entries = breakdown[key]
                if not entries:
                    st.write("No data available.")
                    continue
                frame = pd.DataFrame(entries)
                frame["amount"] = frame["amount"].apply(utils.format_currency)
                frame["share"] = frame["share"].apply(lambda value: f"{value * 100:.1f}%")
                st.dataframe(frame, hide_index=True, width="stretch")

    categories_tab, merchants_tab = st.tabs(["Categories", "Merchants"])
    with categories_tab:
        _render_breakdown(insight_payload["top_categories"])
    with merchants_tab:
        _render_breakdown(insight_payload["top_merchants"])

    st.subheader("Spend composition")
    split = insight_payload["spend_split"]
    fixed_total = max(split["fixed"] + split["variable"], 1.0)
    essentials_total = max(split["essentials"] + split["discretionary"], 1.0)

    fixed_cols = st.columns(2)
    for column, label, amount, total in (
        (fixed_cols[0], "Fixed", split["fixed"], fixed_total),
        (fixed_cols[1], "Variable", split["variable"], fixed_total),
    ):
        column.metric(
            label=label,
            value=utils.format_currency(amount),
            delta=f"{(amount / total) * 100:.1f}% of spend",
        )

    essentials_cols = st.columns(2)
    for column, label, amount, total in (
        (essentials_cols[0], "Essentials", split["essentials"], essentials_total),
        (essentials_cols[1], "Discretionary", split["discretionary"], essentials_total),
    ):
        column.metric(
            label=label,
            value=utils.format_currency(amount),
            delta=f"{(amount / total) * 100:.1f}% of spend",
        )

    st.subheader("Cash projection")
    projection = insight_payload["cash_projection"]
    projection_columns = st.columns(4)
    projection_columns[0].metric(
        "As of",
        projection["as_of"],
    )
    projection_columns[1].metric(
        "Projected month-end balance",
        utils.format_currency(projection["projected_end_balance"]),
    )
    projection_columns[2].metric(
        "Projected additional spend",
        utils.format_currency(projection["projected_burn"]),
    )
    projection_columns[3].metric(
        "Daily burn rate",
        utils.format_currency(projection["daily_burn_rate"]),
        delta=f"{projection['days_remaining']} days remaining",
    )

    st.subheader("Next payday")
    payday = insight_payload["payday"]
    payday_columns = st.columns(2)
    payday_columns[0].metric("Next payday", payday.get("next_payday", "–"))
    days_to_payday = payday.get("days_to_next_payday")
    payday_columns[1].metric(
        "Days to next payday",
        f"{days_to_payday} days" if days_to_payday is not None else "–",
    )

    st.subheader("Recurring payments health check")
    recurring = pd.DataFrame(insight_payload["recurring"])
    if not recurring.empty:
        for column in ["avg_amount", "last_amount", "delta_amount"]:
            recurring[column] = recurring[column].apply(utils.format_currency)
        recurring["delta_percent"] = recurring["delta_percent"].apply(
            lambda value: f"{value:+.1f}%"
        )
        st.dataframe(recurring, hide_index=True, width="stretch")
    else:
        st.write("No recurring payments detected.")

    st.subheader("Potential anomalies")
    anomalies = pd.DataFrame(insight_payload["anomalies"])
    if not anomalies.empty:
        anomalies["amount"] = anomalies["amount"].apply(utils.format_currency)
        anomalies["z_score"] = anomalies["z_score"].apply(lambda value: f"{value:.2f}")
        st.dataframe(anomalies, hide_index=True, width="stretch")
    else:
        st.write("No strong outliers found.")

    st.subheader("Spend by category (lifetime)")
    category_breakdown = pd.DataFrame(insight_payload["category_spend"])
    if not category_breakdown.empty:
        category_breakdown["amount"] = category_breakdown["amount"].round(2)
        st.bar_chart(category_breakdown.set_index("name")["amount"])
    else:
        st.write("No category spend available yet.")

    st.subheader("Spending over time")
    chart = viz.plot_spending_over_time(enriched)
    st.plotly_chart(chart, width="stretch")

    st.subheader("Sample transactions")
    st.dataframe(enriched.head(200), width="stretch")

    sidebar.subheader("Exports")
    sample_csv = enriched.to_csv(index=False)
    sidebar.download_button(
        "Download sample CSV",
        data=sample_csv,
        file_name="synthetic_transactions_sample.csv",
        mime="text/csv",
    )

    full_df = _load_full_dataset(seed)
    full_csv = full_df.to_csv(index=False)
    sidebar.download_button(
        "Download full 10k CSV",
        data=full_csv,
        file_name="synthetic_transactions.csv",
        mime="text/csv",
    )

    st.subheader("AI summary")
    st.info(summary)


if __name__ == "__main__":
    main()
