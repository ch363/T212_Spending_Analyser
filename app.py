"""Streamlit entry point for the Spending Analyser app."""

from __future__ import annotations

from datetime import date

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
    st.caption("Stage 3: interactive insights with bank-style visualisations")

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
    enriched["posted_date"] = pd.to_datetime(enriched["posted_date"])

    min_date = enriched["posted_date"].min().date()
    max_date = enriched["posted_date"].max().date()

    custom_range_value = sidebar.date_input(
        "Custom date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Select an explicit window when the presets aren't enough.",
    )
    custom_start: date
    custom_end: date
    if isinstance(custom_range_value, tuple):
        if len(custom_range_value) >= 2:
            first, second = custom_range_value[0], custom_range_value[1]
            custom_start = pd.to_datetime(first).date()
            custom_end = pd.to_datetime(second).date()
        elif len(custom_range_value) == 1:
            single = pd.to_datetime(custom_range_value[0]).date()
            custom_start = single
            custom_end = single
        else:
            custom_start = min_date
            custom_end = max_date
    else:
        single_date = pd.to_datetime(custom_range_value).date()
        custom_start = single_date
        custom_end = single_date

    quick_options = ["This month", "Last month", "Year to date", "All time", "Custom"]
    quick_choice = sidebar.radio(
        "Quick filters",
        quick_options,
        index=0,
        help="Apply preset date ranges before fine-tuning with the custom picker.",
    )

    def resolve_range(option: str) -> tuple[date, date]:
        current_period = pd.Timestamp(max_date).to_period("M")
        start = min_date
        end = max_date
        if option == "This month":
            start_ts = current_period.to_timestamp()
            end_ts = (current_period + 1).to_timestamp() - pd.Timedelta(days=1)
            start = start_ts.date()
            end = min(max_date, end_ts.date())
        elif option == "Last month":
            prev_period = current_period - 1
            start_ts = prev_period.to_timestamp()
            end_ts = current_period.to_timestamp() - pd.Timedelta(days=1)
            start = max(min_date, start_ts.date())
            end = min(max_date, end_ts.date())
        elif option == "Year to date":
            start = date(max_date.year, 1, 1)
            end = max_date
        elif option == "Custom":
            start = custom_start
            end = custom_end
        else:  # All time
            start = min_date
            end = max_date
        if start > end:
            start, end = end, start
        start = max(start, min_date)
        end = min(end, max_date)
        return start, end

    start_date, end_date = resolve_range(quick_choice)

    filtered = enriched.loc[
        (enriched["posted_date"].dt.date >= start_date)
        & (enriched["posted_date"].dt.date <= end_date)
    ].copy()

    currency_symbol_map = {"GBP": "£", "USD": "$", "EUR": "€"}
    currency_options = sorted(enriched["currency"].dropna().unique())
    currency_choice: str | None = None
    currency_symbol = "£"
    if len(currency_options) == 1:
        currency_symbol = currency_symbol_map.get(currency_options[0], currency_symbol)

    if len(currency_options) > 1:
        selection = sidebar.selectbox(
            "Currency",
            ["All currencies", *currency_options],
            help="Toggle to a single settlement currency when analysing multi-currency accounts.",
        )
        if selection != "All currencies":
            currency_choice = selection
            currency_symbol = currency_symbol_map.get(selection, selection)
        else:
            sidebar.caption("Aggregated totals shown in base currency (£).")
    elif currency_options:
        first_currency = currency_options[0]
        currency_choice = first_currency
        mapped_symbol = currency_symbol_map.get(first_currency)
        if mapped_symbol:
            currency_symbol = mapped_symbol
        sidebar.caption(f"Currency: {currency_choice}")

    if currency_choice:
        filtered = filtered.loc[filtered["currency"] == currency_choice].copy()

    insight_payload = insights.calculate_kpis(filtered)
    summary = summarize.summarize_spending(filtered)

    dataset_meta = f"{len(filtered):,} transactions | {start_date} → {end_date}"
    if currency_choice:
        dataset_meta += f" ({currency_choice})"
    st.caption(f"Filtered sample: {dataset_meta}")

    if filtered.empty:
        st.warning(
            "No transactions match the selected filters. Charts will show placeholders until data is available."
        )

    st.subheader("Key metrics")
    summary_metrics = insight_payload["summary"]
    summary_cards = [
        ("Total spend", utils.format_currency(summary_metrics["total_spend"], currency_symbol)),
        ("Total income", utils.format_currency(summary_metrics["total_income"], currency_symbol)),
        ("Net cashflow", utils.format_currency(summary_metrics["net_cashflow"], currency_symbol)),
        ("Avg. daily spend", utils.format_currency(summary_metrics["avg_daily_spend"], currency_symbol)),
        ("Recurring share", f"{summary_metrics['recurring_share_pct']:.1f}%"),
    ]
    metric_columns = st.columns(len(summary_cards))
    for column, (label, value) in zip(metric_columns, summary_cards, strict=False):
        column.metric(label=label, value=value)

    st.subheader("Net flow & projection")
    flow_col, projection_col = st.columns(2)
    flow_fig = viz.plot_monthly_net_flow(insight_payload["monthly_balance"])
    flow_fig.update_yaxes(tickprefix=currency_symbol)
    flow_col.plotly_chart(flow_fig, use_container_width=True)

    projection_fig = viz.plot_month_to_date_projection(insight_payload["month_to_date_projection"])
    projection_fig.update_yaxes(tickprefix=currency_symbol)
    projection_col.plotly_chart(projection_fig, use_container_width=True)

    st.subheader("Category spend & drill-down")
    donut_col, drill_col = st.columns([1, 1.2])
    category_entries = insight_payload["category_spend"]
    donut_fig = viz.plot_category_donut(category_entries)
    donut_fig.update_traces(hovertemplate=f"%{{label}}<br>{currency_symbol}%{{value:,.2f}}<extra></extra>")
    donut_col.plotly_chart(donut_fig, use_container_width=True)

    with drill_col:
        if category_entries:
            category_names = [entry["name"] for entry in category_entries]
            selected_category = st.selectbox(
                "Category",
                category_names,
                index=0,
                help="Focus the merchant drill-down on a single category.",
            )
            merchant_entries = insight_payload["category_merchants"].get(selected_category, [])
            merchant_fig = viz.plot_merchant_bar(merchant_entries, selected_category)
            merchant_fig.update_yaxes(tickprefix=currency_symbol)
            st.plotly_chart(merchant_fig, use_container_width=True)
        else:
            st.info("No merchant breakdown available for the selected filters.")

    with st.expander("Windowed category and merchant breakdowns"):
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
                    frame["amount"] = frame["amount"].apply(
                        lambda value: utils.format_currency(value, currency_symbol)
                    )
                    frame["share"] = frame["share"].apply(lambda value: f"{value * 100:.1f}%")
                    st.dataframe(frame, hide_index=True, use_container_width=True)

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
            value=utils.format_currency(amount, currency_symbol),
            delta=f"{(amount / total) * 100:.1f}% of spend",
        )

    essentials_cols = st.columns(2)
    for column, label, amount, total in (
        (essentials_cols[0], "Essentials", split["essentials"], essentials_total),
        (essentials_cols[1], "Discretionary", split["discretionary"], essentials_total),
    ):
        column.metric(
            label=label,
            value=utils.format_currency(amount, currency_symbol),
            delta=f"{(amount / total) * 100:.1f}% of spend",
        )

    st.subheader("Cash projection")
    projection = insight_payload["cash_projection"]
    projection_columns = st.columns(4)
    projection_columns[0].metric("As of", projection["as_of"])
    projection_columns[1].metric(
        "Projected month-end balance",
        utils.format_currency(projection["projected_end_balance"], currency_symbol),
    )
    projection_columns[2].metric(
        "Projected additional spend",
        utils.format_currency(projection["projected_burn"], currency_symbol),
    )
    projection_columns[3].metric(
        "Daily burn rate",
        utils.format_currency(projection["daily_burn_rate"], currency_symbol),
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
        display = recurring.copy()
        display["avg_amount"] = display["avg_amount"].apply(
            lambda value: utils.format_currency(value, currency_symbol)
        )
        display["last_amount"] = display["last_amount"].apply(
            lambda value: utils.format_currency(value, currency_symbol)
        )
        display["delta_amount"] = display["delta_amount"].apply(
            lambda value: utils.format_currency(value, currency_symbol)
        )

        def _format_delta(value: float) -> str:
            if value > 0:
                icon = "▲"
            elif value < 0:
                icon = "▼"
            else:
                icon = "●"
            return f"{icon} {value:+.1f}%"

        status_map = {
            "on_track": "✅ On track",
            "missed": "⚠️ Missed",
            "extra": "➕ Extra",
        }
        display["change"] = recurring["delta_percent"].apply(_format_delta)
        display["status_label"] = recurring["status"].map(status_map).fillna("–")

        display = display[
            ["merchant", "avg_amount", "last_amount", "delta_amount", "change", "status_label", "last_paid_date"]
        ]
        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "merchant": "Merchant",
                "avg_amount": st.column_config.TextColumn("Avg"),
                "last_amount": st.column_config.TextColumn("Last"),
                "delta_amount": st.column_config.TextColumn("Δ amount"),
                "change": st.column_config.TextColumn("Δ %"),
                "status_label": st.column_config.TextColumn("Status"),
                "last_paid_date": st.column_config.TextColumn("Last paid"),
            },
        )
    else:
        st.write("No recurring payments detected.")

    st.subheader("Outlier scatter")
    scatter_fig = viz.plot_outlier_scatter(insight_payload["outlier_points"])
    scatter_fig.update_yaxes(tickprefix=currency_symbol)
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.subheader("Potential anomalies")
    anomalies = pd.DataFrame(insight_payload["anomalies"])
    if not anomalies.empty:
        anomalies["amount"] = anomalies["amount"].apply(
            lambda value: utils.format_currency(value, currency_symbol)
        )
        anomalies["z_score"] = anomalies["z_score"].apply(lambda value: f"{value:.2f}")
        st.dataframe(anomalies, hide_index=True, use_container_width=True)
    else:
        st.write("No strong outliers found.")

    st.subheader("Historic spend trend")
    history_fig = viz.plot_spending_over_time(filtered)
    history_fig.update_yaxes(tickprefix=currency_symbol)
    st.plotly_chart(history_fig, use_container_width=True)

    st.subheader("Sample transactions")
    if filtered.empty:
        st.write("No transactions available for the current filters.")
    else:
        st.dataframe(filtered.head(200), use_container_width=True)

    sidebar.subheader("Exports")
    filtered_csv = filtered.to_csv(index=False)
    sidebar.download_button(
        "Download filtered CSV",
        data=filtered_csv,
        file_name="synthetic_transactions_filtered.csv",
        mime="text/csv",
        disabled=filtered.empty,
    )
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
