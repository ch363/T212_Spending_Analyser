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

    st.markdown(
        """
        <style>
        :root {
            --primary-500: #2563eb;
            --primary-600: #1d4ed8;
            --primary-50: #eff6ff;
            --slate-900: #0f172a;
            --slate-700: #334155;
            --slate-500: #64748b;
            --slate-400: #94a3b8;
            --success-500: #15803d;
            --danger-500: #dc2626;
        }

        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 15% 20%, rgba(37, 99, 235, 0.06), transparent 32%),
                        radial-gradient(circle at 85% 15%, rgba(99, 102, 241, 0.05), transparent 35%),
                        #f5f7fb;
            color: var(--slate-900);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.92) !important;
            backdrop-filter: blur(16px);
            border-right: 1px solid rgba(226, 232, 240, 0.8);
        }

        [data-testid="stSidebar"] * label {
            font-weight: 600;
        }

        section[data-testid="stHorizontalBlock"] {
            gap: 1.75rem;
        }

        section[data-testid="stVerticalBlock"] > div {
            margin-bottom: 1.75rem;
        }

        h1 {
            font-size: 2.65rem;
            font-weight: 700;
            line-height: 1.08;
            color: var(--slate-900);
        }

        h2 {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.15;
            color: var(--slate-900);
            margin-bottom: 0.35rem;
        }

        h3 {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--slate-900);
            margin-bottom: 0.4rem;
        }

        .hero-card {
            background: #ffffff;
            border-radius: 24px;
            border: 1px solid rgba(226, 232, 240, 0.9);
            padding: 1.6rem 1.75rem;
            box-shadow: 0 36px 72px -50px rgba(15, 23, 42, 0.55);
        }

        .hero-card__label {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--slate-500);
            margin-bottom: 0.55rem;
        }

        .hero-card__value {
            font-size: 2.45rem;
            font-weight: 700;
            color: var(--slate-900);
            line-height: 1.1;
        }

        .hero-card__delta {
            margin-top: 0.75rem;
            font-weight: 600;
            font-size: 1.02rem;
        }

        .hero-card__delta.hero-card__delta--positive {
            color: var(--success-500);
        }

        .hero-card__delta.hero-card__delta--negative {
            color: var(--danger-500);
        }

        .hero-card__delta.hero-card__delta--neutral {
            color: var(--slate-500);
        }

        .hero-card__meta {
            margin-top: 1.35rem;
            padding-top: 1.05rem;
            border-top: 1px solid rgba(226, 232, 240, 0.9);
            display: grid;
            gap: 0.85rem;
        }

        .hero-card__meta-block {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
        }

        .hero-card__meta-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--slate-500);
            font-weight: 600;
        }

        .hero-card__meta-value {
            font-size: 1rem;
            font-weight: 600;
            color: var(--slate-700);
        }

        .hero-card__divider {
            height: 1px;
            background: rgba(226, 232, 240, 0.9);
        }

        .hero-card--chart {
            padding: 1.1rem 1.2rem 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
        }

        .hero-card--chart .hero-card__label {
            margin-bottom: 0;
        }

        .stMarkdown p {
            color: var(--slate-600);
        }

        .stCaption, .stMarkdown p.caption-text {
            font-size: 0.9rem;
            color: var(--slate-500);
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border-radius: 18px;
            border: 1px solid rgba(226, 232, 240, 0.9);
            padding: 1.15rem 1.25rem;
            box-shadow: 0 30px 60px -46px rgba(15, 23, 42, 0.6);
        }

        div[data-testid="stMetric"] label {
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-size: 0.78rem;
            color: var(--slate-500);
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--slate-900);
        }

        div[data-testid="stMetricDelta"] {
            font-size: 0.95rem;
            font-weight: 600;
        }

        div[data-testid="stMetric"] + p {
            margin-top: 0.35rem;
        }

        div[data-testid="column"] {
            gap: 1.05rem;
        }

        .stTabs [role="tab"] {
            background: rgba(148, 163, 184, 0.14);
            border-radius: 999px;
            padding: 0.45rem 1.1rem;
            border: none;
            font-weight: 600;
            color: var(--slate-500);
        }

        .stTabs [role="tab"][aria-selected="true"] {
            background: var(--primary-500);
            color: #ffffff;
            box-shadow: 0 8px 18px -12px rgba(37, 99, 235, 0.9);
        }

        .stDataFrame {
            border-radius: 18px;
            border: 1px solid rgba(226, 232, 240, 0.9);
            overflow: hidden;
        }

        .stPlotlyChart {
            margin-bottom: 0 !important;
        }

        /* Treat the container that includes the marker as a card */
        div[data-testid="stVerticalBlock"]:has(#hero-plot-card-marker) {
            background: #ffffff;
            border-radius: 24px;
            border: 1px solid rgba(226, 232, 240, 0.9);
            padding: 1.2rem 1.25rem 1.35rem 1.25rem;
            box-shadow: 0 36px 72px -50px rgba(15, 23, 42, 0.55);
        }

        div[data-testid="stVerticalBlock"]:has(#hero-plot-card-marker) .stPlotlyChart {
            margin: 0 !important;
        }

        /* Inside the hero plot card, keep the Plotly container visually flat */
        div[data-testid="stVerticalBlock"]:has(#hero-plot-card-marker) div[data-testid="stPlotlyChart"] {
            background: transparent !important;
            border-radius: 0 !important;
            border: none !important;
            box-shadow: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

    monthly_balance_df = pd.DataFrame(insight_payload["monthly_balance"])
    if not monthly_balance_df.empty:
        monthly_balance_df["month"] = pd.to_datetime(monthly_balance_df["month"])
        monthly_balance_df = monthly_balance_df.sort_values("month")

    dataset_meta = f"{len(filtered):,} transactions · {start_date} → {end_date}"
    if currency_choice:
        dataset_meta += f" · {currency_choice}"

    summary_metrics = insight_payload["summary"]
    total_spend = summary_metrics["total_spend"]
    total_income = summary_metrics["total_income"]

    spend_delta_text: str | None = None
    net_delta_text: str | None = None

    if len(monthly_balance_df) >= 2:
        current_spend = float(monthly_balance_df["spend"].iloc[-1])
        previous_spend = float(monthly_balance_df["spend"].iloc[-2])
        delta_amount = previous_spend - current_spend
        spend_delta_text = (
            f"{'+' if delta_amount >= 0 else ''}{utils.format_currency(delta_amount, currency_symbol)} vs last month"
        )

        current_net = float(monthly_balance_df["net"].iloc[-1])
        previous_net = float(monthly_balance_df["net"].iloc[-2])
        net_change = current_net - previous_net
        net_delta_text = (
            f"{'+' if net_change >= 0 else ''}{utils.format_currency(net_change, currency_symbol)} vs last month"
        )

    top_merchant_entries = insight_payload["top_merchants"].get("current_month", [])
    top_merchant_name = top_merchant_entries[0]["name"] if top_merchant_entries else "—"
    top_merchant_amount = (
        utils.format_currency(top_merchant_entries[0]["amount"], currency_symbol)
        if top_merchant_entries
        else "—"
    )
    top_merchant_share = (
        f"{top_merchant_entries[0]['share'] * 100:.1f}% of spend"
        if top_merchant_entries
        else "No merchants in range"
    )

    recurring_count = len(insight_payload["recurring"])
    recurring_share = summary_metrics["recurring_share_pct"]

    current_period = pd.Timestamp(end_date).to_period("M")
    current_month_label = current_period.to_timestamp().strftime("%B %Y")

    if filtered.empty:
        current_month_df = filtered.copy()
    else:
        current_month_df = filtered.loc[filtered["posted_date"].dt.to_period("M") == current_period].copy()

    current_month_spend = float(
        -current_month_df.loc[current_month_df["amount"] < 0, "amount"].sum()
    ) if not current_month_df.empty else 0.0

    previous_month_spend = 0.0
    previous_period = current_period - 1
    previous_month_df = (
        filtered.loc[filtered["posted_date"].dt.to_period("M") == previous_period]
        if not filtered.empty
        else filtered
    )
    if not previous_month_df.empty:
        previous_month_spend = float(-previous_month_df.loc[previous_month_df["amount"] < 0, "amount"].sum())

    category_totals_df = pd.DataFrame(columns=["category", "value"])
    monthly_category_entries: list[insights.BreakdownEntry] = []
    top_category_name = "—"
    top_category_amount = 0.0
    top_category_share = 0.0
    if not current_month_df.empty:
        month_spend = current_month_df.loc[current_month_df["amount"] < 0].copy()
        if not month_spend.empty:
            month_spend["value"] = -month_spend["amount"]
            category_totals_df = (
                month_spend.groupby("category", as_index=False)
                .agg({"value": "sum"})
                .sort_values("value", ascending=False)
            )
            total_category_value = float(category_totals_df["value"].sum())
            monthly_category_entries = [
                {
                    "name": str(row["category"]),
                    "amount": float(row["value"]),
                    "share": float(row["value"] / total_category_value) if total_category_value else 0.0,
                }
                for row in category_totals_df.to_dict("records")
            ]
            if not category_totals_df.empty:
                top_row = category_totals_df.iloc[0]
                top_category_name = str(top_row["category"])
                top_category_amount = float(top_row["value"])
                top_category_share = float(top_row["value"] / total_category_value) if total_category_value else 0.0

    if top_category_name == "—" and insight_payload["category_spend"]:
        fallback_category = insight_payload["category_spend"][0]
        top_category_name = fallback_category["name"]
        top_category_amount = float(fallback_category["amount"])
        top_category_share = float(fallback_category["share"])

    category_entries_all = insight_payload["category_spend"]
    category_entries_display = monthly_category_entries or category_entries_all

    focus_name = top_category_name if top_category_name != "—" else None

    st.title("Understand your spending in minutes.")
    st.caption(dataset_meta)

    # Wider right card, slightly narrower left summary card
    hero_left, hero_right = st.columns([0.9, 1.1], gap="large")
    with hero_left:
        if focus_name:
            share_text = f" · {top_category_share * 100:.1f}%" if top_category_share else ""
            focus_meta_value = (
                f"{focus_name} · {utils.format_currency(top_category_amount, currency_symbol)}{share_text}"
            )
        else:
            focus_meta_value = "No category data available."

        delta_percent_text = "No spend recorded this month" if current_month_spend <= 0 else "No prior month available"
        change_amount_text = "—"
        delta_class = "neutral"

        if previous_month_spend > 0:
            delta_amount = current_month_spend - previous_month_spend
            pct_change_value = (delta_amount / previous_month_spend) * 100
            if delta_amount > 0:
                delta_percent_text = f"Up {pct_change_value:.1f}% vs last month"
                delta_class = "negative"
                change_amount_prefix = "+"
            elif delta_amount < 0:
                delta_percent_text = f"Down {abs(pct_change_value):.1f}% vs last month"
                delta_class = "positive"
                change_amount_prefix = "−"
            else:
                delta_percent_text = "Flat vs last month"
                delta_class = "neutral"
                change_amount_prefix = ""
            change_amount_text = (
                f"{change_amount_prefix}{utils.format_currency(abs(delta_amount), currency_symbol)}"
                f" ({pct_change_value:+.1f}%)"
                f" vs {utils.format_currency(previous_month_spend, currency_symbol)} last month"
            )
        elif current_month_spend > 0:
            delta_percent_text = "New activity this month"
            change_amount_text = "No prior month to compare."
        else:
            change_amount_text = "No spend recorded."

        card_html = f"""
        <div class="hero-card">
            <div class="hero-card__label">Total this month</div>
            <div class="hero-card__value">{utils.format_currency(current_month_spend, currency_symbol)}</div>
            <div class="hero-card__delta hero-card__delta--{delta_class}">{delta_percent_text}</div>
            <div class="hero-card__meta">
                <div class="hero-card__meta-block">
                    <div class="hero-card__meta-label">Change vs last month</div>
                    <div class="hero-card__meta-value">{change_amount_text}</div>
                </div>
                <div class="hero-card__divider"></div>
                <div class="hero-card__meta-block">
                    <div class="hero-card__meta-label">Focus category</div>
                    <div class="hero-card__meta-value">{focus_meta_value}</div>
                </div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        comparison_caption: str | None
        if not previous_month_df.empty:
            previous_month_label = previous_period.to_timestamp().strftime("%B %Y")
            comparison_caption = f"{current_month_label} vs {previous_month_label}"
        elif previous_month_spend == 0 and current_month_spend > 0:
            comparison_caption = "First month of activity in the selected window."
        elif current_month_spend <= 0:
            comparison_caption = "No spend recorded in the selected window."
        else:
            comparison_caption = None

        if comparison_caption:
            st.caption(comparison_caption)

    with hero_right:
        # Wrap in a container so the marker is a descendant; CSS will pick it up
        # Single, clean card: marker + label + chart
        st.markdown('<div id="hero-plot-card-marker" class="hero-card hero-card--chart">', unsafe_allow_html=True)
        st.markdown('<div class="hero-card__label">Daily spend trend</div>', unsafe_allow_html=True)
        chart_df = current_month_df.copy()
        daily_fig = viz.plot_daily_spend(chart_df, currency_symbol=currency_symbol)
        st.plotly_chart(daily_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    category_row_left, category_row_right = st.columns([1, 1], gap="large")

    with category_row_left:
        category_row_left.markdown("### Category spend split")
        donut_fig = viz.plot_category_donut(category_entries_display)
        donut_fig.update_traces(hovertemplate=f"%{{label}}<br>{currency_symbol}%{{value:,.2f}}<extra></extra>")
        category_row_left.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})

    with category_row_right:
        category_row_right.markdown("### Category values")
        if category_entries_display:
            category_table = pd.DataFrame(category_entries_display)
            category_table = category_table.rename(columns={"name": "Category", "amount": "Amount", "share": "Share"})
            category_table["Amount"] = category_table["Amount"].apply(
                lambda value: utils.format_currency(float(value), currency_symbol)
            )
            category_table["Share"] = category_table["Share"].apply(lambda value: f"{value * 100:.1f}%")
            category_row_right.table(category_table.set_index("Category"))
        else:
            category_row_right.caption("No category spend available for the selected period.")

    with st.container():
        # Auto-generate on page load/refresh
        if filtered.empty:
            st.markdown("### Narrative summary")
            st.caption("LLM-ready copy spanning the full-width canvas")
            st.caption("No data in the current filter to summarise.")
        else:
            with st.spinner("Generating AI summary…"):
                ai_summary = summarize.summarize_spending(filtered)

            # Determine source and render a small badge next to the title
            is_fallback = isinstance(ai_summary, str) and ai_summary.startswith("Highlights —")
            badge_label = "[Fallback]" if is_fallback else "[AI]"
            badge_class = "fallback-badge" if is_fallback else "ai-badge"
            # Pull optional diagnostics/meta written by summarize.summarize_spending
            meta = st.session_state.get("ai_summary_meta", {})
            fallback_reason = meta.get("reason") if isinstance(meta, dict) else None
            tooltip = (
                f"Fallback: {fallback_reason}" if (is_fallback and isinstance(fallback_reason, str) and fallback_reason)
                else ("Generated by OpenAI" if not is_fallback else "Fallback summary (no LLM)")
            )

            st.markdown(
                """
                <style>
                .title-row { display: flex; align-items: center; gap: 8px; margin-bottom: 0.1rem; }
                .ai-badge { font-size: 0.75rem; font-weight: 700; color: #155e75; background: #cffafe; border: 1px solid #06b6d4; border-radius: 999px; padding: 2px 8px; }
                .fallback-badge { font-size: 0.75rem; font-weight: 700; color: #6b7280; background: #f3f4f6; border: 1px solid #d1d5db; border-radius: 999px; padding: 2px 8px; }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="title-row"><h3>Narrative summary</h3>'
                f'<span class="{badge_class}" title="{tooltip}">{badge_label}</span></div>',
                unsafe_allow_html=True,
            )
            st.caption("LLM-ready copy spanning the full-width canvas")
            if is_fallback and isinstance(fallback_reason, str) and fallback_reason:
                st.caption(f"Fallback reason: {fallback_reason}")
            st.markdown(ai_summary)

    projection = insight_payload["cash_projection"]
    payday = insight_payload["payday"]
    days_to_payday = payday.get("days_to_next_payday")
    payday_subtitle = (
        f"{int(days_to_payday)} days" if isinstance(days_to_payday, (int, float)) else "Date pending"
    )

    projection_cols = st.columns(4)
    projection_cols[0].metric(
        "Projected month end",
        utils.format_currency(projection["projected_end_balance"], currency_symbol),
    )
    projection_cols[0].caption("Expected closing balance")
    projection_cols[1].metric(
        "Projected burn",
        utils.format_currency(projection["projected_burn"], currency_symbol),
    )
    projection_cols[1].caption("Spend remaining in period")
    projection_cols[2].metric(
        "Daily burn rate",
        utils.format_currency(projection["daily_burn_rate"], currency_symbol),
    )
    projection_cols[2].caption(f"{projection['days_remaining']} days left")
    projection_cols[3].metric("Next payday", payday.get("next_payday", "—") or "—")
    projection_cols[3].caption(payday_subtitle)

    split = insight_payload["spend_split"]
    spend_total = abs(total_spend)

    split_cols = st.columns(4)
    split_cols[0].metric("Fixed spend", utils.format_currency(split["fixed"], currency_symbol))
    split_cols[0].caption(
        f"{split['fixed'] / spend_total * 100:.1f}% of spend" if spend_total else "0.0% of spend"
    )
    split_cols[1].metric("Variable spend", utils.format_currency(split["variable"], currency_symbol))
    split_cols[1].caption(
        f"{split['variable'] / spend_total * 100:.1f}% of spend" if spend_total else "0.0% of spend"
    )
    split_cols[2].metric("Essentials", utils.format_currency(split["essentials"], currency_symbol))
    split_cols[2].caption(
        f"{split['essentials'] / spend_total * 100:.1f}% essentials" if spend_total else "0.0% essentials"
    )
    split_cols[3].metric("Discretionary", utils.format_currency(split["discretionary"], currency_symbol))
    split_cols[3].caption(
        f"{split['discretionary'] / spend_total * 100:.1f}% discretionary" if spend_total else "0.0% discretionary"
    )

    with st.container():
        chart_col_left, chart_col_right = st.columns([1.35, 1])

        with chart_col_left:
            chart_col_left.markdown("### Spending over time")
            chart_col_left.caption("Daily ledger totals")
            history_fig = viz.plot_spending_over_time(filtered)
            history_fig.update_yaxes(tickprefix=currency_symbol)
            st.plotly_chart(history_fig, use_container_width=True, config={"displayModeBar": False})

        with chart_col_right:
            chart_col_right.markdown("### Month-to-date projection")
            chart_col_right.caption("Projected burn based on current trend")
            projection_fig = viz.plot_month_to_date_projection(insight_payload["month_to_date_projection"])
            projection_fig.update_yaxes(tickprefix=currency_symbol)
            st.plotly_chart(projection_fig, use_container_width=True, config={"displayModeBar": False})

    with st.container():
        flow_col, merchant_col = st.columns([1.05, 1])

        with flow_col:
            flow_col.markdown("### Monthly net flow")
            flow_col.caption("Income vs spend with net overlay")
            flow_fig = viz.plot_monthly_net_flow(insight_payload["monthly_balance"])
            flow_fig.update_yaxes(tickprefix=currency_symbol)
            st.plotly_chart(flow_fig, use_container_width=True, config={"displayModeBar": False})

        with merchant_col:
            merchant_col.markdown("### Merchant drill-down")
            merchant_col.caption("Focus on the merchants driving category spend")
            if category_entries_display:
                category_names = [entry["name"] for entry in category_entries_display]
                selected_category = st.selectbox(
                    "Category",
                    category_names,
                    index=0,
                    key="category_drill_select",
                )
            else:
                selected_category = None

            if selected_category:
                merchant_entries = insight_payload["category_merchants"].get(selected_category, [])
                if merchant_entries:
                    merchant_fig = viz.plot_merchant_bar(merchant_entries, selected_category)
                    merchant_fig.update_yaxes(tickprefix=currency_symbol)
                    merchant_col.plotly_chart(merchant_fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    merchant_col.caption("No merchant data for the selected category.")
            else:
                merchant_col.caption("Add transactions to unlock merchant drill-down.")

            with merchant_col.expander("Windowed breakdowns"):
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

    with st.container():
        scatter_col, recurring_col = st.columns([1, 1])

        with scatter_col:
            scatter_col.markdown("### Outlier scatter")
            scatter_col.caption("Flagged via merchant-level z-scores")
            scatter_fig = viz.plot_outlier_scatter(insight_payload["outlier_points"])
            scatter_fig.update_yaxes(tickprefix=currency_symbol)
            st.plotly_chart(scatter_fig, use_container_width=True, config={"displayModeBar": False})

        with recurring_col:
            recurring_col.markdown("### Recurring payments")
            recurring_col.caption("Change badges surface variance")
            recurring_df = pd.DataFrame(insight_payload["recurring"])
            if not recurring_df.empty:
                display = recurring_df.copy()
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
                display["change"] = recurring_df["delta_percent"].apply(_format_delta)
                display["status_label"] = recurring_df["status"].map(status_map).fillna("–")

                display = display[
                    [
                        "merchant",
                        "avg_amount",
                        "last_amount",
                        "delta_amount",
                        "change",
                        "status_label",
                        "last_paid_date",
                    ]
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
                st.caption("No recurring payments detected for the selected filters.")

    with st.container():
        st.markdown("### Potential anomalies")
        st.caption("Automated merchant-level outliers")
        anomalies_df = pd.DataFrame(insight_payload["anomalies"])
        if not anomalies_df.empty:
            anomalies_df["amount"] = anomalies_df["amount"].apply(
                lambda value: utils.format_currency(value, currency_symbol)
            )
            anomalies_df["z_score"] = anomalies_df["z_score"].apply(
                lambda value: f"{value:.2f}"
            )
            st.dataframe(anomalies_df, hide_index=True, use_container_width=True)
        else:
            st.caption("No strong anomalies detected in this window.")

    with st.container():
        st.markdown("### Sample transactions")
        st.caption(f"Showing up to 200 rows ({len(filtered):,} total)")
        if filtered.empty:
            st.caption("No transactions available for the current filters.")
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


if __name__ == "__main__":
    main()
