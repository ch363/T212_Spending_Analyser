"""Streamlit entry point for the Spending Analyser app."""

from __future__ import annotations

import streamlit as st

from src import features, insights, summarize, synth, viz


def main() -> None:
    """Render the Spending Analyser Streamlit application."""

    st.set_page_config(
        page_title="Spending Analyser",
        page_icon="💸",
        layout="wide",
    )

    st.title("Spending Analyser")
    st.caption("Stage 0 scaffold – synthetic data and placeholder insights")

    transactions = synth.generate_sample_transactions()
    enriched = features.add_engineered_features(transactions)
    kpis = insights.calculate_kpis(enriched)
    summary = summarize.summarize_spending(enriched)

    st.subheader("Key metrics")
    if kpis:
        metric_columns = st.columns(len(kpis))
        for column, (label, value) in zip(metric_columns, kpis.items(), strict=False):
            column.metric(label=label, value=value)
    else:
        st.write("No metrics available yet.")

    st.subheader("Spending over time")
    chart = viz.plot_spending_over_time(enriched)
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Sample transactions")
    st.dataframe(enriched, use_container_width=True)

    st.subheader("AI summary")
    st.info(summary)


if __name__ == "__main__":
    main()
