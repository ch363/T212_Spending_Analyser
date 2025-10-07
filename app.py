"""Streamlit entry point for the Spending Analyser app."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from src import features, insights, summarize, synth, viz


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
    st.caption("Stage 1: deterministic synthetic ledger with 10k baseline rows")

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
    kpis = insights.calculate_kpis(enriched)
    summary = summarize.summarize_spending(enriched)

    dataset_meta = (
        f"{len(enriched):,} transactions | {enriched['posted_date'].min().date()}"
        f" → {enriched['posted_date'].max().date()}"
    )
    st.caption(f"Sample preview: {dataset_meta}")

    st.subheader("Key metrics")
    if kpis:
        metric_columns = st.columns(len(kpis))
        for column, (label, value) in zip(metric_columns, kpis.items(), strict=False):
            column.metric(label=label, value=value)
    else:
        st.write("No metrics available yet.")

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
