"""Regression tests for the Stage 1 synthetic dataset and pipeline."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from src import features, insights, summarize, synth, viz

EXPECTED_COLUMNS = {
    "txn_id",
    "posted_date",
    "value_date",
    "merchant_name",
    "mcc",
    "category",
    "subcategory",
    "amount",
    "currency",
    "balance_after",
    "channel",
    "location_city",
    "location_country",
    "card_last4",
    "user_id",
    "recurring_id",
    "notes",
}


def test_generate_sample_transactions_schema_and_determinism() -> None:
    first = synth.generate_sample_transactions(rows=250, seed=123)
    second = synth.generate_sample_transactions(rows=250, seed=123)

    pd.testing.assert_frame_equal(first, second)
    assert set(first.columns) == EXPECTED_COLUMNS
    assert (first["posted_date"] <= first["value_date"]).all()
    assert first["txn_id"].is_unique


def test_generate_transactions_balances_and_credit_debit_split() -> None:
    df = synth.generate_transactions(rows=1_000, seed=42)
    running = 2_500.0 + df["amount"].cumsum()
    pd.testing.assert_series_equal(df["balance_after"], running.round(2), check_names=False)
    spent = df.loc[df["amount"] < 0, "amount"].sum()
    received = df.loc[df["amount"] > 0, "amount"].sum()
    assert spent < 0
    assert received > 0


def test_write_synthetic_csvs(tmp_path) -> None:
    dataset_path, sample_path = synth.write_synthetic_csvs(
        output_dir=tmp_path,
        rows=600,
        sample_rows=120,
        seed=7,
    )

    dataset = pd.read_csv(dataset_path, parse_dates=["posted_date", "value_date"])
    sample = pd.read_csv(sample_path, parse_dates=["posted_date", "value_date"])

    assert len(dataset) == 600
    assert len(sample) == 120
    pd.testing.assert_frame_equal(sample, dataset.head(120))


def test_features_add_engineered_features_creates_columns() -> None:
    df = synth.generate_sample_transactions(rows=400, seed=2)
    enriched = features.add_engineered_features(df)
    expected = {"month", "weekday", "is_recurring", "abs_amount", "monthly_net_amount", "category_month_share"}
    assert expected.issubset(enriched.columns)
    assert enriched["abs_amount"].ge(0).all()
    assert set(enriched["is_credit"].unique()).issubset({True, False})


def test_insights_calculate_kpis_returns_values() -> None:
    df = features.add_engineered_features(synth.generate_sample_transactions(rows=320, seed=3))
    kpis = insights.calculate_kpis(df)
    assert {
        "Total spent",
        "Total received",
        "Avg. debit",
        "Recurring share",
        "Top spend category",
        "Net cashflow",
    }.issubset(kpis.keys())
    assert all(isinstance(val, str) and val for val in kpis.values())


def test_viz_plot_spending_over_time_returns_fig() -> None:
    df = features.add_engineered_features(synth.generate_sample_transactions(rows=500, seed=4))
    figure = viz.plot_spending_over_time(df)
    assert isinstance(figure, go.Figure)
    assert figure.data, "Chart should plot at least one trace"


def test_summarize_spending_fallback() -> None:
    df = features.add_engineered_features(synth.generate_sample_transactions(rows=220, seed=5))
    summary = summarize.summarize_spending(df, model="gpt-4o-mini")
    assert "Highlights" in summary
