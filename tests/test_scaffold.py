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
    expected = {
        "month",
        "weekday",
        "is_recurring",
        "abs_amount",
        "monthly_net_amount",
        "category_month_share",
        "is_fixed_spend",
        "spend_rhythm",
        "spend_necessity",
    }
    assert expected.issubset(enriched.columns)
    assert enriched["abs_amount"].ge(0).all()
    assert set(enriched["is_credit"].unique()).issubset({True, False})
    assert set(enriched["spend_rhythm"].dropna().unique()).issubset({"Fixed", "Variable"})
    assert set(enriched["spend_necessity"].dropna().unique()).issubset({"Essentials", "Discretionary"})


def test_insights_calculate_kpis_returns_values() -> None:
    df = features.add_engineered_features(synth.generate_sample_transactions(rows=320, seed=3))
    payload = insights.calculate_kpis(df)

    summary = payload["summary"]
    assert summary["total_spend"] > 0
    assert summary["total_income"] > 0
    assert "monthly_balance" in payload and isinstance(payload["monthly_balance"], list)
    assert "top_categories" in payload and "current_month" in payload["top_categories"]
    assert "spend_split" in payload and set(payload["spend_split"].keys()) == {
        "fixed",
        "variable",
        "essentials",
        "discretionary",
    }
    assert "cash_projection" in payload and "projected_end_balance" in payload["cash_projection"]
    assert "payday" in payload and "next_payday" in payload["payday"]
    assert "category_spend" in payload
    assert "category_merchants" in payload and isinstance(payload["category_merchants"], dict)
    assert "month_to_date_projection" in payload and isinstance(payload["month_to_date_projection"], list)
    assert "outlier_points" in payload and isinstance(payload["outlier_points"], list)
    assert "recurring" in payload and isinstance(payload["recurring"], list)
    assert "anomalies" in payload and isinstance(payload["anomalies"], list)
    if payload["category_spend"]:
        sample_category = payload["category_spend"][0]["name"]
        assert sample_category in payload["category_merchants"]


def test_viz_plot_spending_over_time_returns_fig() -> None:
    df = features.add_engineered_features(synth.generate_sample_transactions(rows=500, seed=4))
    figure = viz.plot_spending_over_time(df)
    assert isinstance(figure, go.Figure)
    assert figure.data, "Chart should plot at least one trace"


def test_viz_stage_three_charts_return_figs() -> None:
    df = features.add_engineered_features(synth.generate_sample_transactions(rows=700, seed=6))
    payload = insights.calculate_kpis(df)

    assert isinstance(viz.plot_monthly_net_flow(payload["monthly_balance"]), go.Figure)
    assert isinstance(viz.plot_month_to_date_projection(payload["month_to_date_projection"]), go.Figure)
    assert isinstance(viz.plot_category_donut(payload["category_spend"]), go.Figure)
    assert isinstance(viz.plot_outlier_scatter(payload["outlier_points"]), go.Figure)

    if payload["category_spend"]:
        category_name = payload["category_spend"][0]["name"]
        merchants = payload["category_merchants"].get(category_name, [])
        assert isinstance(viz.plot_merchant_bar(merchants, category_name), go.Figure)


def test_summarize_spending_fallback() -> None:
    df = features.add_engineered_features(synth.generate_sample_transactions(rows=220, seed=5))
    summary = summarize.summarize_spending(df, model="gpt-4o-mini")
    assert "Highlights" in summary
