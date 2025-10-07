"""Regression tests for the Stage 0 scaffold."""

import plotly.graph_objects as go

from src import features, insights, summarize, synth, viz


def test_generate_sample_transactions_shape() -> None:
    df = synth.generate_sample_transactions(rows=10, seed=123)
    assert not df.empty
    assert {"transaction_id", "date", "merchant", "category", "amount"}.issubset(df.columns)
    assert (df["amount"] < 0).all()


def test_features_add_engineered_features_creates_columns() -> None:
    df = synth.generate_sample_transactions(rows=12, seed=2)
    enriched = features.add_engineered_features(df)
    assert {"month", "weekday", "is_recurring", "abs_amount"}.issubset(enriched.columns)
    assert enriched["abs_amount"].ge(0).all()


def test_insights_calculate_kpis_returns_values() -> None:
    df = features.add_engineered_features(synth.generate_sample_transactions(rows=15, seed=3))
    kpis = insights.calculate_kpis(df)
    assert "Total spent" in kpis
    assert all(isinstance(val, str) and val for val in kpis.values())


def test_viz_plot_spending_over_time_returns_fig() -> None:
    df = features.add_engineered_features(synth.generate_sample_transactions(rows=20, seed=4))
    figure = viz.plot_spending_over_time(df)
    assert isinstance(figure, go.Figure)
    assert figure.data, "Chart should plot at least one trace"


def test_summarize_spending_fallback() -> None:
    df = features.add_engineered_features(synth.generate_sample_transactions(rows=8, seed=5))
    summary = summarize.summarize_spending(df, model="gpt-4o-mini")
    assert "Highlights" in summary
