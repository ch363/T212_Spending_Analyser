"""Visualization utilities for Spending Analyser."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import utils


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        font=dict(size=14, color="#6c757d"),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=20))
    return fig


def plot_spending_over_time(transactions: pd.DataFrame) -> go.Figure:
    """Return a Plotly figure visualising historic spending totals."""

    df = utils.ensure_dataframe(transactions).copy()

    if df.empty:
        return _empty_figure("No transactions available.")

    if "month" not in df:
        df["month"] = pd.to_datetime(df["posted_date"]).dt.to_period("M").dt.to_timestamp()

    debit_totals = (
        df.loc[df["amount"] < 0]
        .groupby("month", as_index=False)["amount"]
        .sum()
    )
    if debit_totals.empty:
        return _empty_figure("No debit transactions in range.")

    debit_totals["amount"] = debit_totals["amount"].abs()

    fig = px.bar(
        debit_totals,
        x="month",
        y="amount",
        title="Monthly spend",
        labels={"month": "Month", "amount": "Amount"},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    return fig


def plot_monthly_net_flow(monthly_balance: Iterable[Mapping[str, object]]) -> go.Figure:
    data = list(monthly_balance)
    if not data:
        return _empty_figure("No monthly balance data available.")

    df = pd.DataFrame(data)
    df["month"] = pd.to_datetime(df["month"])

    fig = go.Figure()
    fig.add_bar(
        name="Income",
        x=df["month"],
        y=df["income"],
        marker_color="#2a9d8f",
    )
    fig.add_bar(
        name="Spend",
        x=df["month"],
        y=-df["spend"],
        marker_color="#e76f51",
    )
    fig.add_trace(
        go.Scatter(
            name="Net",
            x=df["month"],
            y=df["net"],
            mode="lines+markers",
            line=dict(color="#264653", width=2),
        )
    )
    fig.update_layout(
        barmode="relative",
        title="Monthly net flow",
        yaxis_title="Amount",
        xaxis_title="Month",
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def plot_category_donut(breakdown: Iterable[Mapping[str, object]]) -> go.Figure:
    data = list(breakdown)
    if not data:
        return _empty_figure("No category spend to display.")

    df = pd.DataFrame(data)
    fig = px.pie(
        df,
        names="name",
        values="amount",
        hole=0.55,
        title="Category spend split",
    )
    fig.update_traces(textinfo="label+percent", pull=[0.03] * len(df))
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


def plot_merchant_bar(entries: Iterable[Mapping[str, object]], category: str) -> go.Figure:
    data = list(entries)
    if not data:
        return _empty_figure("Select a category to view merchants.")

    df = pd.DataFrame(data)
    fig = px.bar(
        df,
        x="name",
        y="amount",
        labels={"name": "Merchant", "amount": "Spend"},
        title=f"Top merchants in {category}",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=80))
    fig.update_xaxes(tickangle=-30)
    return fig


def plot_month_to_date_projection(points: Iterable[Mapping[str, object]]) -> go.Figure:
    data = list(points)
    if not data:
        return _empty_figure("No month-to-date information available.")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()

    actual = df.dropna(subset=["actual"])
    if not actual.empty:
        fig.add_trace(
            go.Scatter(
                name="Actual",
                x=actual["date"],
                y=actual["actual"],
                mode="lines+markers",
                line=dict(color="#1d3557", width=2),
            )
        )

    fig.add_trace(
        go.Scatter(
            name="Projected",
            x=df["date"],
            y=df["projected"],
            mode="lines",
            line=dict(color="#457b9d", dash="dash", width=2),
        )
    )

    fig.update_layout(
        title="Month-to-date spend projection",
        xaxis_title="Date",
        yaxis_title="Cumulative spend",
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


def plot_outlier_scatter(points: Iterable[Mapping[str, object]]) -> go.Figure:
    data = list(points)
    if not data:
        return _empty_figure("No spend transactions to chart.")

    df = pd.DataFrame(data)
    df["category"] = df["category"].fillna("Unknown")
    df["merchant"] = df["merchant"].fillna("Unknown")
    df["flag"] = np.where(df["is_flagged"], "Flagged", "Baseline")

    fig = px.scatter(
        df,
        x="merchant",
        y="amount",
        color="flag",
        hover_data={
            "merchant": True,
            "category": True,
            "amount": ":.2f",
            "z_score": ":.2f",
            "posted_date": True,
        },
        title="Merchant outlier scatter",
    )

    fig.update_traces(marker=dict(size=np.where(df["is_flagged"], 14, 8), opacity=0.8))
    fig.update_layout(
        xaxis_title="Merchant",
        yaxis_title="Transaction amount",
        margin=dict(l=0, r=0, t=45, b=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showticklabels=False)

    return fig
