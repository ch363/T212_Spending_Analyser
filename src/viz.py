"""Visualization utilities for Spending Analyser."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from . import utils


def plot_spending_over_time(transactions: pd.DataFrame):
    """Return a Plotly figure visualising spending over time."""

    df = utils.ensure_dataframe(transactions)

    if "month" not in df:
        df = df.assign(
            month=pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
        )

    grouped = (
        df.groupby("month", as_index=False)["amount"].sum().assign(amount=lambda d: d["amount"].abs())
    )

    fig = px.bar(
        grouped,
        x="month",
        y="amount",
        title="Monthly spend",
        labels={"month": "Month", "amount": "Amount"},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    return fig
