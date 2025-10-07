"""Insights and aggregation helpers for Spending Analyser."""

from __future__ import annotations

import pandas as pd

from . import utils


def calculate_kpis(transactions: pd.DataFrame) -> dict[str, str]:
    """Compute a handful of simple KPIs from the transactions dataframe."""

    df = utils.ensure_dataframe(transactions)

    total_spent = df["amount"].sum()
    average_txn = df["amount"].mean()
    recurring_share = df["is_recurring"].mean() * 100 if "is_recurring" in df else 0.0

    by_category = df.groupby("category")["amount"].sum().sort_values()
    top_category, top_spend = (by_category.index[-1], by_category.iloc[-1]) if not by_category.empty else ("N/A", 0.0)

    return {
        "Total spent": utils.format_currency(abs(total_spent)),
        "Avg. transaction": utils.format_currency(abs(average_txn)),
        "Recurring share": f"{recurring_share:,.1f}%",
        "Top category": f"{top_category} ({utils.format_currency(abs(top_spend))})",
    }
