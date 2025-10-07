"""Insights and aggregation helpers for Spending Analyser."""

from __future__ import annotations

import pandas as pd

from . import utils


def calculate_kpis(transactions: pd.DataFrame) -> dict[str, str]:
    """Compute core KPIs for the Spending Analyser dashboard."""

    df = utils.ensure_dataframe(transactions).copy()

    if "is_recurring" not in df and "recurring_id" in df:
        df["is_recurring"] = df["recurring_id"].notna()

    debits = df.loc[df["amount"] < 0, "amount"]
    credits = df.loc[df["amount"] > 0, "amount"]

    total_debits = -debits.sum()
    total_credits = credits.sum()
    avg_debit = -debits.mean() if not debits.empty else 0.0
    recurring_share = (df.loc[df["amount"] < 0, "is_recurring"].mean() * 100.0) if "is_recurring" in df else 0.0

    by_category = (
        df.loc[df["amount"] < 0]
        .groupby("category")["amount"]
        .sum()
        .sort_values()
    )
    if not by_category.empty:
        top_category = by_category.index[0]
        top_spend = -by_category.iloc[0]
    else:
        top_category, top_spend = "N/A", 0.0

    net_cashflow = total_credits - total_debits

    return {
        "Total spent": utils.format_currency(total_debits),
        "Total received": utils.format_currency(total_credits),
        "Avg. debit": utils.format_currency(avg_debit),
        "Recurring share": f"{recurring_share:,.1f}%",
        "Top spend category": f"{top_category} ({utils.format_currency(top_spend)})",
        "Net cashflow": utils.format_currency(net_cashflow),
    }
