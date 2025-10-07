"""LLM-powered summarization utilities for Spending Analyser."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from . import insights, utils

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


def _build_prompt(df: pd.DataFrame) -> str:
    monthly = df.groupby("month")["amount"].sum().abs().sort_index().tail(3)
    top_merchants = (
        df.groupby("merchant_name")["abs_amount"].sum().sort_values(ascending=False).head(3)
        if "merchant_name" in df
        else df.groupby("merchant")["abs_amount"].sum().sort_values(ascending=False).head(3)
    )
    lines = ["Provide a concise summary of personal spending trends."]
    lines.append("Recent monthly totals:")
    for month, value in monthly.items():
        lines.append(f"- {month.strftime('%b %Y')}: {utils.format_currency(value)}")

    lines.append("Top merchants by spend:")
    for merchant, value in top_merchants.items():
        lines.append(f"- {merchant}: {utils.format_currency(value)}")

    return "\n".join(lines)


def _fallback_summary(df: pd.DataFrame) -> str:
    """Return a compact human-friendly summary without calling an LLM.

    We intentionally avoid dumping raw dicts; instead we surface a few
    high-signal metrics that mirror the UI.
    """

    payload = insights.calculate_kpis(df)
    s = payload["summary"]

    # Month-over-month change using the last two rows of monthly_balance
    mb = pd.DataFrame(payload.get("monthly_balance", []))
    mom_text = ""
    if not mb.empty:
        mb["month"] = pd.to_datetime(mb["month"])
        mb = mb.sort_values("month")
        if len(mb) >= 2:
            current = float(mb.iloc[-1]["spend"]) or 0.0
            previous = float(mb.iloc[-2]["spend"]) or 0.0
            if previous > 0:
                delta = current - previous
                pct = (delta / previous) * 100
                direction = "up" if delta > 0 else ("down" if delta < 0 else "flat")
                mom_text = f"Spend is {direction} {abs(pct):.1f}% vs last month."
            else:
                mom_text = "First month of activity in the selected window."

    # Top category and merchant (current month where possible)
    top_cat_list = payload.get("top_categories", {}).get("current_month") or payload.get("category_spend", [])
    if top_cat_list:
        top_cat = top_cat_list[0]
        top_cat_text = f"{top_cat['name']} ({utils.format_currency(float(top_cat['amount']))}, {top_cat['share']*100:.1f}%)"
    else:
        top_cat_text = "—"

    top_merch_list = payload.get("top_merchants", {}).get("current_month", [])
    if top_merch_list:
        top_merch = top_merch_list[0]
        top_merch_text = f"{top_merch['name']} ({utils.format_currency(float(top_merch['amount']))})"
    else:
        top_merch_text = "—"

    recurring_share = s.get("recurring_share_pct", 0.0)

    total_spend = utils.format_currency(abs(float(s.get("total_spend", 0.0))))
    total_income = utils.format_currency(float(s.get("total_income", 0.0)))
    net_cashflow = utils.format_currency(float(s.get("net_cashflow", 0.0)))

    start = pd.to_datetime(df["posted_date"].min()).date() if "posted_date" in df else pd.to_datetime(df.get("date")).min().date()
    end = pd.to_datetime(df["posted_date"].max()).date() if "posted_date" in df else pd.to_datetime(df.get("date")).max().date()

    lines = [
        f"Across {len(df):,} transactions from {start} to {end}, you spent {total_spend} and received {total_income}, for a net cash flow of {net_cashflow}.",
    ]
    if mom_text:
        lines.append(mom_text)
    lines.append(f"Biggest category: {top_cat_text}.")
    if top_merch_text != "—":
        lines.append(f"Top merchant: {top_merch_text}.")
    lines.append(f"Recurring payments account for {recurring_share:.1f}% of spend.")

    return " " .join(lines)


def summarize_spending(transactions: pd.DataFrame, *, model: str = "gpt-4o-mini") -> str:
    """Summarise spending patterns using an LLM when available."""

    df = utils.ensure_dataframe(transactions)
    # Normalise temporal columns
    if "posted_date" in df:
        base_dates = pd.to_datetime(df["posted_date"])  # type: ignore[assignment]
    elif "date" in df:
        base_dates = pd.to_datetime(df["date"])  # type: ignore[assignment]
        df["posted_date"] = base_dates
    else:
        # Fallback to index if no date column exists
        base_dates = pd.to_datetime(df.index)
        df["posted_date"] = base_dates

    if "month" not in df:
        df["month"] = base_dates.dt.to_period("M").dt.to_timestamp()

    if "abs_amount" not in df:
        df["abs_amount"] = df["amount"].abs()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return _fallback_summary(df)

    client = OpenAI(api_key=api_key)
    prompt = _build_prompt(df)

    try:
        response: Any = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=200,
        )
        text = getattr(response, "output_text", "")
        return text.strip() or _fallback_summary(df)
    except Exception:  # pragma: no cover - network/API errors
        return _fallback_summary(df)
