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
    monthly = (
        df.groupby("month")["amount"].sum().abs().sort_index().tail(3)
    )
    top_merchants = (
        df.groupby("merchant")["abs_amount"].sum().sort_values(ascending=False).head(3)
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
    kpis = insights.calculate_kpis(df)
    bullet_list = "\n".join(f"• {label}: {value}" for label, value in kpis.items())
    return (
        "This is a synthetic dataset for the Spending Analyser scaffold.\n"
        "Highlights:\n"
        f"{bullet_list}"
    )


def summarize_spending(transactions: pd.DataFrame, *, model: str = "gpt-4o-mini") -> str:
    """Summarise spending patterns using an LLM when available."""

    df = utils.ensure_dataframe(transactions)
    if "abs_amount" not in df:
        df["abs_amount"] = df["amount"].abs()
    if "month" not in df:
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()

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
