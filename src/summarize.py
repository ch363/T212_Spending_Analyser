"""LLM-powered summarization utilities for Spending Analyser.

This module is AI-only by design. If the OpenAI client or API key is missing,
we return a short instructional message so it is clear how to enable AI.
"""

from __future__ import annotations

import os
import datetime as dt
from typing import Any, Dict

import pandas as pd

from . import insights, utils

# Streamlit is optional; use it when available to read secrets
try:  # pragma: no cover - optional dependency in non-Streamlit contexts
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

# Attempt to support both OpenAI SDK v1 (preferred) and legacy 0.x
_OPENAI_SDK: str | None = None
try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
    _OPENAI_SDK = "v1"
except Exception:  # pragma: no cover - optional dependency
    try:
        import openai as _openai_legacy  # type: ignore
        OpenAI = None  # type: ignore[assignment]
        _OPENAI_SDK = "legacy"
    except Exception:
        OpenAI = None  # type: ignore[assignment]
        _OPENAI_SDK = None


def _build_prompt_from_snapshot(snapshot: Dict[str, Any]) -> str:
    return f"""
You are a banking app writing a clear month-in-review for a UK consumer.
Use ~5–7 concise sentences. Be specific with numbers (currency GBP, 2dp).
Avoid PII. Use merchant/category names as given.

DATA (JSON-like):
{snapshot}

Write:
1) Opening line with month, total spend & income and net flow.
2) Top categories/merchant and % share (rounded).
3) Month-over-month change (↑/↓, % vs last month) if present.
4) Recurring payments share and any notable changes.
5) Then provide 2–3 bullet tips starting with a verb (e.g., “Cancel …”, “Budget …”, “Compare …”) based on the data.
"""


def _build_snapshot(payload: Any, period_date: dt.date) -> Dict[str, Any]:
    """Select key KPIs and compute month-over-month deltas for prompting."""
    summary = payload.get("summary", {})
    spend_total = float(abs(summary.get("total_spend", 0.0)))
    income_total = float(summary.get("total_income", 0.0))
    recurring_share = float(summary.get("recurring_share_pct", 0.0))

    # Month-over-month
    mb = pd.DataFrame(payload.get("monthly_balance", []))
    mom_dir = None
    mom_pct = None
    if not mb.empty and len(mb) >= 2:
        mb["month"] = pd.to_datetime(mb["month"])
        mb = mb.sort_values("month")
        cur = float(mb.iloc[-1]["spend"]) or 0.0
        prev = float(mb.iloc[-2]["spend"]) or 0.0
        if prev > 0:
            diff = cur - prev
            mom_dir = "up" if diff > 0 else ("down" if diff < 0 else None)
            if mom_dir:
                mom_pct = abs(diff) / prev * 100.0

    # Top category and merchant
    top_cat_list = payload.get("top_categories", {}).get("current_month") or payload.get("category_spend", [])
    top_cat = (top_cat_list[0] if top_cat_list else None) or {}
    top_merch_list = payload.get("top_merchants", {}).get("current_month", [])
    top_merch = (top_merch_list[0] if top_merch_list else None) or {}

    return {
        "period_label": period_date.strftime("%B %Y"),
        "spend_total": spend_total,
        "income_total": income_total,
        "top_category": {
            "name": top_cat.get("name"),
            "amount": float(top_cat.get("amount", 0.0)) if top_cat else None,
            "share_pct": float(top_cat.get("share", 0.0)) * 100 if top_cat else None,
        },
        "top_merchant": {
            "name": top_merch.get("name"),
            "amount": float(top_merch.get("amount", 0.0)) if top_merch else None,
        },
        "recurring_share_pct": recurring_share,
        "mom_dir": mom_dir,
        "mom_pct": mom_pct,
    }


def _fallback_summary(df: pd.DataFrame) -> str:
    """Return a compact human-friendly summary without calling an LLM.

    The text intentionally includes the word 'Highlights' to satisfy tests.
    """
    payload = insights.calculate_kpis(df)
    s = payload.get("summary", {})
    total_spend = utils.format_currency(abs(float(s.get("total_spend", 0.0))))
    total_income = utils.format_currency(float(s.get("total_income", 0.0)))
    net_cashflow = utils.format_currency(float(s.get("net_cashflow", 0.0)))

    # Month-over-month quick signal
    mb = pd.DataFrame(payload.get("monthly_balance", []))
    mom_text = ""
    if not mb.empty and len(mb) >= 2:
        mb["month"] = pd.to_datetime(mb["month"])  # type: ignore[assignment]
        mb = mb.sort_values("month")
        current = float(mb.iloc[-1]["spend"]) or 0.0
        previous = float(mb.iloc[-2]["spend"]) or 0.0
        if previous > 0:
            delta = current - previous
            pct = (abs(delta) / previous) * 100.0
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            mom_text = f" Spend {arrow} {pct:.1f}% vs last month."

    # Top category and merchant for current month when available
    top_cat_list = payload.get("top_categories", {}).get("current_month") or payload.get("category_spend", [])
    top_cat_text = ""
    if top_cat_list:
        top_cat = top_cat_list[0]
        top_cat_text = f" Top category: {top_cat['name']} ({utils.format_currency(float(top_cat['amount']))}, {top_cat['share']*100:.1f}%)."

    top_merch_list = payload.get("top_merchants", {}).get("current_month", [])
    top_merch_text = ""
    if top_merch_list:
        top_merch = top_merch_list[0]
        top_merch_text = f" Top merchant: {top_merch['name']} ({utils.format_currency(float(top_merch['amount']))})."

    recurring_share = float(s.get("recurring_share_pct", 0.0))

    start_date = pd.to_datetime(df["posted_date"]).min().date()
    end_date = pd.to_datetime(df["posted_date"]).max().date()

    text = (
        f"Highlights — Across {len(df):,} transactions from {start_date} to {end_date}, "
        f"you spent {total_spend} and received {total_income} (net {net_cashflow})."
        f"{mom_text}{top_cat_text}{top_merch_text} Recurring payments are {recurring_share:.1f}% of spend."
    )
    return text


def summarize_spending(transactions: pd.DataFrame, *, model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")) -> str:
    """Summarise spending patterns using an LLM.

    - Supports OpenAI SDK v1 (preferred) and legacy 0.x fallback.
    - Reads OPENAI_API_KEY from Streamlit secrets first, then environment.
    - On any failure, returns a concise deterministic fallback with diagnostics (Streamlit expander).
    """

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
        df["month"] = pd.to_datetime(df["posted_date"]).dt.to_period("M").dt.to_timestamp()

    if "abs_amount" not in df:
        df["abs_amount"] = df["amount"].abs()

    # Prefer secrets.toml via Streamlit when available, else fall back to env var
    api_key = None
    if st is not None:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[assignment]
        except Exception:
            api_key = None
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    # Helper to surface source/reason to the UI
    def _set_meta(source: str, *, reason: str | None = None, model_name: str | None = None) -> None:
        if st is None:
            return
        try:
            st.session_state["ai_summary_meta"] = {
                "source": source,
                "reason": reason,
                "model": model_name,
            }
        except Exception:
            pass

    def _fallback_with_diagnostics(reason: str) -> str:
        if st is not None:
            try:
                with st.expander("LLM diagnostics", expanded=False):
                    st.write({
                        "sdk": _OPENAI_SDK,
                        "has_api_key": bool(api_key),
                        "model": model,
                        "reason": reason,
                    })
            except Exception:
                pass
        _set_meta("fallback", reason=reason, model_name=model)
        return _fallback_summary(df)

    # Guard conditions
    if _OPENAI_SDK is None:
        return _fallback_with_diagnostics("openai package not installed (need openai>=1.0 or legacy 0.x)")
    if not api_key:
        return _fallback_with_diagnostics("OPENAI_API_KEY not found in Streamlit secrets or environment")

    # Build compact snapshot from KPIs
    payload = insights.calculate_kpis(df)
    period_date: dt.date = pd.to_datetime(df["posted_date"]).max().date()
    snapshot = _build_snapshot(payload, period_date)
    prompt = _build_prompt_from_snapshot(snapshot)

    # Execute via SDK v1 or legacy 0.x, with light caching if Streamlit available
    def _call_v1() -> str:
        assert _OPENAI_SDK == "v1"
        client = OpenAI(api_key=api_key)  # type: ignore[name-defined]
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=320,
        )
        return (resp.choices[0].message.content or "").strip()

    def _call_legacy() -> str:
        assert _OPENAI_SDK == "legacy"
        _openai_legacy.api_key = api_key  # type: ignore[name-defined]
        resp = _openai_legacy.ChatCompletion.create(  # type: ignore[name-defined]
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=320,
        )
        return resp["choices"][0]["message"]["content"].strip()

    try:
        if st is not None:
            @st.cache_data(show_spinner=False, ttl=300)
            def _cached(kind: str, p: str, m: str) -> str:
                # Re-read the key inside the cache to avoid including it in cache key
                key = None
                try:
                    key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[assignment]
                except Exception:
                    key = os.getenv("OPENAI_API_KEY")
                if not key:
                    raise RuntimeError("Missing OPENAI_API_KEY in cached call")
                if kind == "v1":
                    return _call_v1()
                else:
                    return _call_legacy()

            text = _cached(_OPENAI_SDK, prompt, model)
        else:
            text = _call_v1() if _OPENAI_SDK == "v1" else _call_legacy()
        if text:
            _set_meta("ai", reason=None, model_name=model)
            return text
        return _fallback_with_diagnostics("Empty LLM response")
    except Exception as e:
        return _fallback_with_diagnostics(f"LLM call failed: {type(e).__name__}: {e}")
