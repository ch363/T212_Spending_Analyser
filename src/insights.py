"""Insights and aggregation helpers for Spending Analyser."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from . import utils


class SummaryMetrics(TypedDict):
    total_spend: float
    total_income: float
    net_cashflow: float
    avg_daily_spend: float
    recurring_share_pct: float


class MonthlyBalance(TypedDict):
    month: str
    income: float
    spend: float
    net: float


class BreakdownEntry(TypedDict):
    name: str
    amount: float
    share: float


class SpendSplit(TypedDict):
    fixed: float
    variable: float
    essentials: float
    discretionary: float


class CashProjection(TypedDict):
    as_of: str
    projected_end_balance: float
    projected_burn: float
    daily_burn_rate: float
    days_remaining: int


class PaydayStats(TypedDict):
    next_payday: str | None
    days_to_next_payday: int | None


class RecurringInsight(TypedDict):
    merchant: str
    recurring_id: str | None
    avg_amount: float
    last_amount: float
    delta_amount: float
    delta_percent: float
    status: Literal["on_track", "missed", "extra"]
    expected_interval_days: float | None
    last_paid_date: str


class AnomalyInsight(TypedDict):
    txn_id: str
    posted_date: str
    merchant: str
    amount: float
    z_score: float
    category: str


class InsightsPayload(TypedDict):
    summary: SummaryMetrics
    monthly_balance: list[MonthlyBalance]
    top_categories: dict[str, list[BreakdownEntry]]
    top_merchants: dict[str, list[BreakdownEntry]]
    spend_split: SpendSplit
    cash_projection: CashProjection
    payday: PaydayStats
    recurring: list[RecurringInsight]
    anomalies: list[AnomalyInsight]
    category_spend: list[BreakdownEntry]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["posted_date"] = pd.to_datetime(df["posted_date"])  # type: ignore[assignment]
    df["amount"] = df["amount"].astype(float)
    if "abs_amount" not in df:
        df["abs_amount"] = df["amount"].abs()
    if "is_recurring" not in df:
        if "recurring_id" in df:
            df["is_recurring"] = df["recurring_id"].notna()
        else:
            df["is_recurring"] = False
    if "is_credit" not in df:
        df["is_credit"] = df["amount"] > 0
    if "is_fixed_spend" not in df:
        df["is_fixed_spend"] = df["is_recurring"] & (~df["is_credit"])
    if "spend_necessity" not in df:
        df["spend_necessity"] = np.where(
            df["is_recurring"], "Essentials", "Discretionary"
        )
    if "month" not in df:
        df["month"] = df["posted_date"].dt.to_period("M").dt.to_timestamp()
    return df


def _format_month(ts: pd.Timestamp) -> str:
    return ts.to_pydatetime().strftime("%Y-%m-%d")


def _top_breakdown(
    frame: pd.DataFrame,
    group_column: str,
    limit: int = 5,
) -> list[BreakdownEntry]:
    if frame.empty:
        return []

    spend = frame.loc[frame["amount"] < 0].copy()
    if spend.empty:
        return []

    spend["value"] = spend["amount"].abs()
    totals = (
        spend.groupby(group_column)["value"]
        .sum()
        .sort_values(ascending=False)
        .head(limit)
    )
    overall = spend["value"].sum()
    return [
        {
            "name": str(name),
            "amount": float(value),
            "share": float(value / overall) if overall else 0.0,
        }
        for name, value in totals.items()
    ]


def _monthly_balance(df: pd.DataFrame) -> list[MonthlyBalance]:
    if df.empty:
        return []
    grouped = df.groupby("month")
    records = []
    for month, bucket in grouped:
        credits = bucket.loc[bucket["amount"] > 0, "amount"].sum()
        debits = -bucket.loc[bucket["amount"] < 0, "amount"].sum()
        records.append(
            {
                "month": _format_month(month if isinstance(month, pd.Timestamp) else month.to_timestamp()),
                "income": float(credits),
                "spend": float(debits),
                "net": float(credits - debits),
            }
        )
    records.sort(key=lambda row: row["month"])
    return records


def _window_frame(df: pd.DataFrame, reference: pd.Timestamp, months: int) -> pd.DataFrame:
    end = (reference.to_period("M") + 1).to_timestamp()
    start_period = reference.to_period("M") - (months - 1)
    start = start_period.to_timestamp()
    return df.loc[(df["posted_date"] >= start) & (df["posted_date"] < end)].copy()


def _spend_split(df: pd.DataFrame) -> SpendSplit:
    spend = df.loc[df["amount"] < 0].copy()
    if spend.empty:
        return {"fixed": 0.0, "variable": 0.0, "essentials": 0.0, "discretionary": 0.0}

    spend["value"] = spend["amount"].abs()

    fixed = spend.loc[spend.get("is_fixed_spend", False), "value"].sum()
    essentials = spend.loc[spend.get("spend_necessity") == "Essentials", "value"].sum()
    total = spend["value"].sum()
    variable = total - fixed
    discretionary = total - essentials

    return {
        "fixed": float(fixed),
        "variable": float(max(variable, 0.0)),
        "essentials": float(essentials),
        "discretionary": float(max(discretionary, 0.0)),
    }


def _monthly_bounds(reference: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = reference.to_period("M").to_timestamp()
    end = (reference.to_period("M") + 1).to_timestamp() - pd.Timedelta(days=1)
    return start, end


def _cash_projection(df: pd.DataFrame, reference: pd.Timestamp) -> CashProjection:
    if df.empty:
        return {
            "as_of": reference.strftime("%Y-%m-%d"),
            "projected_end_balance": 0.0,
            "projected_burn": 0.0,
            "daily_burn_rate": 0.0,
            "days_remaining": 0,
        }

    month_start, month_end = _monthly_bounds(reference)
    current_month = df.loc[
        (df["posted_date"] >= month_start) & (df["posted_date"] <= reference)
    ].copy()

    if current_month.empty:
        days_remaining = int((month_end - reference).days)
        return {
            "as_of": reference.strftime("%Y-%m-%d"),
            "projected_end_balance": float(df.get("balance_after", pd.Series([0])).iloc[-1]),
            "projected_burn": 0.0,
            "daily_burn_rate": 0.0,
            "days_remaining": max(days_remaining, 0),
        }

    days_elapsed = max((reference - month_start).days + 1, 1)
    days_remaining = max((month_end - reference).days, 0)

    spend = current_month.loc[current_month["amount"] < 0, "amount"].abs().sum()
    income = current_month.loc[current_month["amount"] > 0, "amount"].sum()

    daily_burn_rate = float(spend / days_elapsed) if spend else 0.0
    daily_income = float(income / days_elapsed) if income else 0.0

    projected_burn = daily_burn_rate * days_remaining
    projected_income = daily_income * days_remaining

    if "balance_after" in df:
        latest_balance = float(
            df.sort_values("posted_date")["balance_after"].iloc[-1]
        )
    else:
        latest_balance = float(df.sort_values("posted_date")["amount"].cumsum().iloc[-1])

    projected_end_balance = latest_balance - projected_burn + projected_income

    return {
        "as_of": reference.strftime("%Y-%m-%d"),
        "projected_end_balance": float(projected_end_balance),
        "projected_burn": float(projected_burn),
        "daily_burn_rate": float(daily_burn_rate),
        "days_remaining": int(days_remaining),
    }


def _next_payday(reference: date) -> date:
    target = date(reference.year, reference.month, 25)
    if reference > target:
        if reference.month == 12:
            target = date(reference.year + 1, 1, 25)
        else:
            target = date(reference.year, reference.month + 1, 25)

    while target.weekday() >= 5:
        target -= timedelta(days=1)
    return target


def _payday_stats(reference: pd.Timestamp) -> PaydayStats:
    ref_date = reference.date()
    payday = _next_payday(ref_date)
    days = (payday - ref_date).days
    return {
        "next_payday": payday.isoformat(),
        "days_to_next_payday": int(days),
    }


def _recurring_insights(df: pd.DataFrame, reference: pd.Timestamp) -> list[RecurringInsight]:
    recurring = df.loc[
        (df["amount"] < 0) & df.get("is_recurring", False)
    ].copy()
    if recurring.empty:
        return []

    insights: list[RecurringInsight] = []
    grouped = recurring.groupby("recurring_id", dropna=False)
    for rid, group in grouped:
        group = group.sort_values("posted_date")
        values = group["amount"].abs()
        avg_amount = float(values.mean()) if not values.empty else 0.0
        last_amount = float(values.iloc[-1]) if not values.empty else 0.0
        delta_amount = last_amount - avg_amount
        delta_percent = (delta_amount / avg_amount * 100.0) if avg_amount else 0.0

        deltas = group["posted_date"].diff().dt.days.dropna()
        expected_interval = float(deltas.median()) if not deltas.empty else None

        last_paid = group["posted_date"].iloc[-1]
        days_since_last = (reference - last_paid).days

        status: Literal["on_track", "missed", "extra"] = "on_track"
        if expected_interval:
            if days_since_last > expected_interval * 1.6:
                status = "missed"
            elif days_since_last < expected_interval * 0.6:
                status = "extra"

        merchant = group.get("merchant_name")
        merchant_name = (
            merchant.mode().iat[0]
            if isinstance(merchant, pd.Series) and not merchant.empty
            else "Unknown"
        )

        insights.append(
            {
                "merchant": str(merchant_name),
                "recurring_id": str(rid) if pd.notna(rid) else None,
                "avg_amount": float(avg_amount),
                "last_amount": float(last_amount),
                "delta_amount": float(delta_amount),
                "delta_percent": float(delta_percent),
                "status": status,
                "expected_interval_days": expected_interval,
                "last_paid_date": last_paid.strftime("%Y-%m-%d"),
            }
        )

    insights.sort(key=lambda item: abs(item["delta_percent"]), reverse=True)
    return insights[:10]


def _anomaly_insights(df: pd.DataFrame) -> list[AnomalyInsight]:
    spend = df.loc[df["amount"] < 0].copy()
    if spend.empty:
        return []

    stats = spend.groupby("merchant_name")["abs_amount"].agg(["mean", "std"])
    spend = spend.join(stats, on="merchant_name", rsuffix="_merchant")
    spend["std"] = spend["std"].replace(0, np.nan)
    spend["z_score"] = (spend["abs_amount"] - spend["mean"]) / spend["std"]
    spend.loc[~np.isfinite(spend["z_score"]), "z_score"] = np.nan
    anomalies = spend.loc[spend["z_score"].abs() >= 2.5].copy()
    if anomalies.empty:
        return []

    anomalies = anomalies.sort_values("z_score", ascending=False)
    return [
        {
            "txn_id": str(row.get("transaction_id") or row.get("id", "")),
            "posted_date": pd.to_datetime(row["posted_date"]).strftime("%Y-%m-%d"),
            "merchant": str(row.get("merchant_name", "Unknown")),
            "amount": float(-row["amount"]),
            "z_score": float(row["z_score"]),
            "category": str(row.get("category", "")),
        }
        for _, row in anomalies.head(20).iterrows()
    ]


def _category_spend(df: pd.DataFrame) -> list[BreakdownEntry]:
    spend = df.loc[df["amount"] < 0].copy()
    if spend.empty:
        return []
    totals = spend.groupby("category")["amount"].sum().sort_values()
    overall = spend["amount"].abs().sum()
    return [
        {
            "name": str(category),
            "amount": float(-value),
            "share": float((-value) / overall) if overall else 0.0,
        }
        for category, value in totals.items()
    ]


def calculate_kpis(transactions: pd.DataFrame) -> InsightsPayload:
    """Compute structured KPI outputs ready for UI charts and tables."""

    df = _ensure_columns(utils.ensure_dataframe(transactions).copy())

    if df.empty:
        zero_summary: SummaryMetrics = {
            "total_spend": 0.0,
            "total_income": 0.0,
            "net_cashflow": 0.0,
            "avg_daily_spend": 0.0,
            "recurring_share_pct": 0.0,
        }
        return {
            "summary": zero_summary,
            "monthly_balance": [],
            "top_categories": {},
            "top_merchants": {},
            "spend_split": {"fixed": 0.0, "variable": 0.0, "essentials": 0.0, "discretionary": 0.0},
            "cash_projection": {
                "as_of": datetime.utcnow().strftime("%Y-%m-%d"),
                "projected_end_balance": 0.0,
                "projected_burn": 0.0,
                "daily_burn_rate": 0.0,
                "days_remaining": 0,
            },
            "payday": {"next_payday": None, "days_to_next_payday": None},
            "recurring": [],
            "anomalies": [],
            "category_spend": [],
        }

    reference_date = df["posted_date"].max().normalize()
    debits = df.loc[df["amount"] < 0, "amount"]
    credits = df.loc[df["amount"] > 0, "amount"]

    total_spend = float(-debits.sum())
    total_income = float(credits.sum())
    net_cashflow = float(total_income - total_spend)

    min_date = df["posted_date"].min().normalize()
    days_span = max((reference_date - min_date).days + 1, 1)
    avg_daily_spend = float(total_spend / days_span) if total_spend else 0.0

    recurring_share_pct = float(
        df.loc[df["amount"] < 0, "is_recurring"].mean() * 100.0
    ) if (df["amount"] < 0).any() else 0.0

    summary: SummaryMetrics = {
        "total_spend": round(total_spend, 2),
        "total_income": round(total_income, 2),
        "net_cashflow": round(net_cashflow, 2),
        "avg_daily_spend": round(avg_daily_spend, 2),
        "recurring_share_pct": round(recurring_share_pct, 2),
    }

    windows = {
        "current_month": 1,
        "last_3_months": 3,
        "last_6_months": 6,
        "last_12_months": 12,
    }

    top_categories = {
        name: _top_breakdown(_window_frame(df, reference_date, months), "category")
        for name, months in windows.items()
    }
    top_merchants = {
        name: _top_breakdown(_window_frame(df, reference_date, months), "merchant_name")
        for name, months in windows.items()
    }

    payload: InsightsPayload = {
        "summary": summary,
        "monthly_balance": _monthly_balance(df),
        "top_categories": top_categories,
        "top_merchants": top_merchants,
        "spend_split": _spend_split(df),
        "cash_projection": _cash_projection(df, reference_date),
        "payday": _payday_stats(reference_date),
        "recurring": _recurring_insights(df, reference_date),
        "anomalies": _anomaly_insights(df),
        "category_spend": _category_spend(df),
    }

    return payload
