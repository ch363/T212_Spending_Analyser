"""Regenerate a minimal CSV with exactly 200 transactions.

- 100 transactions from the current month
- 100 transactions from the previous month

Both month samples aim to cover all available top-level categories present in
that month (at least one row per category if possible), then fill the remainder
with a deterministic random sample.

Output: data/synthetic_transactions.csv

This script does not touch other files. You can safely delete
data/synthetic_transactions_sample.csv separately if you don't want it.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src import synth


def _month_start(d: date) -> pd.Timestamp:
    ts = pd.Timestamp(d)
    return ts.to_period("M").to_timestamp()


def _month_end(d: date) -> pd.Timestamp:
    ts = pd.Timestamp(d)
    start = ts.to_period("M").to_timestamp()
    next_start = (start + pd.offsets.MonthEnd(1)).normalize()
    # MonthEnd returns end-of-month relative; derive end date as next_start - 1 day
    return (next_start - pd.Timedelta(days=1)).normalize()


def _filter_month(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    mask = (df["posted_date"].dt.year == year) & (df["posted_date"].dt.month == month)
    return df.loc[mask].copy()


def _stratified_sample_by_category(df: pd.DataFrame, target: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if df.empty:
        return df

    # Ensure at least one of each category (if available)
    parts: list[pd.DataFrame] = []
    for cat, group in df.groupby("category"):
        pick = group.sample(n=1, random_state=int(rng.integers(0, 1_000_000)))
        parts.append(pick)

    base = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=df.columns)
    # Fill remaining rows with random sample from what is left; pad with replacement if needed
    remaining = target - len(base)
    if remaining > 0:
        pool = df.drop(index=base.index, errors="ignore")
        if len(pool) > 0:
            take = min(remaining, len(pool))
            extra = pool.sample(n=take, random_state=int(rng.integers(0, 1_000_000)))
            base = pd.concat([base, extra], ignore_index=True)
            remaining -= len(extra)
        if remaining > 0:
            # Pad by sampling with replacement from the full month to reach exact target
            pad = df.sample(n=remaining, replace=True, random_state=int(rng.integers(0, 1_000_000)))
            base = pd.concat([base, pad], ignore_index=True)

    # If we overshot (can happen if categories > target), trim deterministically
    if len(base) > target:
        base = base.sample(n=target, random_state=seed).reset_index(drop=True)

    return base


def _recompute_balances(df: pd.DataFrame, starting_balance: float = 2_500.0) -> pd.DataFrame:
    df = df.sort_values(["posted_date", "txn_id"]).reset_index(drop=True).copy()
    df["balance_after"] = (starting_balance + df["amount"].cumsum()).round(2)
    # Ensure dtypes and rounding consistency
    df["amount"] = df["amount"].round(2)
    return df


def build_minimal_two_months(rows_per_month: int, *, seed: int) -> pd.DataFrame:
    today = date.today()
    this_year, this_month = today.year, today.month
    # Previous month handling across year boundary
    if this_month == 1:
        prev_year, prev_month = this_year - 1, 12
    else:
        prev_year, prev_month = this_year, this_month - 1

    # Generate a sufficiently large synthetic set, then filter to the two months
    # We overshoot to ensure enough rows for stratification.
    base = synth.generate_transactions(rows=10_000, seed=seed)
    base["posted_date"] = pd.to_datetime(base["posted_date"])  # guard dtype

    this_df = _filter_month(base, this_year, this_month)
    prev_df = _filter_month(base, prev_year, prev_month)

    # If either month is too sparse (shouldn't happen with generator), fall back to
    # building from the nearest available months.
    if this_df.empty or prev_df.empty:
        # Try to expand search window by +/- 1 month if needed
        # but keep the final counts exact.
        months: list[tuple[int, int, pd.DataFrame]] = []
        for offset in range(-2, 3):
            ts = (pd.Timestamp(this_year, this_month, 1) + pd.DateOffset(months=offset)).to_pydatetime().date()
            mdf = _filter_month(base, ts.year, ts.month)
            months.append((ts.year, ts.month, mdf))
        # Prefer most recent two non-empty months
        non_empty = [(y, m, d) for (y, m, d) in months if not d.empty]
        non_empty = sorted(non_empty, key=lambda t: (t[0], t[1]))[-2:]
        if len(non_empty) == 2:
            (this_year, this_month, this_df), (prev_year, prev_month, prev_df) = non_empty

    this_sample = _stratified_sample_by_category(this_df, rows_per_month, seed=seed + 1)
    prev_sample = _stratified_sample_by_category(prev_df, rows_per_month, seed=seed + 2)

    minimal = pd.concat([prev_sample, this_sample], ignore_index=True)
    minimal = _recompute_balances(minimal, starting_balance=2_500.0)
    # Enforce column order like synth.generate_transactions
    ordered_columns = [
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
    ]
    return minimal[ordered_columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate minimal two-month synthetic CSV (200 rows)")
    parser.add_argument("--rows-per-month", type=int, default=100)
    parser.add_argument("--seed", type=int, default=synth.DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=Path("data") / "synthetic_transactions.csv")
    args = parser.parse_args()

    df = build_minimal_two_months(args.rows_per_month, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(df)} rows")


if __name__ == "__main__":
    main()
