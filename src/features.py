"""Feature engineering helpers for Spending Analyser."""

from __future__ import annotations

import pandas as pd

from . import utils


def add_engineered_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add basic derived fields to the transactions dataset."""

    df = utils.ensure_dataframe(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["weekday"] = df["date"].dt.day_name()

    recurring_merchants = df["merchant"].value_counts()
    df["is_recurring"] = df["merchant"].map(lambda m: recurring_merchants[m] >= 3)

    df["abs_amount"] = df["amount"].abs()

    return df
