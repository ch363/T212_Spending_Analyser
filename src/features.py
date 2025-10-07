"""Feature engineering helpers for Spending Analyser."""

from __future__ import annotations

import pandas as pd

from . import utils


def add_engineered_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add derived fields required by downstream analytics and visuals."""

    df = utils.ensure_dataframe(transactions).copy()

    df["posted_date"] = pd.to_datetime(df["posted_date"])
    df["value_date"] = pd.to_datetime(df["value_date"])

    df["month"] = df["posted_date"].dt.to_period("M").dt.to_timestamp()
    df["weekday"] = df["posted_date"].dt.day_name()
    df["day"] = df["posted_date"].dt.day

    df["merchant"] = df["merchant_name"]
    df["is_recurring"] = df["recurring_id"].notna()
    df["is_credit"] = df["amount"] > 0
    df["abs_amount"] = df["amount"].abs()

    monthly_net = df.groupby(["user_id", "month"])["amount"].transform("sum")
    df["monthly_net_amount"] = monthly_net.round(2)

    category_share = (
        df.groupby(["user_id", "month", "category"])["abs_amount"].transform("sum")
        / df.groupby(["user_id", "month"])["abs_amount"].transform("sum")
    )
    df["category_month_share"] = category_share.fillna(0).round(4)

    return df
