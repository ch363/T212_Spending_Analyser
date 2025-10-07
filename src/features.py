"""Feature engineering helpers for Spending Analyser."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import utils

ESSENTIAL_CATEGORIES = {
    "Housing",
    "Utilities",
    "Bills",
    "Groceries",
    "Transport",
}

ESSENTIAL_SUBCATEGORIES = {
    ("Food & Drink", "Delivery"),
    ("Food & Drink", "Supermarket"),
    ("Wellness", "Gym"),
}

FIXED_CATEGORIES = {
    "Housing",
    "Utilities",
    "Bills",
    "Wellness",
}


def _is_essential(category: str, subcategory: str) -> bool:
    if category in ESSENTIAL_CATEGORIES:
        return True
    if (category, subcategory) in ESSENTIAL_SUBCATEGORIES:
        return True
    return False


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

    df["is_fixed_spend"] = df["is_recurring"] | df["category"].isin(FIXED_CATEGORIES)
    df.loc[df["is_credit"], "is_fixed_spend"] = False

    df["is_essential"] = df.apply(
        lambda row: _is_essential(row["category"], row.get("subcategory", "")),
        axis=1,
    )
    df.loc[df["is_credit"], "is_essential"] = False

    df["spend_rhythm"] = np.where(df["is_fixed_spend"], "Fixed", "Variable")
    df["spend_necessity"] = np.where(
        df["is_essential"], "Essentials", "Discretionary"
    )

    monthly_net = df.groupby(["user_id", "month"])["amount"].transform("sum")
    df["monthly_net_amount"] = monthly_net.round(2)

    category_share = (
        df.groupby(["user_id", "month", "category"])["abs_amount"].transform("sum")
        / df.groupby(["user_id", "month"])["abs_amount"].transform("sum")
    )
    df["category_month_share"] = category_share.fillna(0).round(4)

    return df
