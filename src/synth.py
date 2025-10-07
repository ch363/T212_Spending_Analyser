"""Synthetic data generation utilities for Spending Analyser."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

CATEGORIES = [
    "Groceries",
    "Rent",
    "Utilities",
    "Transport",
    "Eating Out",
    "Entertainment",
    "Health",
    "Subscriptions",
]

MERCHANTS = [
    "Tesco",
    "Sainsbury's",
    "Shell",
    "Uber",
    "Netflix",
    "Spotify",
    "Amazon",
    "Pret A Manger",
]


def generate_sample_transactions(rows: int = 60, seed: int | None = 7) -> pd.DataFrame:
    """Create a reproducible synthetic transactions table."""

    rng = np.random.default_rng(seed)
    base_date = datetime.now().date() - timedelta(days=90)

    dates = [base_date + timedelta(days=int(rng.integers(0, 90))) for _ in range(rows)]
    amounts = np.round(rng.uniform(5, 250, size=rows), 2) * -1
    categories = rng.choice(CATEGORIES, size=rows)
    merchants = rng.choice(MERCHANTS, size=rows)

    data: dict[str, Any] = {
        "transaction_id": [f"txn-{i:04d}" for i in range(1, rows + 1)],
        "date": dates,
        "merchant": merchants,
        "category": categories,
        "amount": amounts,
    }

    return pd.DataFrame(data).sort_values("date").reset_index(drop=True)
