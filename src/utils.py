"""Shared utilities for the Spending Analyser project."""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd


def ensure_dataframe(transactions: Iterable[Mapping] | pd.DataFrame) -> pd.DataFrame:
    """Ensure the input payload is normalised to a :class:`pandas.DataFrame`."""

    if isinstance(transactions, pd.DataFrame):
        return transactions.copy()

    return pd.DataFrame(list(transactions))


def format_currency(value: float, currency: str = "£") -> str:
    """Return a human-readable currency string."""

    return f"{currency}{value:,.2f}"
