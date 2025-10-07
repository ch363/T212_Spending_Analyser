"""Stage 1 synthetic data generation utilities.

The generator produces deterministic, realistic debit card transaction ledgers with
seasonality, merchant catalogues, and edge cases such as refunds and foreign currency.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import numpy as np
import pandas as pd

DEFAULT_DATASET_ROWS = 10_000
DEFAULT_SAMPLE_ROWS = 200
DEFAULT_SEED = 7

CARD_LAST4 = ("1045", "2219", "4488", "9074")
USER_IDS = ("user_001", "user_002")
USER_WEIGHTS = (0.82, 0.18)


@dataclass(frozen=True)
class MerchantProfile:
    """Static metadata for a merchant."""

    name: str
    mcc: str
    category: str
    subcategory: str
    channel: str
    city: str
    country: str
    amount_range: tuple[float, float]
    flow: str = "debit"  # "debit" or "credit"
    recurring: bool = False
    currency_options: tuple[str, ...] = ("GBP",)


def _merchant_catalogue() -> dict[str, MerchantProfile]:
    """Return keyed merchant catalogue for deterministic lookups."""

    merchants = [
        MerchantProfile(
            name="Crown Estates Lettings",
            mcc="6513",
            category="Housing",
            subcategory="Rent",
            channel="POS",
            city="London",
            country="GB",
            amount_range=(1450.0, 1650.0),
            recurring=True,
        ),
        MerchantProfile(
            name="Octopus Energy",
            mcc="4900",
            category="Utilities",
            subcategory="Electricity",
            channel="Direct Debit",
            city="London",
            country="GB",
            amount_range=(110.0, 145.0),
            recurring=True,
        ),
        MerchantProfile(
            name="Thames Water",
            mcc="4900",
            category="Utilities",
            subcategory="Water",
            channel="Direct Debit",
            city="Reading",
            country="GB",
            amount_range=(32.0, 45.0),
            recurring=True,
        ),
        MerchantProfile(
            name="City Council Tax",
            mcc="9399",
            category="Utilities",
            subcategory="Council Tax",
            channel="Direct Debit",
            city="London",
            country="GB",
            amount_range=(150.0, 190.0),
            recurring=True,
        ),
        MerchantProfile(
            name="O2 UK",
            mcc="4814",
            category="Bills",
            subcategory="Mobile",
            channel="Direct Debit",
            city="Slough",
            country="GB",
            amount_range=(35.0, 55.0),
            recurring=True,
        ),
        MerchantProfile(
            name="Spotify",
            mcc="5735",
            category="Entertainment",
            subcategory="Subscriptions",
            channel="E-Com",
            city="Stockholm",
            country="SE",
            amount_range=(7.99, 9.99),
            recurring=True,
            currency_options=("GBP", "EUR"),
        ),
        MerchantProfile(
            name="Netflix",
            mcc="7841",
            category="Entertainment",
            subcategory="Subscriptions",
            channel="E-Com",
            city="Los Gatos",
            country="US",
            amount_range=(9.99, 15.99),
            recurring=True,
            currency_options=("GBP", "USD"),
        ),
        MerchantProfile(
            name="Pret A Manger",
            mcc="5814",
            category="Food & Drink",
            subcategory="Eating Out",
            channel="POS",
            city="London",
            country="GB",
            amount_range=(5.5, 12.0),
        ),
        MerchantProfile(
            name="Tesco",
            mcc="5411",
            category="Groceries",
            subcategory="Supermarket",
            channel="POS",
            city="London",
            country="GB",
            amount_range=(18.0, 85.0),
        ),
        MerchantProfile(
            name="Sainsbury's",
            mcc="5411",
            category="Groceries",
            subcategory="Supermarket",
            channel="POS",
            city="London",
            country="GB",
            amount_range=(14.0, 70.0),
        ),
        MerchantProfile(
            name="Deliveroo",
            mcc="5812",
            category="Food & Drink",
            subcategory="Delivery",
            channel="E-Com",
            city="London",
            country="GB",
            amount_range=(12.0, 34.0),
        ),
        MerchantProfile(
            name="Uber UK",
            mcc="4121",
            category="Transport",
            subcategory="Ride Hailing",
            channel="E-Com",
            city="London",
            country="GB",
            amount_range=(8.0, 36.0),
        ),
        MerchantProfile(
            name="Transport for London",
            mcc="4111",
            category="Transport",
            subcategory="Transit",
            channel="POS",
            city="London",
            country="GB",
            amount_range=(3.5, 12.0),
        ),
        MerchantProfile(
            name="Airbnb",
            mcc="7011",
            category="Travel",
            subcategory="Accommodation",
            channel="E-Com",
            city="Dublin",
            country="IE",
            amount_range=(120.0, 420.0),
            currency_options=("GBP", "EUR", "USD"),
        ),
        MerchantProfile(
            name="British Airways",
            mcc="4511",
            category="Travel",
            subcategory="Flights",
            channel="E-Com",
            city="London",
            country="GB",
            amount_range=(180.0, 620.0),
            currency_options=("GBP", "EUR", "USD"),
        ),
        MerchantProfile(
            name="Local Market",
            mcc="5499",
            category="Groceries",
            subcategory="Specialty",
            channel="POS",
            city="London",
            country="GB",
            amount_range=(6.0, 22.0),
        ),
        MerchantProfile(
            name="GymBox",
            mcc="7997",
            category="Wellness",
            subcategory="Gym",
            channel="Direct Debit",
            city="London",
            country="GB",
            amount_range=(45.0, 65.0),
            recurring=True,
        ),
        MerchantProfile(
            name="T212 Capital",
            mcc="6011",
            category="Income",
            subcategory="Salary",
            channel="Credit",
            city="London",
            country="GB",
            amount_range=(2850.0, 3450.0),
            flow="credit",
            recurring=True,
        ),
        MerchantProfile(
            name="HM Revenue & Customs",
            mcc="9311",
            category="Income",
            subcategory="Tax Refund",
            channel="Credit",
            city="London",
            country="GB",
            amount_range=(150.0, 450.0),
            flow="credit",
        ),
        MerchantProfile(
            name="Amazon UK",
            mcc="5942",
            category="Shopping",
            subcategory="Online Retail",
            channel="E-Com",
            city="London",
            country="GB",
            amount_range=(9.0, 120.0),
        ),
        MerchantProfile(
            name="Apple.com/bill",
            mcc="5734",
            category="Shopping",
            subcategory="Electronics",
            channel="E-Com",
            city="Cupertino",
            country="US",
            amount_range=(0.99, 210.0),
            currency_options=("GBP", "USD"),
        ),
        MerchantProfile(
            name="ATM Withdrawal",
            mcc="6011",
            category="Cash",
            subcategory="ATM",
            channel="ATM",
            city="London",
            country="GB",
            amount_range=(60.0, 120.0),
        ),
        MerchantProfile(
            name="Waitrose",
            mcc="5411",
            category="Groceries",
            subcategory="Supermarket",
            channel="POS",
            city="London",
            country="GB",
            amount_range=(20.0, 90.0),
        ),
        MerchantProfile(
            name="Shell",
            mcc="5541",
            category="Transport",
            subcategory="Fuel",
            channel="POS",
            city="London",
            country="GB",
            amount_range=(30.0, 95.0),
        ),
        MerchantProfile(
            name="Airline Refund",
            mcc="4511",
            category="Travel",
            subcategory="Refund",
            channel="Credit",
            city="London",
            country="GB",
            amount_range=(60.0, 250.0),
            flow="credit",
        ),
    ]

    return {m.name: m for m in merchants}


CATALOGUE = _merchant_catalogue()

DAILY_SPEND_MERCHANTS = (
    "Tesco",
    "Sainsbury's",
    "Pret A Manger",
    "Deliveroo",
    "Uber UK",
    "Transport for London",
    "Amazon UK",
    "Apple.com/bill",
    "Local Market",
    "Waitrose",
    "Shell",
)

TRAVEL_MERCHANTS = ("Airbnb", "British Airways")
REFUND_MERCHANTS = ("Airline Refund", "HM Revenue & Customs")
RECURRING_MERCHANTS = (
    "Crown Estates Lettings",
    "Octopus Energy",
    "Thames Water",
    "City Council Tax",
    "O2 UK",
    "Spotify",
    "Netflix",
    "GymBox",
    "T212 Capital",
)


def _business_day(candidate: date, *, direction: str = "forward") -> date:
    """Adjust dates falling on weekend to nearest business day."""

    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be 'forward' or 'backward'")

    delta = 1 if direction == "forward" else -1
    adjusted = candidate
    while adjusted.weekday() >= 5:
        adjusted += timedelta(days=delta)
    return adjusted


def _value_date(posted: date, *, delay_days: int) -> date:
    """Calculate banking value date with weekend roll-forward."""

    value = posted + timedelta(days=delay_days)
    while value.weekday() >= 5:
        value += timedelta(days=1)
    return value


def _recurring_id(merchant: MerchantProfile, user_id: str) -> str:
    digest = hashlib.sha1(f"{merchant.name}:{user_id}".encode(), usedforsecurity=False).hexdigest()
    return f"rec_{digest[:10]}"


def _choose_currency(merchant: MerchantProfile, rng: np.random.Generator) -> str:
    options = merchant.currency_options
    if len(options) == 1:
        return options[0]
    weights = np.full(len(options), 1.0 / len(options))
    # Slightly favour GBP if present.
    for idx, code in enumerate(options):
        if code == "GBP":
            weights[idx] *= 2.0
    weights /= weights.sum()
    return rng.choice(options, p=weights)


def _fx_note(currency: str, amount: float, rng: np.random.Generator) -> str:
    if currency == "GBP":
        return ""
    # Simulate a simple FX rate around common averages.
    rate_lookup = {"USD": 1.24, "EUR": 1.14}
    base_rate = rate_lookup.get(currency, 1.0)
    jitter = rng.normal(0, 0.03)
    rate = max(0.5, base_rate + jitter)
    foreign_amount = round(abs(amount) * rate, 2)
    return f"FX purchase {foreign_amount} {currency} @ rate {rate:.2f}"


def _uuid4_from_rng(rng: np.random.Generator) -> str:
    raw = bytearray(rng.bytes(16))
    raw[6] = (raw[6] & 0x0F) | 0x40  # version 4
    raw[8] = (raw[8] & 0x3F) | 0x80  # variant 10
    return str(UUID(bytes=bytes(raw)))


def _build_transaction(
    merchant: MerchantProfile,
    *,
    posted_date: date,
    rng: np.random.Generator,
    amount_override: float | None = None,
    user_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    min_amt, max_amt = merchant.amount_range
    magnitude = amount_override if amount_override is not None else rng.uniform(min_amt, max_amt)
    magnitude = round(magnitude, 2)
    amount = magnitude if merchant.flow == "credit" else -magnitude

    currency = _choose_currency(merchant, rng)
    fx_note = _fx_note(currency, amount, rng)

    selected_user = user_id or rng.choice(USER_IDS, p=USER_WEIGHTS)
    recurring_id = (
        _recurring_id(merchant, selected_user)
        if merchant.recurring
        else pd.NA
    )

    delay = int(rng.integers(0, 3)) if merchant.channel != "Credit" else 0
    value_date = _value_date(posted_date, delay_days=delay)

    notes = note or fx_note or pd.NA

    return {
        "txn_id": _uuid4_from_rng(rng),
        "posted_date": pd.Timestamp(posted_date),
        "value_date": pd.Timestamp(value_date),
        "merchant_name": merchant.name,
        "mcc": merchant.mcc,
        "category": merchant.category,
        "subcategory": merchant.subcategory,
        "amount": round(amount, 2),
        "currency": currency,
        "balance_after": 0.0,  # placeholder updated later
        "channel": merchant.channel,
        "location_city": merchant.city,
        "location_country": merchant.country,
        "card_last4": rng.choice(CARD_LAST4),
        "user_id": selected_user,
        "recurring_id": recurring_id,
        "notes": notes,
    }


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    return (next_month - date(year, month, 1)).days


def _generate_month_transactions(year: int, month: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    catalogue = CATALOGUE
    transactions: list[dict[str, Any]] = []
    month_start = date(year, month, 1)

    # Salary on 25th shifting to previous business day.
    salary_date = _business_day(date(year, month, 25), direction="backward")
    transactions.append(
        _build_transaction(catalogue["T212 Capital"], posted_date=salary_date, rng=rng)
    )

    # Recurring bills
    scheduled = [
        (catalogue["Crown Estates Lettings"], _business_day(month_start, direction="forward")),
        (catalogue["Octopus Energy"], _business_day(month_start + timedelta(days=6), direction="forward")),
        (catalogue["Thames Water"], _business_day(month_start + timedelta(days=10), direction="forward")),
        (catalogue["City Council Tax"], _business_day(month_start + timedelta(days=12), direction="forward")),
        (catalogue["O2 UK"], _business_day(month_start + timedelta(days=14), direction="forward")),
        (catalogue["GymBox"], _business_day(month_start + timedelta(days=16), direction="forward")),
        (catalogue["Spotify"], _business_day(month_start + timedelta(days=18), direction="forward")),
        (catalogue["Netflix"], _business_day(month_start + timedelta(days=20), direction="forward")),
    ]

    for merchant, when in scheduled:
        transactions.append(_build_transaction(merchant, posted_date=when, rng=rng))

    # Daily discretionary spending using Poisson distribution.
    days = _days_in_month(year, month)
    for offset in range(days):
        current_day = month_start + timedelta(days=offset)
        weekday = current_day.weekday()
        lam = 1.8 if weekday < 5 else 2.6
        count = int(max(0, rng.poisson(lam)))
        if weekday >= 5:
            count += 1
        for _ in range(count):
            merchant_name = rng.choice(DAILY_SPEND_MERCHANTS)
            merchant = catalogue[merchant_name]
            transactions.append(_build_transaction(merchant, posted_date=current_day, rng=rng))

    # Occasional ATM withdrawals (twice per month on average).
    atm_count = int(rng.poisson(1.5))
    for _ in range(atm_count):
        day_offset = int(rng.integers(0, days))
        atm_date = month_start + timedelta(days=day_offset)
        transactions.append(_build_transaction(catalogue["ATM Withdrawal"], posted_date=atm_date, rng=rng))

    # Travel spikes quarterly on average.
    if rng.random() < 0.3:
        travel_merchant = catalogue[rng.choice(TRAVEL_MERCHANTS)]
        trip_day = month_start + timedelta(days=int(rng.integers(5, min(27, days))))
        trip_note = "Business travel" if rng.random() < 0.4 else "Holiday booking"
        transactions.append(
            _build_transaction(travel_merchant, posted_date=trip_day, rng=rng, note=trip_note)
        )

    return transactions


def _inject_refunds(transactions: list[dict[str, Any]], rng: np.random.Generator) -> None:
    """Add refund rows referencing existing debits."""

    debit_indices = [idx for idx, txn in enumerate(transactions) if float(txn["amount"]) < 0]
    rng.shuffle(debit_indices)
    refund_count = max(4, int(len(transactions) * 0.02))

    for idx in debit_indices[:refund_count]:
        original = cast(dict[str, Any], transactions[idx])
        merchant = CATALOGUE[rng.choice(REFUND_MERCHANTS)]
        original_posted = pd.Timestamp(original["posted_date"]).date()
        posted_date = original_posted + timedelta(days=int(rng.integers(7, 21)))
        amount_override = abs(float(original["amount"])) * rng.uniform(0.4, 1.0)
        note = f"Refund for {original['merchant_name']}"
        refund_txn = _build_transaction(
            merchant,
            posted_date=posted_date,
            rng=rng,
            amount_override=round(amount_override, 2),
            user_id=cast(str, original["user_id"]),
            note=note,
        )
        transactions.append(refund_txn)


def _calculate_balances(df: pd.DataFrame, *, starting_balance: float = 2500.0) -> pd.Series:
    balance = starting_balance + df["amount"].cumsum()
    return (balance.round(2)).astype(float)


def generate_transactions(
    rows: int = DEFAULT_DATASET_ROWS,
    *,
    seed: int | None = DEFAULT_SEED,
    starting_balance: float = 2_500.0,
) -> pd.DataFrame:
    """Generate a deterministic ledger that satisfies Stage 1 requirements."""

    if rows <= 0:
        raise ValueError("rows must be positive")

    rng = np.random.default_rng(seed)
    start = date(2024, 1, 1)

    transactions: list[dict[str, Any]] = []
    buffer = max(500, int(rows * 0.15))

    year = start.year
    month = start.month
    while len(transactions) < rows + buffer:
        transactions.extend(_generate_month_transactions(year, month, rng))
        month += 1
        if month > 12:
            month = 1
            year += 1

    _inject_refunds(transactions, rng)

    df = pd.DataFrame(transactions)
    df.sort_values(["posted_date", "txn_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < rows:
        raise RuntimeError("Insufficient transactions generated; increase generation horizon")

    df = df.iloc[:rows].copy()
    df["balance_after"] = _calculate_balances(df, starting_balance=starting_balance)

    # Ensure deterministic rounding and dtypes.
    df["amount"] = df["amount"].round(2)
    df["balance_after"] = df["balance_after"].round(2)

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

    return df[ordered_columns]


def generate_sample_transactions(rows: int = DEFAULT_SAMPLE_ROWS, seed: int | None = DEFAULT_SEED) -> pd.DataFrame:
    """Return a smaller sample for quick visualisation or tests."""

    return generate_transactions(rows=rows, seed=seed)


def write_synthetic_csvs(
    *,
    rows: int = DEFAULT_DATASET_ROWS,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    seed: int | None = DEFAULT_SEED,
    output_dir: str | Path = Path("data"),
) -> tuple[Path, Path]:
    """Persist the main dataset and a sample CSV to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = generate_transactions(rows=rows, seed=seed)
    dataset_path = output_path / "synthetic_transactions.csv"
    dataset.to_csv(dataset_path, index=False)

    sample = dataset.head(sample_rows).copy()
    sample_path = output_path / "synthetic_transactions_sample.csv"
    sample.to_csv(sample_path, index=False)

    return dataset_path, sample_path


def main() -> None:  # pragma: no cover - convenience CLI
    dataset_path, sample_path = write_synthetic_csvs()
    print(f"Wrote {dataset_path}")
    print(f"Wrote {sample_path}")


if __name__ == "__main__":  # pragma: no cover - module CLI
    main()
