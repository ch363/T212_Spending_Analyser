# Spending Analyser – Stage 1 transaction schema

The Stage 1 generator produces deterministic debit card ledgers that match the column
contract below. Amounts are expressed in account currency (GBP) with negative values for
debits and positives for credits/refunds.

| Column | Type | Required | Description |
| --- | --- | --- | --- |
| `txn_id` | `uuid` | ✅ | UUID v4 assigned per row. Stable for a given seed. |
| `posted_date` | `date` | ✅ | Ledger posting date. Weekends allowed. |
| `value_date` | `date` | ✅ | Banking value date. Rolls forward to Monday on weekends and may include 0–2 day delays. |
| `merchant_name` | `string` | ✅ | Merchant/payee label from the catalogue. |
| `mcc` | `string` | ✅ | Merchant Category Code aligned with Visa/Mastercard taxonomy. |
| `category` | `string` | ✅ | High-level spend group (Housing, Groceries, Travel, etc.). |
| `subcategory` | `string` | ✅ | Granular subcategory (Rent, Supermarket, Flights, ...). |
| `amount` | `float` | ✅ | Account currency amount. Debits are negative, credits positive. |
| `currency` | `string` | ✅ | Original transaction currency (`GBP`, `EUR`, `USD`). Most rows are `GBP`. |
| `balance_after` | `float` | ✅ | Running balance after the transaction, seeded from £2,500. |
| `channel` | `string` | ✅ | Acquisition channel (`POS`, `E-Com`, `ATM`, `Direct Debit`, `Credit`). |
| `location_city` | `string` | ✅ | Merchant city. |
| `location_country` | `string` | ✅ | ISO alpha-2 country. |
| `card_last4` | `string` | ✅ | Masked card identifier sampled from a small pool. |
| `user_id` | `string` | ✅ | Customer identifier (`user_001` / `user_002`). |
| `recurring_id` | `string` | ➖ | Deterministic recurring handle for subscriptions/bills, else `NA`. |
| `notes` | `string` | ➖ | Free-form annotations (refund references, FX notes). Empty when not applicable. |

### Generation assumptions

- **Seeded RNG:** `numpy.random.default_rng(seed)` ensures identical output when repeating a run with the same seed.
- **Pay-cycle modelling:** Salary from `T212 Capital` lands on the 25th (shifted to the previous business day). Rent and utilities fall on fixed offsets each month.
- **Seasonality:** Groceries, transport, and dining use Poisson-distributed daily counts with weekend uplifts. Travel spikes occur in ~30% of months.
- **Edge cases:** The dataset includes refunds, ATM withdrawals, foreign currency purchases (with FX note), and positive inflows (salary/tax refunds).
- **Balances:** Running balance starts at £2,500 and incorporates each amount sequentially after sorting by `posted_date` and transaction UUID.

Downstream feature engineering (`src/features.py`) augments the ledger with:

- `month`, `weekday`, `day`
- `is_recurring`, `is_credit`
- `abs_amount`
- `monthly_net_amount`
- `category_month_share`

When ingesting real data, align with the schema above to guarantee compatibility with the
feature, insight, and visualisation layers.
