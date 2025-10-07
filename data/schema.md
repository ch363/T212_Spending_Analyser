# Spending Analyser – Transaction schema

The Stage 0 scaffold ships with synthetic transactions. Real ingestion should respect the
following column contract to guarantee compatibility with the downstream feature and insight
pipelines.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `transaction_id` | `string` | ✅ | Stable unique identifier per row. |
| `date` | `date` | ✅ | Local transaction date (timezone-naive). |
| `merchant` | `string` | ✅ | Merchant or payee label. |
| `category` | `string` | ✅ | High-level spending group (e.g. Groceries, Rent). |
| `amount` | `float` | ✅ | Monetary amount where negative values represent outflows. |
| `currency` | `string` | ➖ | ISO-4217 code. Defaults to `GBP` in the scaffold. |
| `notes` | `string` | ➖ | Free-form memo captured at source. |

Derived fields that the feature layer produces automatically:

- `month` – First-of-month timestamp computed from `date`.
- `weekday` – Human-readable day of the week.
- `is_recurring` – Boolean flag for merchants appearing >=3 times.
- `abs_amount` – Absolute value of the transaction (positive).

When integrating new data sources, ensure numeric fields are parsed as floats/ints and dates
follow ISO 8601. Any additional columns will be preserved and passed through untouched.
