# T212 Spending Analyser

Stage 0 scaffolding for a Streamlit-based personal finance assistant. The app ships with
synthetic transactions, high-level KPIs, a Plotly chart, and an optional LLM-powered summary
to demonstrate the end-to-end user experience.

## Quick start 🚀

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
streamlit run app.py
```

Open the URL that Streamlit prints to see the **Spending Analyser** home page. The scaffold uses
only in-memory synthetic data today and can be extended to ingest real transactions later.

## Features included

- 💸 Synthetic dataset seeded for deterministic demos (`src/synth.py`).
- ✏️ Feature engineering helpers adding recurring flags and date parts (`src/features.py`).
- 📊 KPI and aggregation utilities (`src/insights.py`).
- 📈 Plotly bar chart visualising month-over-month spending (`src/viz.py`).
- 🤖 LLM summary wrapper with graceful local fallback (`src/summarize.py`).
- 🧰 Shared helpers for dataframe handling and formatting (`src/utils.py`).

## Tooling & scripts

- **Run the app:** `streamlit run app.py`
- **Unit tests:** `pytest`
- **Lint:** `ruff check .`
- **Format:** `ruff format .` or `black .`
- **Pre-commit hooks:** `pre-commit install`

All commands assume an activated virtual environment.

## Configuring the OpenAI summary

Set an API key to enable hosted LLM summaries:

```bash
export OPENAI_API_KEY="sk-..."
```

Without the key the app falls back to a deterministic text summary powered by local KPIs.
You can swap the client inside `src/summarize.py` for another provider if needed.

## Project layout

```
app.py                  # Streamlit entry point
data/schema.md          # Expected transaction columns
src/                    # Core business logic modules
tests/                  # Pytest-based regression suite
.streamlit/config.toml  # Optional theme configuration
```

See `data/schema.md` for the living contract of the transaction payload.

## Data schema snapshot

| Column          | Type      | Example        | Notes                               |
|-----------------|-----------|----------------|-------------------------------------|
| `transaction_id`| `string`  | `txn-0001`     | Stable unique identifier             |
| `date`          | `date`    | `2025-06-15`   | Local transaction date               |
| `merchant`      | `string`  | `Tesco`        | Merchant / payee label               |
| `category`      | `string`  | `Groceries`    | High level spending group            |
| `amount`        | `float`   | `-42.15`       | Negative numbers represent outflows  |
| `month`         | `datetime`| `2025-06-01`   | Derived first-of-month timestamp     |
| `weekday`       | `string`  | `Sunday`       | Derived day name                     |
| `is_recurring`  | `bool`    | `True`         | Frequent merchants flagged           |
| `abs_amount`    | `float`   | `42.15`        | Absolute spend used for charting     |

## Next steps

- Replace the synthetic generator with secure bank/CSV ingestion.
- Expand KPI coverage (savings rate, cash flow trends, budgets).
- Introduce session state and basic filters for timeframe & categories.
- Plug in vector-backed LLM retrieval for personalised insights.
