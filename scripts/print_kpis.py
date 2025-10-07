"""Utility script to print Stage 2 KPI payload for a sample dataset."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from src import features, insights, synth


def _default_serializer(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def main() -> None:
    transactions = synth.generate_sample_transactions(rows=500, seed=synth.DEFAULT_SEED)
    enriched = features.add_engineered_features(transactions)
    payload = insights.calculate_kpis(enriched)
    print(json.dumps(payload, indent=2, default=_default_serializer))


if __name__ == "__main__":
    main()
