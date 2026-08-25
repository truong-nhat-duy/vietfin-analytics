"""Shared utility: make a DataFrame safe to write to Parquet.

VERIFIED NECESSARY live on 2026-08-24 / 2026-08-25: vnstock responses can
contain object columns that mix scalar values with dicts/lists (e.g.
Company.overview()'s 'prev_insight' column holds a dict like
{'targetPrice': 42600.0, 'rating': 'BUY', ...}), or object columns that
are mostly blank/numeric-looking with one genuine string value mixed in
(e.g. a 'ratio' statement's 'item_en' column). Both shapes crash
pyarrow's type inference (ArrowInvalid) and, uncaught, can kill a
multi-hour/multi-thousand-ticker batch job over a single odd ticker.

This is intentionally lossy-safe rather than schema-perfect: the BRONZE
layer's job is raw fidelity + never crashing, not typed correctness --
type coercion belongs in Silver normalization (Sprint 4), which can
parse these stringified values back out (including with
`ast.literal_eval` for the dict-repr case) with full context about the
expected target schema.
"""

from __future__ import annotations

import pandas as pd


def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with every object-dtype column stringified
    (NaN preserved as null), so pyarrow can never mis-infer a type or
    choke on an unsupported nested value (dict/list) again."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].apply(lambda x: x if pd.isna(x) else str(x))
    return out


def write_parquet_safely(df: pd.DataFrame, out_path) -> None:
    """Write df to out_path, sanitizing first; falls back to a full-frame
    string coercion + retry if something still slips through, so one bad
    ticker/record can never crash a large batch job."""
    sanitized = sanitize_for_parquet(df)
    try:
        sanitized.to_parquet(out_path, index=False)
    except Exception:  # noqa: BLE001
        sanitized.astype(str).to_parquet(out_path, index=False)
