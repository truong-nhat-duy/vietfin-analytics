#!/usr/bin/env python3
"""Collect corporate profile, shareholders, officers, ratio time-series,
and price history for Vietnamese listed companies.

Usage:
    # quick smoke test
    python scripts/03_collect_corporate_info.py --tickers VCB,TCB,HPG,VNM,FPT

    # full run against the Sprint 2 universe
    python scripts/03_collect_corporate_info.py --from-universe data/gold/company_master.parquet

    # resume an interrupted full run
    python scripts/03_collect_corporate_info.py --from-universe data/gold/company_master.parquet --resume
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from vietfin.ingestion.sources.vnstock_corporate import (  # noqa: E402
    VNStockCorporateCollector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("vietfin.scripts.collect_corporate_info")

MANIFEST_PATH = Path("data/bronze/corporate_manifest.parquet")
DATASETS = ("overview", "shareholders", "officers", "ratio_summary", "price_history")


def load_tickers(args) -> list[str]:
    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.from_universe:
        df = pd.read_parquet(args.from_universe)
        col = "ticker" if "ticker" in df.columns else df.columns[0]
        return sorted(df[col].dropna().astype(str).str.upper().unique().tolist())
    raise SystemExit("Provide either --tickers or --from-universe")


def load_manifest() -> pd.DataFrame:
    if MANIFEST_PATH.exists():
        return pd.read_parquet(MANIFEST_PATH)
    return pd.DataFrame(columns=["ticker", "dataset", "rows", "bronze_path", "document_id", "document_hash", "retrieved_at", "status", "error"])


def save_manifest(df: pd.DataFrame) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MANIFEST_PATH, index=False)


def already_done(manifest: pd.DataFrame, ticker: str, dataset: str) -> bool:
    if manifest.empty:
        return False
    mask = (manifest["ticker"] == ticker) & (manifest["dataset"] == dataset) & (manifest["status"] == "ok")
    return bool(mask.any())


def append_result(manifest: pd.DataFrame, result) -> pd.DataFrame:
    row = {
        "ticker": result.ticker, "dataset": result.dataset, "rows": result.rows,
        "bronze_path": result.bronze_path, "document_id": result.document_id,
        "document_hash": result.document_hash, "retrieved_at": result.retrieved_at.isoformat(),
        "status": result.status, "error": result.error,
    }
    return pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect VN corporate profile/shareholder/price data")
    parser.add_argument("--tickers", help="Comma-separated ticker list (quick test)")
    parser.add_argument("--from-universe", help="Path to company_master.parquet from Sprint 2")
    parser.add_argument("--datasets", default=",".join(DATASETS), help=f"Comma-separated subset of {DATASETS}")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-interval", type=float, default=1.5)
    parser.add_argument("--price-history-years", type=int, default=5)
    parser.add_argument("--bronze-dir", default="data/bronze")
    args = parser.parse_args()

    tickers = load_tickers(args)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    logger.info("Starting corporate-info collection: %d tickers x %d datasets (resume=%s)",
                len(tickers), len(datasets), args.resume)

    collector = VNStockCorporateCollector(
        bronze_dir=args.bronze_dir,
        min_request_interval_seconds=args.min_interval,
        price_history_years=args.price_history_years,
    )
    fetch_fn = {
        "overview": collector.fetch_overview,
        "shareholders": collector.fetch_shareholders,
        "officers": collector.fetch_officers,
        "ratio_summary": collector.fetch_ratio_summary,
        "price_history": collector.fetch_price_history,
    }

    manifest = load_manifest() if args.resume else load_manifest().iloc[0:0]
    started_at = datetime.now(timezone.utc)
    counts = {"ok": 0, "empty": 0, "error": 0, "skipped": 0}

    for i, ticker in enumerate(tickers, start=1):
        for dataset in datasets:
            if args.resume and already_done(manifest, ticker, dataset):
                counts["skipped"] += 1
                continue
            result = fetch_fn[dataset](ticker)
            manifest = append_result(manifest, result)
            counts[result.status] = counts.get(result.status, 0) + 1

        if i % 25 == 0 or i == len(tickers):
            save_manifest(manifest)
            logger.info("Progress %d/%d tickers | %s", i, len(tickers), counts)

    save_manifest(manifest)
    duration = (datetime.now(timezone.utc) - started_at).total_seconds()
    logger.info("Done. duration_s=%.0f %s manifest=%s", duration, counts, MANIFEST_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
