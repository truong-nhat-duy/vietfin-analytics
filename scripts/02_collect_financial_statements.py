#!/usr/bin/env python3
"""Collect 10 years of financial statements (balance sheet, income
statement, cash flow, ratios; annual + quarterly) for Vietnamese listed
companies.

Reads tickers from company_master.parquet (built in Sprint 2), or from an
explicit --tickers list for a quick test run. Writes raw statements to
data/bronze/financial_statements/... and a run manifest (document_master
-style log) to data/bronze/financial_statements_manifest.parquet.

Resumable: re-running the script skips (ticker, statement_type, period)
combinations already marked "ok" in the manifest from a prior run, unless
--force is passed. This matters because a full-universe run (~1,700
tickers x 4 statement types x 2 periods = ~13,600 calls at ~1.5s/call) can
take several hours and may need to be resumed after an interruption.

Usage:
    # quick smoke test on 5 tickers
    python scripts/02_collect_financial_statements.py --tickers VCB,TCB,HPG,VNM,FPT

    # full run against the Sprint 2 universe
    python scripts/02_collect_financial_statements.py --from-universe data/gold/company_master.parquet

    # resume an interrupted full run
    python scripts/02_collect_financial_statements.py --from-universe data/gold/company_master.parquet --resume
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from vietfin.ingestion.sources.vnstock_financials import (  # noqa: E402
    PERIODS,
    STATEMENT_TYPES,
    FetchResult,
    VNStockFinancialsCollector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("vietfin.scripts.collect_financial_statements")

MANIFEST_PATH = Path("data/bronze/financial_statements_manifest.parquet")


def load_tickers(args) -> list[str]:
    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.from_universe:
        df = pd.read_parquet(args.from_universe)
        col = "ticker" if "ticker" in df.columns else df.columns[0]
        tickers = sorted(df[col].dropna().astype(str).str.upper().unique().tolist())
        return tickers
    raise SystemExit("Provide either --tickers or --from-universe")


def load_manifest() -> pd.DataFrame:
    if MANIFEST_PATH.exists():
        return pd.read_parquet(MANIFEST_PATH)
    return pd.DataFrame(
        columns=[
            "ticker", "statement_type", "period", "rows", "bronze_path",
            "document_id", "document_hash", "retrieved_at", "status", "error",
        ]
    )


def save_manifest(df: pd.DataFrame) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MANIFEST_PATH, index=False)


def already_done(manifest: pd.DataFrame, ticker: str, statement_type: str, period: str) -> bool:
    if manifest.empty:
        return False
    mask = (
        (manifest["ticker"] == ticker)
        & (manifest["statement_type"] == statement_type)
        & (manifest["period"] == period)
        & (manifest["status"] == "ok")
    )
    return bool(mask.any())


def append_result(manifest: pd.DataFrame, result: FetchResult) -> pd.DataFrame:
    row = {
        "ticker": result.ticker,
        "statement_type": result.statement_type,
        "period": result.period,
        "rows": result.rows,
        "bronze_path": result.bronze_path,
        "document_id": result.document_id,
        "document_hash": result.document_hash,
        "retrieved_at": result.retrieved_at.isoformat(),
        "status": result.status,
        "error": result.error,
    }
    return pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect VN financial statements (10y lookback)")
    parser.add_argument("--tickers", help="Comma-separated ticker list (quick test)")
    parser.add_argument("--from-universe", help="Path to company_master.parquet from Sprint 2")
    parser.add_argument("--years", type=int, default=10, help="Lookback window in years")
    parser.add_argument(
        "--statement-types",
        default=",".join(STATEMENT_TYPES),
        help=f"Comma-separated subset of {STATEMENT_TYPES}",
    )
    parser.add_argument(
        "--periods", default=",".join(PERIODS), help=f"Comma-separated subset of {PERIODS}"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip combinations already marked ok in the manifest"
    )
    parser.add_argument(
        "--min-interval", type=float, default=1.5, help="Min seconds between requests (politeness)"
    )
    parser.add_argument("--bronze-dir", default="data/bronze")
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "vnstock API key (from https://vnstocks.com/login) to authenticate "
            "as Community/Sponsor tier. Without this, vnstock runs in Guest "
            "mode and caps you at 4 historical periods per statement -- not "
            "enough for a 10-year lookback. See vnstock_financials.py docstring."
        ),
    )
    parser.add_argument(
        "--source-backend", 
        default="KBS", 
        choices=["KBS", "VCI"], # Chỉ định rõ chỉ chấp nhận 2 nguồn này, loại bỏ TCBS
        help="vnstock data provider: KBS or VCI" 
    )
    args = parser.parse_args()

    tickers = load_tickers(args)
    statement_types = [s.strip() for s in args.statement_types.split(",") if s.strip()]
    periods = [p.strip() for p in args.periods.split(",") if p.strip()]

    logger.info(
        "Starting collection: %d tickers x %d statement types x %d periods (resume=%s)",
        len(tickers), len(statement_types), len(periods), args.resume,
    )

    if not args.api_key:
        logger.warning(
            "No --api-key provided. vnstock Guest mode caps you at 4 historical "
            "periods per statement -- a 10-year lookback needs 10 (annual) or "
            "40 (quarterly). Register a free account at https://vnstocks.com/login "
            "for 8 periods, or a Sponsor account for full history."
        )

    collector = VNStockFinancialsCollector(
        bronze_dir=args.bronze_dir,
        min_request_interval_seconds=args.min_interval,
        lookback_years=args.years,
        source_backend=args.source_backend,
        api_key=args.api_key,
    )

    manifest = load_manifest() if args.resume else load_manifest().iloc[0:0]
    started_at = datetime.now(timezone.utc)
    n_ok = n_empty = n_error = n_skipped = 0

    for i, ticker in enumerate(tickers, start=1):
        for statement_type in statement_types:
            for period in periods:
                if args.resume and already_done(manifest, ticker, statement_type, period):
                    n_skipped += 1
                    continue
                result = collector.fetch_one(ticker, statement_type, period)  # type: ignore[arg-type]
                manifest = append_result(manifest, result)
                if result.status == "ok":
                    n_ok += 1
                elif result.status == "empty":
                    n_empty += 1
                else:
                    n_error += 1

        if i % 25 == 0 or i == len(tickers):
            save_manifest(manifest)
            logger.info(
                "Progress %d/%d tickers | ok=%d empty=%d error=%d skipped=%d",
                i, len(tickers), n_ok, n_empty, n_error, n_skipped,
            )

    save_manifest(manifest)
    duration = (datetime.now(timezone.utc) - started_at).total_seconds()
    logger.info(
        "Done. duration_s=%.0f ok=%d empty=%d error=%d skipped=%d manifest=%s",
        duration, n_ok, n_empty, n_error, n_skipped, MANIFEST_PATH,
    )

    if n_error > 0:
        logger.warning(
            "%d fetches failed. Re-run with --resume to retry only the missing "
            "combinations once the underlying issue is fixed.", n_error,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())