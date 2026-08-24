#!/usr/bin/env python3
"""Sprint 2 CLI: build company_master + company_identifier_history.

Usage:
    python scripts/01_build_universe.py --source vnstock --out data/gold
    python scripts/01_build_universe.py --source vnstock --exchange HOSE
    python scripts/01_build_universe.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vietfin.database.duckdb import DuckDBStore, write_parquet  # noqa: E402
from vietfin.ingestion.sources.base import SourceAccessError  # noqa: E402
from vietfin.ingestion.universe import (  # noqa: E402
    build_company_master,
    build_identifier_history,
    collect_all,
    data_quality_report,
    deduplicate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("vietfin.scripts.build_universe")

SOURCE_CHOICES = ("vnstock", "hose", "hnx", "upcom")


def get_collector(name: str):
    if name == "vnstock":
        from vietfin.ingestion.sources.vnstock_source import VNStockUniverseCollector

        return VNStockUniverseCollector()
    if name == "hose":
        from vietfin.ingestion.sources.hose import HOSECollector

        return HOSECollector()
    if name == "hnx":
        from vietfin.ingestion.sources.hnx import HNXCollector

        return HNXCollector()
    if name == "upcom":
        from vietfin.ingestion.sources.upcom import UPCOMCollector

        return UPCOMCollector()
    raise ValueError(f"Unknown source: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the VIETFIN company universe")
    parser.add_argument(
        "--source",
        action="append",
        choices=SOURCE_CHOICES,
        default=None,
        help="Source connector(s) to use. Repeatable. Default: vnstock",
    )
    parser.add_argument(
        "--exchange",
        choices=["HOSE", "HNX", "UPCOM"],
        default=None,
        help="Restrict to a single exchange (default: all)",
    )
    parser.add_argument(
        "--out", default="data/gold", help="Output directory for Parquet files"
    )
    parser.add_argument(
        "--duckdb", default="data/vietfin.duckdb", help="Path to DuckDB file"
    )
    args = parser.parse_args()

    sources = args.source or ["vnstock"]
    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    logger.info("run_id=%s sources=%s exchange=%s", run_id, sources, args.exchange)

    collectors = []
    for name in sources:
        try:
            collectors.append(get_collector(name))
        except Exception:  # noqa: BLE001
            logger.exception("Could not initialize collector %s", name)

    if not collectors:
        logger.error("No collectors available, aborting.")
        return 1

    records = collect_all(collectors, exchange=args.exchange)
    if not records:
        logger.error(
            "No records collected from any source. Check network access, "
            "vnstock installation, or try a different --source."
        )
        return 1

    deduped = deduplicate(records)
    company_master_df = build_company_master(deduped)
    identifier_history_df = build_identifier_history(deduped)

    report = data_quality_report(company_master_df)
    logger.info("Data quality report: %s", report)

    out_dir = Path(args.out)
    write_parquet(company_master_df, out_dir / "company_master.parquet")
    write_parquet(identifier_history_df, out_dir / "company_identifier_history.parquet")

    with DuckDBStore(args.duckdb) as store:
        n1 = store.upsert_company_master(company_master_df)
        n2 = store.upsert_identifier_history(identifier_history_df)
        logger.info("Upserted %d company_master rows, %d identifier_history rows", n1, n2)
        summary = store.data_quality_summary()
        logger.info("DuckDB data quality summary: %s", summary)

    finished_at = datetime.now(timezone.utc)
    logger.info(
        "Sprint 2 run complete. run_id=%s duration_s=%.1f total_companies=%d",
        run_id,
        (finished_at - started_at).total_seconds(),
        report["total_companies"],
    )

    if report["duplicate_ticker_exchange_pairs"] > 0:
        logger.warning(
            "%d duplicate (ticker, exchange) pairs found among active rows -- "
            "investigate before proceeding to Sprint 3.",
            report["duplicate_ticker_exchange_pairs"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
