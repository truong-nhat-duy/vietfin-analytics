"""Build company_master and company_identifier_history from one or more
source connectors.

Key responsibilities (Section IV/XXXIV of the spec):
  * Assign a STABLE company_id that is independent of ticker (tickers can
    change, companies can switch exchanges -- the id must not).
  * Deduplicate across sources/pages.
  * Maintain company_identifier_history so ticker/exchange changes and
    delistings are tracked instead of silently overwritten
    (avoids survivorship bias).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timezone

import polars as pl

from vietfin.ingestion.sources.base import BaseCollector, CollectedCompany

logger = logging.getLogger("vietfin.ingestion.universe")

_COMPANY_MASTER_COLUMNS = [
    "company_id",
    "ticker",
    "company_name",
    "exchange",
    "isin",
    "listing_date",
    "delisting_date",
    "status",
    "sector",
    "industry",
    "website",
    "source_name",
    "source_url",
    "retrieved_at",
    "created_at",
    "updated_at",
]

_IDENTIFIER_HISTORY_COLUMNS = [
    "company_id",
    "ticker",
    "exchange",
    "valid_from",
    "valid_to",
    "status",
    "source_name",
    "retrieved_at",
]


def make_stable_company_id(ticker: str, exchange: str, isin: str | None = None) -> str:
    """Deterministic, source-independent id.

    Uses ISIN when available (most stable real-world identifier); falls
    back to a hash of (ticker, exchange) at first-sight. This is a
    reasonable v1 -- if a company later changes ticker, the identifier
    history mechanism (not company_id) is what records the change, so
    company_id assigned at first ingestion is preserved by callers
    re-resolving on isin-or-(original ticker, exchange) rather than the
    live ticker. See `resolve_existing_company_id`.
    """
    basis = isin.strip().upper() if isin else f"{ticker.strip().upper()}::{exchange.strip().upper()}"
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
    return f"CID_{digest}"


def collect_all(
    collectors: list[BaseCollector], exchange: str | None = None
) -> list[CollectedCompany]:
    """Run every collector and merge results, tolerating individual
    collector failures (a source going down shouldn't take the whole
    pipeline down -- log and continue with the others)."""
    records: list[CollectedCompany] = []
    for collector in collectors:
        try:
            batch = list(collector.fetch_companies(exchange=exchange))
            logger.info(
                "collector=%s exchange=%s records=%d",
                collector.source_name,
                exchange,
                len(batch),
            )
            records.extend(batch)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Collector %s failed; continuing with remaining collectors",
                getattr(collector, "source_name", collector.__class__.__name__),
            )
    return records


def deduplicate(records: list[CollectedCompany]) -> list[CollectedCompany]:
    """Deduplicate by (ticker, exchange), preferring the most recently
    retrieved record when the same company appears from multiple
    sources/pages."""
    best: dict[tuple[str, str], CollectedCompany] = {}
    for rec in records:
        key = (rec.ticker, rec.exchange)
        existing = best.get(key)
        if existing is None or rec.retrieved_at >= existing.retrieved_at:
            best[key] = rec
    deduped = list(best.values())
    logger.info("Deduplicated %d records -> %d unique (ticker, exchange)", len(records), len(deduped))
    return deduped


def build_company_master(records: list[CollectedCompany]) -> pl.DataFrame:
    now = datetime.now(timezone.utc)
    rows = []
    for rec in records:
        company_id = make_stable_company_id(rec.ticker, rec.exchange, rec.isin)
        rows.append(
            {
                "company_id": company_id,
                "ticker": rec.ticker,
                "company_name": rec.company_name,
                "exchange": rec.exchange,
                "isin": rec.isin,
                "listing_date": rec.listing_date,
                "delisting_date": rec.delisting_date,
                "status": rec.status,
                "sector": rec.sector,
                "industry": rec.industry,
                "website": rec.website,
                "source_name": rec.source_name,
                "source_url": rec.source_url,
                "retrieved_at": rec.retrieved_at,
                "created_at": now,
                "updated_at": now,
            }
        )
    if not rows:
        return pl.DataFrame(schema={c: pl.Utf8 for c in _COMPANY_MASTER_COLUMNS})
    return pl.DataFrame(rows, schema_overrides={"retrieved_at": pl.Datetime, "created_at": pl.Datetime, "updated_at": pl.Datetime})


def build_identifier_history(
    records: list[CollectedCompany], as_of: date | None = None
) -> pl.DataFrame:
    """One open-ended (valid_to = NULL) history row per currently-observed
    (company_id, ticker, exchange). Sprint 2 seeds the history; later
    sprints close out old rows (`valid_to`) when a ticker/exchange change
    is detected on a subsequent run, rather than deleting them."""
    as_of = as_of or datetime.now(timezone.utc).date()
    rows = []
    for rec in records:
        company_id = make_stable_company_id(rec.ticker, rec.exchange, rec.isin)
        rows.append(
            {
                "company_id": company_id,
                "ticker": rec.ticker,
                "exchange": rec.exchange,
                "valid_from": rec.listing_date or as_of,
                "valid_to": rec.delisting_date,
                "status": rec.status,
                "source_name": rec.source_name,
                "retrieved_at": rec.retrieved_at,
            }
        )
    if not rows:
        return pl.DataFrame(schema={c: pl.Utf8 for c in _IDENTIFIER_HISTORY_COLUMNS})
    return pl.DataFrame(rows, schema_overrides={"retrieved_at": pl.Datetime})


def data_quality_report(df: pl.DataFrame) -> dict:
    """Lightweight in-memory quality report (mirrors DuckDBStore.data_quality_summary,
    usable before anything is persisted)."""
    if df.is_empty():
        return {
            "total_companies": 0,
            "active_by_exchange": {},
            "duplicate_ticker_exchange_pairs": 0,
            "rows_missing_required_fields": 0,
        }
    active = df.filter(pl.col("status") == "active")
    by_exchange = (
        active.group_by("exchange").len().sort("len", descending=True).to_dicts()
    )
    dup = (
        active.group_by(["ticker", "exchange"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    missing = df.filter(
        pl.col("company_name").is_null()
        | pl.col("exchange").is_null()
        | pl.col("source_name").is_null()
    ).height
    return {
        "total_companies": df.height,
        "active_by_exchange": {r["exchange"]: r["len"] for r in by_exchange},
        "duplicate_ticker_exchange_pairs": dup,
        "rows_missing_required_fields": missing,
    }
