"""DuckDB + Parquet persistence for the company universe (dev database).

Production uses PostgreSQL (see database/postgres.py, later sprint).
Raw documents are never stored in the relational database -- only
structured/tabular data such as company_master lives here (Section XIX).
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import polars as pl

logger = logging.getLogger("vietfin.database")

SCHEMA_SQL_PATH = Path(__file__).resolve().parents[3] / "sql" / "schema.sql"


class DuckDBStore:
    """Thin wrapper around a DuckDB file used for the dev environment."""

    def __init__(self, db_path: str | Path = "data/vietfin.duckdb") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path))
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        if not SCHEMA_SQL_PATH.exists():
            logger.warning("Schema file not found at %s", SCHEMA_SQL_PATH)
            return
        sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
        self._conn.execute(sql)
        logger.info("Schema ensured from %s", SCHEMA_SQL_PATH)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DuckDBStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # company_master
    # ------------------------------------------------------------------
    def upsert_company_master(self, df: pl.DataFrame) -> int:
        """Replace-by-primary-key upsert into company_master.

        DuckDB doesn't have a native MERGE in older versions, so this
        uses a delete-then-insert pattern scoped to the incoming
        company_ids, which is safe and idempotent for a full-refresh
        pipeline like Sprint 2's universe build.
        """
        if df.is_empty():
            return 0
        self._conn.register("incoming_company_master", df.to_arrow())
        self._conn.execute(
            "DELETE FROM company_master WHERE company_id IN "
            "(SELECT company_id FROM incoming_company_master)"
        )
        self._conn.execute(
            "INSERT INTO company_master SELECT * FROM incoming_company_master"
        )
        self._conn.unregister("incoming_company_master")
        return df.height

    def upsert_identifier_history(self, df: pl.DataFrame) -> int:
        if df.is_empty():
            return 0
        self._conn.register("incoming_identifier_history", df.to_arrow())
        self._conn.execute(
            """
            DELETE FROM company_identifier_history
            WHERE (company_id, ticker, exchange, valid_from) IN (
                SELECT company_id, ticker, exchange, valid_from
                FROM incoming_identifier_history
            )
            """
        )
        self._conn.execute(
            "INSERT INTO company_identifier_history "
            "SELECT * FROM incoming_identifier_history"
        )
        self._conn.unregister("incoming_identifier_history")
        return df.height

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def read_company_master(self) -> pl.DataFrame:
        arrow = self._conn.execute("SELECT * FROM company_master").arrow()
        return pl.from_arrow(arrow)

    def read_identifier_history(self) -> pl.DataFrame:
        arrow = self._conn.execute(
            "SELECT * FROM company_identifier_history"
        ).arrow()
        return pl.from_arrow(arrow)

    def data_quality_summary(self) -> dict:
        con = self._conn
        total = con.execute("SELECT COUNT(*) FROM company_master").fetchone()[0]
        by_exchange = con.execute(
            "SELECT exchange, COUNT(*) FROM company_master "
            "WHERE status='active' GROUP BY exchange ORDER BY 2 DESC"
        ).fetchall()
        dup_ticker_exchange = con.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT ticker, exchange FROM company_master
                WHERE status = 'active'
                GROUP BY ticker, exchange HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        missing_required = con.execute(
            "SELECT COUNT(*) FROM company_master "
            "WHERE company_name IS NULL OR exchange IS NULL "
            "OR source_name IS NULL"
        ).fetchone()[0]
        return {
            "total_companies": total,
            "active_by_exchange": dict(by_exchange),
            "duplicate_ticker_exchange_pairs": dup_ticker_exchange,
            "rows_missing_required_fields": missing_required,
        }


def write_parquet(df: pl.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("Wrote %d rows to %s", df.height, path)
