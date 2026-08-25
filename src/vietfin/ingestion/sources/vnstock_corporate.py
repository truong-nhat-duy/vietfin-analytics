"""Collector for corporate profile, shareholders, officers, financial
ratio time-series, and historical price data.

STATUS: working, non-commercial/research use.

Every method here is built ONLY from calls verified live against the
real vnstock 4.0.7 installation on 2026-08-25 (see diagnose_company_info.py
output) -- unlike earlier drafts in this project, nothing here is a
guess from documentation alone. Confirmed live:

    from vnstock import Company, Quote

    Company(source='VCI', symbol=ticker).overview()          # market/sector data, 37 cols
    Company(source='KBS', symbol=ticker).overview()          # tax_id, address, CEO, phone/email
    Company(source='VCI', symbol=ticker).shareholders(mode='detailed')
    Company(source='VCI', symbol=ticker).officers(filter_by='working')
    Company(source='KBS', symbol=ticker).officers()
    Company(source='VCI', symbol=ticker).ratio_summary()     # TIME SERIES, ~40 periods, 61 cols
    Quote(symbol=ticker, source='VCI').history(start=..., end=..., interval='1D')

CONFIRMED NOT TO WORK / NOT NEEDED:
  - `vnstock_data` is not a separate installable package; `Company` and
    `Quote` live in the main `vnstock` package.
  - `stock_historical_data()` (the function used in an earlier draft
    pasted into this project) no longer exists -- ImportError, confirmed
    live. Use `Quote(...).history(...)` instead.

TAX ID CORRECTION: an earlier message in this project incorrectly stated
vnstock has no tax-ID field anywhere. It does -- `tax_id` is present in
Company(source='KBS').overview(), confirmed live (e.g. TCB ->
'0100230800'). This collector sources tax_id from KBS accordingly.

LICENSE NOTE: same as vnstock_financials.py -- personal/non-commercial
use only unless a commercial license is obtained from the vnstock author.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from tenacity import retry, stop_after_attempt, wait_exponential

from vietfin.ingestion.sources.base import RateLimiter, SourceAccessError
from vietfin.ingestion.sources.parquet_safety import write_parquet_safely

logger = logging.getLogger("vietfin.ingestion.vnstock_corporate")

Dataset = Literal["overview", "shareholders", "officers", "ratio_summary", "price_history"]

_SOURCE_NAME = "vnstock_corporate"
_SOURCE_URL = "https://github.com/thinh-vu/vnstock"


@dataclass(frozen=True)
class FetchResult:
    ticker: str
    dataset: Dataset
    rows: int
    bronze_path: str | None
    document_id: str
    document_hash: str | None
    retrieved_at: datetime
    status: str  # ok | empty | error
    error: str | None = None


class VNStockCorporateCollector:
    """Fetches corporate profile, shareholders, officers, ratio
    time-series, and price history for a ticker, writing each to the
    bronze data lake layer with provenance."""

    source_name = _SOURCE_NAME

    def __init__(
        self,
        bronze_dir: str | Path = "data/bronze",
        min_request_interval_seconds: float = 1.5,
        price_history_years: int = 5,
    ) -> None:
        self.bronze_dir = Path(bronze_dir)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)
        self._rate_limiter = RateLimiter(min_request_interval_seconds)
        self.price_history_years = price_history_years
        self.logger = logging.getLogger(f"vietfin.ingestion.{self.source_name}")

    def _get_company(self, ticker: str, source: str):
        try:
            from vnstock import Company
        except ImportError as exc:  # pragma: no cover
            raise SourceAccessError(
                "vnstock is not installed. Run `pip install vnstock`."
            ) from exc
        return Company(source=source, symbol=ticker)

    def _get_quote(self, ticker: str, source: str = "VCI"):
        try:
            from vnstock import Quote
        except ImportError as exc:  # pragma: no cover
            raise SourceAccessError(
                "vnstock is not installed. Run `pip install vnstock`."
            ) from exc
        return Quote(symbol=ticker, source=source)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20), reraise=True)
    def _call(self, fn, *args, **kwargs):
        self._rate_limiter.wait()
        return fn(*args, **kwargs)

    # ------------------------------------------------------------------
    def fetch_overview(self, ticker: str) -> FetchResult:
        """Merges VCI overview (market/sector) with KBS overview (tax_id,
        address, CEO, contact info) into one bronze record per ticker."""
        retrieved_at = datetime.now(timezone.utc)
        document_id = f"{ticker}_overview_{retrieved_at:%Y%m%d%H%M%S}"
        try:
            import pandas as pd

            vci_df = self._call(self._get_company(ticker, "VCI").overview)
            kbs_df = self._call(self._get_company(ticker, "KBS").overview)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to fetch overview for %s: %s", ticker, exc)
            return FetchResult(
                ticker=ticker, dataset="overview", rows=0, bronze_path=None,
                document_id=document_id, document_hash=None,
                retrieved_at=retrieved_at, status="error", error=str(exc),
            )

        if (vci_df is None or len(vci_df) == 0) and (kbs_df is None or len(kbs_df) == 0):
            return FetchResult(
                ticker=ticker, dataset="overview", rows=0, bronze_path=None,
                document_id=document_id, document_hash=None,
                retrieved_at=retrieved_at, status="empty",
            )

        # Suffix overlapping columns so nothing silently overwrites; the
        # Silver stage decides which source wins per field (e.g. prefer
        # KBS's tax_id/address, VCI's sector/market data).
        vci_df = vci_df.add_suffix("_vci") if vci_df is not None and len(vci_df) else vci_df
        kbs_df = kbs_df.add_suffix("_kbs") if kbs_df is not None and len(kbs_df) else kbs_df
        if vci_df is not None and kbs_df is not None and len(vci_df) and len(kbs_df):
            merged = pd.concat(
                [vci_df.reset_index(drop=True), kbs_df.reset_index(drop=True)], axis=1
            )
        else:
            merged = vci_df if vci_df is not None and len(vci_df) else kbs_df

        return self._write_and_result(merged, ticker, "overview", retrieved_at, document_id)

    def fetch_shareholders(self, ticker: str) -> FetchResult:
        retrieved_at = datetime.now(timezone.utc)
        document_id = f"{ticker}_shareholders_{retrieved_at:%Y%m%d%H%M%S}"
        try:
            df = self._call(self._get_company(ticker, "VCI").shareholders, mode="detailed")
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to fetch shareholders for %s: %s", ticker, exc)
            return FetchResult(
                ticker=ticker, dataset="shareholders", rows=0, bronze_path=None,
                document_id=document_id, document_hash=None,
                retrieved_at=retrieved_at, status="error", error=str(exc),
            )
        if df is None or len(df) == 0:
            return FetchResult(
                ticker=ticker, dataset="shareholders", rows=0, bronze_path=None,
                document_id=document_id, document_hash=None,
                retrieved_at=retrieved_at, status="empty",
            )
        return self._write_and_result(df, ticker, "shareholders", retrieved_at, document_id)

    def fetch_officers(self, ticker: str) -> FetchResult:
        """Merges VCI officers (has ownership %) with KBS officers (has
        Vietnamese position titles + English translation) via a simple
        concat with a source tag -- schemas differ enough (see module
        docstring) that a real join isn't safe without name-matching
        logic Silver normalization should own, not this raw layer."""
        retrieved_at = datetime.now(timezone.utc)
        document_id = f"{ticker}_officers_{retrieved_at:%Y%m%d%H%M%S}"
        import pandas as pd

        frames = []
        errors = []
        for source in ("VCI", "KBS"):
            try:
                if source == "VCI":
                    df = self._call(self._get_company(ticker, "VCI").officers, filter_by="working")
                else:
                    df = self._call(self._get_company(ticker, "KBS").officers)
                if df is not None and len(df) > 0:
                    df = df.copy()
                    df["_officer_source"] = source
                    frames.append(df)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{source}: {exc}")
                self.logger.warning("Failed to fetch %s officers for %s: %s", source, ticker, exc)

        if not frames:
            status = "error" if errors else "empty"
            return FetchResult(
                ticker=ticker, dataset="officers", rows=0, bronze_path=None,
                document_id=document_id, document_hash=None, retrieved_at=retrieved_at,
                status=status, error="; ".join(errors) if errors else None,
            )
        merged = pd.concat(frames, ignore_index=True, sort=False)
        return self._write_and_result(merged, ticker, "officers", retrieved_at, document_id)

    def fetch_ratio_summary(self, ticker: str) -> FetchResult:
        """VCI's ratio_summary() is a TIME SERIES (confirmed live: ~40
        periods for TCB, both RATIO_TTM and RATIO_YEAR rows), not a
        single snapshot -- a very useful head start for Sprint
        5's financial_fact/ratio work, though it should still be
        cross-checked against ratios computed from the raw statements
        rather than trusted blindly."""
        retrieved_at = datetime.now(timezone.utc)
        document_id = f"{ticker}_ratio_summary_{retrieved_at:%Y%m%d%H%M%S}"
        try:
            df = self._call(self._get_company(ticker, "VCI").ratio_summary)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to fetch ratio_summary for %s: %s", ticker, exc)
            return FetchResult(
                ticker=ticker, dataset="ratio_summary", rows=0, bronze_path=None,
                document_id=document_id, document_hash=None,
                retrieved_at=retrieved_at, status="error", error=str(exc),
            )
        if df is None or len(df) == 0:
            return FetchResult(
                ticker=ticker, dataset="ratio_summary", rows=0, bronze_path=None,
                document_id=document_id, document_hash=None,
                retrieved_at=retrieved_at, status="empty",
            )
        return self._write_and_result(df, ticker, "ratio_summary", retrieved_at, document_id)

    def fetch_price_history(self, ticker: str) -> FetchResult:
        retrieved_at = datetime.now(timezone.utc)
        document_id = f"{ticker}_price_history_{retrieved_at:%Y%m%d%H%M%S}"
        end = date.today()
        start = end - timedelta(days=365 * self.price_history_years)
        try:
            quote = self._get_quote(ticker, "VCI")
            df = self._call(
                quote.history,
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1D",
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to fetch price history for %s: %s", ticker, exc)
            return FetchResult(
                ticker=ticker, dataset="price_history", rows=0, bronze_path=None,
                document_id=document_id, document_hash=None,
                retrieved_at=retrieved_at, status="error", error=str(exc),
            )
        if df is None or len(df) == 0:
            return FetchResult(
                ticker=ticker, dataset="price_history", rows=0, bronze_path=None,
                document_id=document_id, document_hash=None,
                retrieved_at=retrieved_at, status="empty",
            )
        return self._write_and_result(df, ticker, "price_history", retrieved_at, document_id)

    def fetch_all_for_ticker(self, ticker: str) -> list[FetchResult]:
        return [
            self.fetch_overview(ticker),
            self.fetch_shareholders(ticker),
            self.fetch_officers(ticker),
            self.fetch_ratio_summary(ticker),
            self.fetch_price_history(ticker),
        ]

    # ------------------------------------------------------------------
    def _write_and_result(self, df, ticker, dataset: Dataset, retrieved_at, document_id) -> FetchResult:
        enriched = df.copy()
        enriched["ticker"] = ticker
        enriched["dataset"] = dataset
        enriched["source_name"] = _SOURCE_NAME
        enriched["source_url"] = _SOURCE_URL
        enriched["retrieved_at"] = retrieved_at.isoformat()
        enriched["document_id"] = document_id

        out_dir = self.bronze_dir / "corporate" / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ticker}_{retrieved_at:%Y%m%d}.parquet"

        try:
            write_parquet_safely(enriched, out_path)
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Bronze write permanently failed for %s/%s: %s", ticker, dataset, exc)
            return FetchResult(
                ticker=ticker, dataset=dataset, rows=0, bronze_path=None,
                document_id=document_id, document_hash=None,
                retrieved_at=retrieved_at, status="error", error=str(exc),
            )

        document_hash = hashlib.sha256(
            df.to_json(orient="records", date_format="iso").encode("utf-8")
        ).hexdigest()
        return FetchResult(
            ticker=ticker, dataset=dataset, rows=len(df), bronze_path=str(out_path),
            document_id=document_id, document_hash=document_hash,
            retrieved_at=retrieved_at, status="ok",
        )
