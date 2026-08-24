"""Collector for Vietnamese company financial statements (10-year lookback).

STATUS: working, non-commercial/research use.

Uses vnstock's `Finance` adapter, which wraps public financial-statement
endpoints from KBS/VCI/TCBS (the same data providers used for the Sprint
2 company universe). No authentication bypass, no CAPTCHA handling, no
scraping of the exchanges' own HTML -- this only calls the documented
JSON endpoints vnstock already wraps.

VERIFIED against the vnstock README as of 2026-08-21 (v4.0.2). Two
things changed from earlier assumptions and matter a lot here:

1. Constructor/method signature. Confirmed current usage is:

       from vnstock import Finance
       finance = Finance(symbol=ticker, source='KBS')   # no `period` here
       finance.balance_sheet(period='year')             # period goes here
       finance.income_statement(period='year')
       finance.cash_flow(period='year')
       finance.ratio(period='year', lang='vi')

   (earlier code in this project incorrectly passed `period` and
   `get_all`/`show_log` to the constructor -- fixed below.)

2. *** ACCOUNT-TIER PERIOD LIMIT (important, affects the 10-year goal) ***
   vnstock now gates how many historical reporting periods you can pull
   based on account tier:
     - Guest (no signup):        up to  4 periods
     - Community (free signup):  up to  8 periods
     - Sponsor:                  full history
   A 10-year lookback needs 10 annual periods or 40 quarterly periods --
   Guest and Community tiers are NOT sufficient for the full window this
   collector is configured for. This collector does not and cannot work
   around that limit (doing so would mean bypassing an access control,
   which the project's data policy forbids). Use `register_user()` /
   the `api_key` parameter below to authenticate at whatever tier you
   are entitled to, and expect fewer periods than `lookback_years`
   requests if you are on Guest/Community. `FetchResult` reports the
   actual row count returned so downstream code can detect this.

LICENSE NOTE: vnstock is released under a personal / non-commercial
license. Using it as a data source inside a commercial product (Section
XXII of the VIETFIN spec: FREE/PRO/ENTERPRISE tiers) requires contacting
the vnstock author for a commercial license first. This collector is
safe to use for internal research / the golden-sample validation in
Section XXXII; do not wire it into a paid tier without that license.

Every statement pulled is persisted to the BRONZE layer as-is (raw
account labels, raw values, source page) with full provenance, per
Section II / VI of the spec. Normalization into canonical_account /
financial_fact happens in a later step (Sprint 5), not here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from vietfin.ingestion.sources.base import RateLimiter, SourceAccessError

logger = logging.getLogger("vietfin.ingestion.financials")

StatementType = Literal["balance_sheet", "income_statement", "cash_flow", "ratio"]
Period = Literal["year", "quarter"]

STATEMENT_TYPES: tuple[StatementType, ...] = (
    "balance_sheet",
    "income_statement",
    "cash_flow",
    "ratio",
)
PERIODS: tuple[Period, ...] = ("year", "quarter")

_SOURCE_NAME = "vnstock_finance"
_SOURCE_URL = "https://github.com/thinh-vu/vnstock"


@dataclass(frozen=True)
class FetchResult:
    """Outcome of one (ticker, statement_type, period) fetch."""

    ticker: str
    statement_type: StatementType
    period: Period
    rows: int
    bronze_path: str | None
    document_id: str
    document_hash: str | None
    retrieved_at: datetime
    status: str  # ok | empty | error
    error: str | None = None


class RetryableFetchError(RuntimeError):
    """Raised so tenacity retries only transient failures, not bad input."""


class VNStockFinancialsCollector:
    """Fetches balance sheet / income statement / cash flow / ratios for a
    ticker, for both annual and quarterly periods, and writes each result
    to the bronze data lake layer with provenance.
    """

    source_name = _SOURCE_NAME

    def __init__(
        self,
        bronze_dir: str | Path = "data/bronze",
        min_request_interval_seconds: float = 1.5,
        max_retries: int = 3,
        source_backend: str = "KBS",  # "KBS", "VCI", or 
        lookback_years: int = 10,
        api_key: str | None = None,
    ) -> None:
        self.bronze_dir = Path(bronze_dir)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)
        self._rate_limiter = RateLimiter(min_request_interval_seconds)
        self.max_retries = max_retries
        self.source_backend = source_backend
        self.lookback_years = lookback_years
        self.api_key = api_key
        self._registered = False
        self.logger = logging.getLogger(f"vietfin.ingestion.{self.source_name}")

    # ------------------------------------------------------------------
    def _ensure_registered(self) -> None:
        """Authenticate once per collector instance if an api_key was
        provided. Without this, vnstock runs in Guest mode (max 4
        periods per statement) -- see module docstring."""
        if self._registered or not self.api_key:
            return
        try:
            from vnstock import register_user

            register_user(api_key=self.api_key)
            self._registered = True
            self.logger.info("vnstock registered with provided API key")
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "vnstock register_user() failed, continuing in Guest mode: %s", exc
            )

    def _get_finance_client(self, ticker: str, period: Period, backend: str | None = None):
        try:
            from vnstock import Finance
        except ImportError as exc:  # pragma: no cover
            raise SourceAccessError(
                "vnstock is not installed. Run `pip install vnstock`."
            ) from exc
        self._ensure_registered()
        # period is NOT a constructor argument in current vnstock -- it is
        # passed per-call to balance_sheet()/income_statement()/etc.
        return Finance(symbol=ticker, source=backend or self.source_backend)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(RetryableFetchError),
        reraise=True,
    )
    def _call_statement(self, finance, statement_type: StatementType, ticker: str, period: Period):
        self._rate_limiter.wait()
        method = getattr(finance, statement_type)
        try:
            # `symbol` is bound at Finance(symbol=..., source=...)
            # construction time in current vnstock -- only `period` (and
            # optionally `lang`) are passed per-call.
            df = method(period=period, lang="en")
        except Exception as exc:  # noqa: BLE001
            # Network/timeout/5xx-style failures are worth retrying;
            # a clearly malformed-symbol error is not, but vnstock does
            # not currently distinguish these cleanly, so we retry a
            # bounded number of times and give up gracefully either way.
            raise RetryableFetchError(str(exc)) from exc
        return df

    # ------------------------------------------------------------------
    def _min_allowed_year(self) -> int:
        return date.today().year - self.lookback_years + 1

    def _filter_last_n_years(self, df, period: Period):
        """Keep only columns within the configured lookback window.

        VERIFIED 2026-08-24 against live vnstock/KBS output: statements
        come back in WIDE format -- one row per line item (account), one
        COLUMN per period, e.g. columns ['item', 'item_id', '2025-Năm',
        '2024-Năm', '2023-Năm', ...] for period='year', and presumably
        an analogous '<year>-Q<n>' pattern for period='quarter'. This is
        the opposite of the long/row-per-period format assumed in an
        earlier version of this file -- filtering must operate on
        COLUMNS, not rows.
        """
        if df is None or len(df) == 0:
            return df
        min_year = self._min_allowed_year()
        year_pattern = re.compile(r"(20\d{2})")
        keep_cols = []
        for c in df.columns:
            match = year_pattern.search(str(c))
            if match:
                year = int(match.group(1))
                if year >= min_year:
                    keep_cols.append(c)
                # else: column is older than the lookback window, drop it
            else:
                # Non-year columns (e.g. 'item', 'item_id') are always kept.
                keep_cols.append(c)
        if not keep_cols:
            return df
        return df[keep_cols]

    # ------------------------------------------------------------------
    def fetch_one(
        self, ticker: str, statement_type: StatementType, period: Period
    ) -> FetchResult:
        retrieved_at = datetime.now(timezone.utc)
        document_id = f"{ticker}_{statement_type}_{period}_{retrieved_at:%Y%m%d%H%M%S}"

        # Try the configured backend first, then fall back to alternate
        # legitimate providers if it comes back empty. VERIFIED 2026-08-24:
        # KBS returns genuinely empty data for balance_sheet on at least
        # some tickers/periods even though the call succeeds -- this is a
        # gap in that specific provider's coverage, not an error, so we
        # try another public, permitted source rather than give up.
        backends_to_try = [self.source_backend] + [
            b for b in ("KBS", "VCI", ) if b != self.source_backend
        ]

        last_error: str | None = None
        for backend in backends_to_try:
            try:
                finance = self._get_finance_client(ticker, period, backend)
                df = self._call_statement(finance, statement_type, ticker, period)
            except SourceAccessError as exc:
                last_error = str(exc)
                self.logger.error("Source inaccessible for %s (%s): %s", ticker, backend, exc)
                continue
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                self.logger.warning(
                    "Failed to fetch %s/%s/%s via %s after retries: %s",
                    ticker, statement_type, period, backend, exc,
                )
                continue

            df = self._filter_last_n_years(df, period)
            if df is not None and len(df) > 0:
                if backend != self.source_backend:
                    self.logger.info(
                        "%s/%s/%s: %s was empty, succeeded via fallback backend %s",
                        ticker, statement_type, period, self.source_backend, backend,
                    )
                bronze_path = self._write_bronze(
                    df, ticker, statement_type, period, retrieved_at, document_id
                )
                document_hash = self._hash_dataframe(df)
                return FetchResult(
                    ticker=ticker, statement_type=statement_type, period=period, rows=len(df),
                    bronze_path=str(bronze_path), document_id=document_id,
                    document_hash=document_hash, retrieved_at=retrieved_at, status="ok",
                )
            # else: this backend returned empty too, try the next one

        if last_error:
            return FetchResult(
                ticker=ticker, statement_type=statement_type, period=period, rows=0,
                bronze_path=None, document_id=document_id, document_hash=None,
                retrieved_at=retrieved_at, status="error", error=last_error,
            )
        return FetchResult(
            ticker=ticker, statement_type=statement_type, period=period, rows=0,
            bronze_path=None, document_id=document_id, document_hash=None,
            retrieved_at=retrieved_at, status="empty",
        )

    def fetch_all_for_ticker(self, ticker: str) -> Iterator[FetchResult]:
        for statement_type in STATEMENT_TYPES:
            for period in PERIODS:
                yield self.fetch_one(ticker, statement_type, period)

    # ------------------------------------------------------------------
    def _write_bronze(self, df, ticker, statement_type, period, retrieved_at, document_id) -> Path:
        out_dir = self.bronze_dir / "financial_statements" / statement_type / period
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ticker}_{retrieved_at:%Y%m%d}.parquet"

        enriched = df.copy()
        enriched["ticker"] = ticker
        enriched["statement_type"] = statement_type
        enriched["period"] = period
        enriched["source_name"] = _SOURCE_NAME
        enriched["source_url"] = _SOURCE_URL
        enriched["retrieved_at"] = retrieved_at.isoformat()
        enriched["document_id"] = document_id

        enriched = enriched.astype(str)
        enriched.to_parquet(out_path, index=False)
        return out_path

    @staticmethod
    def _hash_dataframe(df) -> str:
        payload = df.to_json(orient="records", date_format="iso").encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
