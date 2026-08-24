"""Working connector: listed-company universe via the vnstock library.

vnstock (https://github.com/thinh-vu/vnstock) wraps public market-data
endpoints operated by Vietnamese data providers (KBS, VCI, TCBS) that
publish the full list of tickers listed on HOSE/HNX/UPCOM. This is the
"permitted public source" referred to in the project data policy, used
instead of scraping the exchanges' own websites directly (see hose.py /
hnx.py / upcom.py placeholders).

VERIFIED LIVE against vnstock 4.0.7 on 2026-08-24:

    from vnstock import Listing
    Listing(source='KBS').all_symbols()
    # -> columns ['symbol', 'organ_name'] ONLY -- no exchange field at all.

That call cannot tell us which exchange a ticker is on, so this
connector instead uses the "Unified UI" Reference domain, which the
vnstock README documents as having a dedicated exchange-listing method:

    from vnstock import Reference
    Reference().equity.list_by_exchange()   # [KBS] -> DataFrame

NOTE: `list_by_exchange()` itself has NOT yet been verified live (no
network access while writing this) -- run diagnose_vnstock.py's
companion check (or a quick manual call) before a full production run.
Column-matching below is written defensively (multiple candidate names)
specifically because of this, and will raise a clear SourceAccessError
with the real column list if none of the candidates match, so a
mismatch fails loudly instead of silently returning zero rows (the bug
that hit `all_symbols()`-based filtering earlier).

If vnstock's API surface changes again, only this file needs to change
-- callers depend solely on BaseCollector.fetch_companies().
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator

from tenacity import retry, stop_after_attempt, wait_exponential

from vietfin.ingestion.sources.base import (
    BaseCollector,
    CollectedCompany,
    SourceAccessError,
)

_EXCHANGES = ("HOSE", "HNX", "UPCOM")

_SOURCE_URL = "https://github.com/thinh-vu/vnstock"


class VNStockUniverseCollector(BaseCollector):
    """Fetches the company universe for HOSE, HNX, and UPCOM via vnstock."""

    source_name = "vnstock"

    def __init__(
        self, *args: Any, source_backend: str = "KBS", **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.source_backend = source_backend

    def _get_reference_client(self):
        """Lazily import vnstock so the rest of the codebase can be used /
        tested without the dependency installed."""
        try:
            from vnstock import Reference
        except ImportError as exc:  # pragma: no cover - exercised only w/o dep
            raise SourceAccessError(
                "vnstock is not installed. Run `pip install vnstock` or "
                "`pip install -e .` from the project root."
            ) from exc
        return Reference()

    def _get_listing_client(self):
        try:
            from vnstock import Listing
        except ImportError as exc:  # pragma: no cover
            raise SourceAccessError(
                "vnstock is not installed. Run `pip install vnstock` or "
                "`pip install -e .` from the project root."
            ) from exc
        return Listing(source=self.source_backend)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        reraise=True,
    )
    def _fetch_list_by_exchange(self, ref):
        self._rate_limiter.wait()
        try:
            df = ref.equity.list_by_exchange()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("vnstock Reference.equity.list_by_exchange() failed: %s", exc)
            raise
        if df is None or len(df) == 0:
            raise SourceAccessError(
                "vnstock returned no data from Reference().equity.list_by_exchange()"
            )
        return df

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        reraise=True,
    )
    def _fetch_all_symbols(self, listing):
        self._rate_limiter.wait()
        try:
            df = listing.all_symbols()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("vnstock all_symbols() failed: %s", exc)
            raise
        if df is None or len(df) == 0:
            raise SourceAccessError("vnstock returned no data from all_symbols()")
        return df

    def fetch_companies(
        self, exchange: str | None = None
    ) -> Iterator[CollectedCompany]:
        retrieved_at = datetime.now(timezone.utc)

        ref = self._get_reference_client()
        exch_df = self._fetch_list_by_exchange(ref)

        cols = {c.lower(): c for c in exch_df.columns}

        def col(*candidates: str):
            for c in candidates:
                if c in cols:
                    return cols[c]
            return None

        ticker_col = col("symbol", "ticker")
        exch_col = col("exchange", "board", "comgroupcode", "group")
        name_col = col("organ_name", "organname", "company_name", "companyname")

        if ticker_col is None or exch_col is None:
            raise SourceAccessError(
                "Reference().equity.list_by_exchange() schema not recognized "
                f"(need a ticker column and an exchange column); "
                f"columns={list(exch_df.columns)}. Update the column-matching "
                "logic in vnstock_source.py to match this real schema, then "
                "re-run."
            )

        # all_symbols() reliably carries the full company name (verified
        # live); use it to backfill name if list_by_exchange() doesn't
        # carry one, joining on ticker.
        name_by_ticker: dict[str, str] = {}
        if name_col is None:
            try:
                listing = self._get_listing_client()
                symbols_df = self._fetch_all_symbols(listing)
                s_cols = {c.lower(): c for c in symbols_df.columns}
                s_ticker_col = s_cols.get("symbol") or s_cols.get("ticker")
                s_name_col = s_cols.get("organ_name") or s_cols.get("organname")
                if s_ticker_col and s_name_col:
                    for _, r in symbols_df.iterrows():
                        name_by_ticker[str(r[s_ticker_col]).strip().upper()] = str(
                            r[s_name_col]
                        ).strip()
            except SourceAccessError:
                self.logger.warning(
                    "Could not backfill company names from all_symbols(); "
                    "company_name will fall back to ticker."
                )

        for _, row in exch_df.iterrows():
            raw_exchange = str(row[exch_col]).upper().strip()
            row_exchange = next((e for e in _EXCHANGES if e in raw_exchange), None)
            if row_exchange is None:
                continue
            if exchange and row_exchange != exchange:
                continue

            ticker = str(row[ticker_col]).strip().upper()
            if not ticker:
                continue

            if name_col:
                name_val = row[name_col]
                name = str(name_val).strip() if name_val not in (None, "", "nan") else ticker
            else:
                name = name_by_ticker.get(ticker, ticker)

            yield CollectedCompany(
                ticker=ticker,
                company_name=name or ticker,
                exchange=row_exchange,
                status="active",
                source_name=self.source_name,
                source_url=_SOURCE_URL,
                retrieved_at=retrieved_at,
                source_page="Reference.equity.list_by_exchange",
            )

