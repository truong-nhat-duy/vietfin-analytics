"""Placeholder connector for the Ho Chi Minh Stock Exchange (HOSE).

STATUS: NOT IMPLEMENTED.

This class exists to satisfy the connector architecture (Section III of
the spec: `Source -> HOSE / HNX / UPCOM / CafeF / ...`), but direct
scraping of https://www.hsx.vn has not been implemented because:

  1. Its current terms of use / robots.txt have not been verified from
     this environment, and
  2. `VNStockUniverseCollector` already provides a legitimately-sourced
     HOSE universe via a permitted public API (see
     vietfin/ingestion/sources/vnstock_source.py), so there is no need to
     risk an unverified scrape to unblock Sprint 2.

To implement this connector for real, a human should:
  * Confirm HOSE publishes a public/open-data feed for the listed-company
    directory (preferred), or obtain a licensed data agreement.
  * Respect robots.txt and any documented rate limits.
  * Never solve CAPTCHAs or bypass authentication programmatically.
  * Populate CollectedCompany.source_url / source_page for every row.

Then flip `status: active` for `hose` in config/sources.yaml.
"""

from __future__ import annotations

from typing import Iterator

from vietfin.ingestion.sources.base import BaseCollector, CollectedCompany


class HOSECollector(BaseCollector):
    source_name = "hose_official"

    def fetch_companies(self, exchange: str | None = None) -> Iterator[CollectedCompany]:
        raise NotImplementedError(
            "HOSECollector is a placeholder. See module docstring in "
            "src/vietfin/ingestion/sources/hose.py -- use "
            "VNStockUniverseCollector for a working HOSE universe source, "
            "or implement this connector only after verifying legitimate "
            "access to www.hsx.vn."
        )
        yield  # pragma: no cover - keeps this a generator function
