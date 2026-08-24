"""Placeholder connector for UPCOM (administratively run by HNX).

STATUS: NOT IMPLEMENTED. Kept as its own connector class for
architectural symmetry (Section III of the spec lists UPCOM as a
first-class source), even though UPCOM data is published via HNX's
systems. `VNStockUniverseCollector` already covers UPCOM through a
permitted public API.
"""

from __future__ import annotations

from typing import Iterator

from vietfin.ingestion.sources.base import BaseCollector, CollectedCompany


class UPCOMCollector(BaseCollector):
    source_name = "upcom_official"

    def fetch_companies(self, exchange: str | None = None) -> Iterator[CollectedCompany]:
        raise NotImplementedError(
            "UPCOMCollector is a placeholder. See module docstring in "
            "src/vietfin/ingestion/sources/upcom.py -- use "
            "VNStockUniverseCollector for a working UPCOM universe source, "
            "or implement this connector only after verifying legitimate "
            "access to the UPCOM data feed."
        )
        yield  # pragma: no cover
