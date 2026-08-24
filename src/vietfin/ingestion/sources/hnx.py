"""Placeholder connector for the Hanoi Stock Exchange (HNX).

STATUS: NOT IMPLEMENTED. See hose.py for the rationale -- the same
applies here for https://www.hnx.vn. `VNStockUniverseCollector` already
covers HNX through a permitted public API.
"""

from __future__ import annotations

from typing import Iterator

from vietfin.ingestion.sources.base import BaseCollector, CollectedCompany


class HNXCollector(BaseCollector):
    source_name = "hnx_official"

    def fetch_companies(self, exchange: str | None = None) -> Iterator[CollectedCompany]:
        raise NotImplementedError(
            "HNXCollector is a placeholder. See module docstring in "
            "src/vietfin/ingestion/sources/hnx.py -- use "
            "VNStockUniverseCollector for a working HNX universe source, "
            "or implement this connector only after verifying legitimate "
            "access to www.hnx.vn."
        )
        yield  # pragma: no cover
