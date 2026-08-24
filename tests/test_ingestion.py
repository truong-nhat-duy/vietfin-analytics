"""Unit tests for Sprint 2: collector contract, id stability, dedup,
identifier-history construction.

These tests use a FakeCollector so they run with no network access,
per project testing standards (Section XXVII).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Iterator

import pytest

from vietfin.ingestion.sources.base import (
    BaseCollector,
    CollectedCompany,
    SourceAccessError,
)
from vietfin.ingestion.universe import (
    build_company_master,
    build_identifier_history,
    collect_all,
    data_quality_report,
    deduplicate,
    make_stable_company_id,
)


class FakeCollector(BaseCollector):
    source_name = "fake_source"

    def __init__(self, records: list[CollectedCompany], fail: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._records = records
        self._fail = fail

    def fetch_companies(self, exchange: str | None = None) -> Iterator[CollectedCompany]:
        if self._fail:
            raise SourceAccessError("simulated outage")
        for rec in self._records:
            if exchange is None or rec.exchange == exchange:
                yield rec


def make_company(**overrides) -> CollectedCompany:
    defaults = dict(
        ticker="ABC",
        company_name="ABC Corp",
        exchange="HOSE",
        source_name="fake_source",
        retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return CollectedCompany(**defaults)


# ----------------------------------------------------------------------
# CollectedCompany contract
# ----------------------------------------------------------------------

def test_collected_company_requires_source_name():
    with pytest.raises(ValueError):
        CollectedCompany(ticker="ABC", company_name="ABC Corp", exchange="HOSE", source_name="")


def test_collected_company_requires_ticker():
    with pytest.raises(ValueError):
        make_company(ticker="")


def test_collected_company_requires_exchange():
    with pytest.raises(ValueError):
        make_company(exchange="")


# ----------------------------------------------------------------------
# company_id stability
# ----------------------------------------------------------------------

def test_company_id_deterministic_same_inputs():
    id1 = make_stable_company_id("ABC", "HOSE")
    id2 = make_stable_company_id("ABC", "HOSE")
    assert id1 == id2


def test_company_id_differs_by_exchange():
    id_hose = make_stable_company_id("ABC", "HOSE")
    id_hnx = make_stable_company_id("ABC", "HNX")
    assert id_hose != id_hnx


def test_company_id_prefers_isin_when_present():
    id_with_isin = make_stable_company_id("ABC", "HOSE", isin="VN000000ABC1")
    id_same_isin_diff_ticker = make_stable_company_id("XYZ", "HNX", isin="VN000000ABC1")
    assert id_with_isin == id_same_isin_diff_ticker


def test_company_id_is_not_the_ticker():
    company_id = make_stable_company_id("ABC", "HOSE")
    assert company_id != "ABC"
    assert company_id.startswith("CID_")


# ----------------------------------------------------------------------
# collect_all: multi-collector merge + fault tolerance
# ----------------------------------------------------------------------

def test_collect_all_merges_multiple_collectors():
    c1 = FakeCollector([make_company(ticker="AAA"), make_company(ticker="BBB")])
    c2 = FakeCollector([make_company(ticker="CCC")])
    records = collect_all([c1, c2])
    tickers = {r.ticker for r in records}
    assert tickers == {"AAA", "BBB", "CCC"}


def test_collect_all_tolerates_one_collector_failing():
    good = FakeCollector([make_company(ticker="AAA")])
    bad = FakeCollector([], fail=True)
    records = collect_all([bad, good])
    assert [r.ticker for r in records] == ["AAA"]


def test_collect_all_respects_exchange_filter():
    c1 = FakeCollector(
        [make_company(ticker="AAA", exchange="HOSE"), make_company(ticker="BBB", exchange="HNX")]
    )
    records = collect_all([c1], exchange="HOSE")
    assert [r.ticker for r in records] == ["AAA"]


# ----------------------------------------------------------------------
# deduplicate
# ----------------------------------------------------------------------

def test_deduplicate_keeps_most_recent():
    old = make_company(ticker="AAA", company_name="Old Name", retrieved_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    new = make_company(ticker="AAA", company_name="New Name", retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    deduped = deduplicate([old, new])
    assert len(deduped) == 1
    assert deduped[0].company_name == "New Name"


def test_deduplicate_treats_different_exchange_as_distinct():
    a = make_company(ticker="AAA", exchange="HOSE")
    b = make_company(ticker="AAA", exchange="HNX")
    deduped = deduplicate([a, b])
    assert len(deduped) == 2


# ----------------------------------------------------------------------
# build_company_master / build_identifier_history
# ----------------------------------------------------------------------

def test_build_company_master_shape():
    records = [make_company(ticker="AAA"), make_company(ticker="BBB")]
    df = build_company_master(records)
    assert df.height == 2
    assert set(["company_id", "ticker", "exchange", "source_name"]).issubset(df.columns)
    assert df["company_id"].null_count() == 0


def test_build_company_master_empty_input():
    df = build_company_master([])
    assert df.height == 0


def test_build_identifier_history_defaults_valid_from_to_today_when_no_listing_date():
    records = [make_company(ticker="AAA", listing_date=None)]
    df = build_identifier_history(records, as_of=date(2026, 8, 16))
    assert df.height == 1
    assert df["valid_from"][0] == date(2026, 8, 16)
    assert df["valid_to"][0] is None


def test_build_identifier_history_uses_listing_date_when_present():
    records = [make_company(ticker="AAA", listing_date=date(2015, 6, 1))]
    df = build_identifier_history(records)
    assert df["valid_from"][0] == date(2015, 6, 1)


# ----------------------------------------------------------------------
# data_quality_report
# ----------------------------------------------------------------------

def test_data_quality_report_flags_duplicates():
    # Build a master df manually with a forced duplicate (bypassing dedup)
    # to verify the report catches it.
    records = [
        make_company(ticker="AAA", exchange="HOSE"),
    ]
    df = build_company_master(records)
    # Duplicate the single row to simulate a data issue slipping through.
    import polars as pl

    df2 = pl.concat([df, df])
    report = data_quality_report(df2)
    assert report["duplicate_ticker_exchange_pairs"] == 1


def test_data_quality_report_empty():
    import polars as pl

    df = pl.DataFrame(schema={"company_name": pl.Utf8, "exchange": pl.Utf8, "source_name": pl.Utf8, "status": pl.Utf8, "ticker": pl.Utf8})
    report = data_quality_report(df)
    assert report["total_companies"] == 0


def test_data_quality_report_flags_missing_required_fields():
    records = [make_company(ticker="AAA")]
    df = build_company_master(records)
    import polars as pl

    df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("company_name_new"))
    df = df.drop("company_name").rename({"company_name_new": "company_name"})
    report = data_quality_report(df)
    assert report["rows_missing_required_fields"] == 1


# ----------------------------------------------------------------------
# Placeholder connectors must clearly fail, never silently no-op
# ----------------------------------------------------------------------

def test_hose_placeholder_raises_not_implemented():
    from vietfin.ingestion.sources.hose import HOSECollector

    collector = HOSECollector()
    with pytest.raises(NotImplementedError):
        next(iter(collector.fetch_companies()))


def test_hnx_placeholder_raises_not_implemented():
    from vietfin.ingestion.sources.hnx import HNXCollector

    collector = HNXCollector()
    with pytest.raises(NotImplementedError):
        next(iter(collector.fetch_companies()))


def test_upcom_placeholder_raises_not_implemented():
    from vietfin.ingestion.sources.upcom import UPCOMCollector

    collector = UPCOMCollector()
    with pytest.raises(NotImplementedError):
        next(iter(collector.fetch_companies()))
