"""Unit tests for VNStockFinancialsCollector.

Uses a fake `Finance` object monkeypatched in place of vnstock's real
class, so these tests need neither network access nor vnstock installed.
Sample data shape (item/item_id rows, one column per period like
'2026-Năm') and the column-based 10-year filter mirror the REAL vnstock
4.0.7 / KBS response verified live on 2026-08-24 -- see
vnstock_financials.py's module docstring.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from vietfin.ingestion.sources.vnstock_financials import VNStockFinancialsCollector


class FakeFinance:
    """Stands in for vnstock.Finance in tests."""

    def __init__(self, frame: pd.DataFrame | None = None, raise_on_call: bool = False):
        self._frame = frame
        self._raise = raise_on_call

    def balance_sheet(self, period, lang="en"):
        if self._raise:
            raise RuntimeError("simulated upstream failure")
        return self._frame

    income_statement = balance_sheet
    cash_flow = balance_sheet
    ratio = balance_sheet


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """Wide-format sample matching real vnstock/KBS output: one row per
    line item, one column per period (e.g. '2026-Năm')."""
    current_year = date.today().year
    return pd.DataFrame(
        {
            "item": ["Total assets", "Total liabilities", "Total equity"],
            "item_id": ["total_assets", "total_liabilities", "total_equity"],
            f"{current_year}-Năm": [1000, 600, 400],
            f"{current_year - 5}-Năm": [800, 500, 300],
            f"{current_year - 15}-Năm": [200, 150, 50],
        }
    )


def _single_backend_client(frame_or_collector):
    """Helper: returns a _get_finance_client replacement that ignores
    which backend is requested and always returns the same FakeFinance --
    used for tests where the fallback-to-other-backends path should never
    trigger (i.e. the first backend already succeeds or genuinely has no
    data anywhere)."""

    def _client(ticker, period, backend=None):
        return frame_or_collector

    return _client


def test_fetch_one_writes_bronze_and_reports_ok(tmp_path, monkeypatch, sample_frame):
    collector = VNStockFinancialsCollector(bronze_dir=tmp_path, min_request_interval_seconds=0)
    monkeypatch.setattr(
        collector, "_get_finance_client", _single_backend_client(FakeFinance(sample_frame))
    )

    result = collector.fetch_one("VNM", "balance_sheet", "year")

    assert result.status == "ok"
    assert result.rows == 3  # 3 line items, unaffected by column-based year filtering
    assert result.bronze_path is not None
    assert (tmp_path / "financial_statements" / "balance_sheet" / "year").exists()


def test_fetch_one_filters_columns_older_than_lookback(tmp_path, monkeypatch, sample_frame):
    collector = VNStockFinancialsCollector(
        bronze_dir=tmp_path, min_request_interval_seconds=0, lookback_years=10
    )
    monkeypatch.setattr(
        collector, "_get_finance_client", _single_backend_client(FakeFinance(sample_frame))
    )
    result = collector.fetch_one("VNM", "balance_sheet", "year")
    written = pd.read_parquet(result.bronze_path)
    # The column from 15 years ago should have been dropped; the current
    # and 5-years-ago period columns should remain.
    current_year = date.today().year
    assert f"{current_year}-Năm" in written.columns
    assert f"{current_year - 5}-Năm" in written.columns
    assert f"{current_year - 15}-Năm" not in written.columns


def test_fetch_one_reports_empty_when_all_backends_empty(tmp_path, monkeypatch):
    empty_df = pd.DataFrame(columns=["item", "item_id"])
    collector = VNStockFinancialsCollector(bronze_dir=tmp_path, min_request_interval_seconds=0)
    monkeypatch.setattr(
        collector, "_get_finance_client", _single_backend_client(FakeFinance(empty_df))
    )
    result = collector.fetch_one("XYZ", "balance_sheet", "year")
    assert result.status == "empty"
    assert result.bronze_path is None


def test_fetch_one_falls_back_to_alternate_backend_when_primary_empty(tmp_path, monkeypatch, sample_frame):
    """Mirrors the real, observed case: KBS returns empty for
    balance_sheet on some tickers even though the call succeeds; the
    collector should try VCI/TCBS next rather than giving up."""
    empty_df = pd.DataFrame(columns=["item", "item_id"])
    collector = VNStockFinancialsCollector(
        bronze_dir=tmp_path, min_request_interval_seconds=0, source_backend="KBS"
    )

    def fake_client(ticker, period, backend=None):
        if backend == "KBS":
            return FakeFinance(empty_df)
        # VCI (or any non-KBS fallback) has the data.
        return FakeFinance(sample_frame)

    monkeypatch.setattr(collector, "_get_finance_client", fake_client)
    result = collector.fetch_one("VNM", "balance_sheet", "year")
    assert result.status == "ok"
    assert result.rows == 3


def test_fetch_one_reports_error_instead_of_raising(tmp_path, monkeypatch):
    collector = VNStockFinancialsCollector(bronze_dir=tmp_path, min_request_interval_seconds=0)
    monkeypatch.setattr(
        collector, "_get_finance_client", _single_backend_client(FakeFinance(raise_on_call=True))
    )
    # fetch_one must not raise -- callers (the CLI script) rely on this
    # to keep looping across thousands of tickers.
    result = collector.fetch_one("BAD", "balance_sheet", "year")
    assert result.status == "error"
    assert result.error is not None


def test_bronze_output_has_provenance_columns(tmp_path, monkeypatch, sample_frame):
    collector = VNStockFinancialsCollector(bronze_dir=tmp_path, min_request_interval_seconds=0)
    monkeypatch.setattr(
        collector, "_get_finance_client", _single_backend_client(FakeFinance(sample_frame))
    )
    result = collector.fetch_one("VNM", "balance_sheet", "year")
    written = pd.read_parquet(result.bronze_path)
    for col in ("ticker", "statement_type", "period", "source_name", "source_url", "retrieved_at", "document_id"):
        assert col in written.columns
