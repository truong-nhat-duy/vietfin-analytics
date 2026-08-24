# VIETFIN — Vietnam Corporate Financial Intelligence Platform
<div align="center">

# VIETFIN ANALYTICS
### Enterprise Financial Intelligence & Analytical Data Platform for the Vietnamese Stock Market

[![Python Version](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![OLAP Engine](https://img.shields.io/badge/Engine-DuckDB-FFF000?style=flat&logo=duckdb&logoColor=black)](https://duckdb.org/)
[![Cloud Data Warehouse](https://img.shields.io/badge/Cloud-MotherDuck-FF6B00?style=flat)](https://motherduck.com/)
[![Architecture](https://img.shields.io/badge/Architecture-Medallion-blueviolet?style=flat)]()
[![Data Quality](https://img.shields.io/badge/Governance-Enforced-success?style=flat)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img src="logo.png" alt="VietFin Analytics Logo" width="200" style="border-radius: 50%;">

*An end-to-end, enterprise-grade ETL/ELT pipeline implementing Medallion Architecture to collect, clean, model, and serve financial statement data across all Vietnamese public exchanges (HOSE, HNX, UPCOM).*

</div>

---

## 📌 Executive Summary

**VietFin Analytics** is a scalable Data Engineering and Analytical Engine designed to standardize and process corporate financial statements from public markets in Vietnam. Built upon the **Medallion Data Lakehouse Architecture**, the platform ingests unstructured and semi-structured financial reporting metrics, standardizes them under the Vietnamese Accounting Standards (VAS) framework, and serves aggregated financial metrics via cloud-native OLAP engines (**MotherDuck / DuckDB**).

### Core Capabilities & Technical Metrics
* **Scale & Volume:** Manages **820,000+ granular financial statement line items** spanning balance sheets, income statements, and cash flow reports across the entire Vietnamese equity market.
* **Storage & Compute Paradigm:** Employs columnar local storage (**Apache Parquet**) for high-performance localized I/O, seamlessly integrated with **MotherDuck Cloud DW** for serverless, distributed analytical querying.
* **Feature Engineering (Gold Layer):** Automated calculation of corporate performance, liquidity, and leverage ratios, including Return on Equity (**ROE**), Return on Assets (**ROA**), Debt-to-Equity (**D/E**), Gross Profit Margin, and Net Profit Margin.
* **Data Governance & Provenance:** Full auditability via immutable tracking metadata (`source_name`, `retrieved_at`, `document_hash`), combined with strict execution rate-limiting.

---

## 🏗 Medallion Data Architecture

The platform processes data through three distinct architectural layers to guarantee data lineage, quality, and fast analytical performance:
## Sprint 2: Company Universe

This sprint builds the foundation of the whole platform: the canonical list
of Vietnamese public companies (`company_master`) and their ticker/exchange
history (`company_identifier_history`), populated through a pluggable
connector architecture.

### What is in this sprint

- `src/vietfin/ingestion/sources/base.py` — `BaseCollector` abstract class
  every source connector implements.
- `src/vietfin/ingestion/sources/vnstock_source.py` — **working** connector.
  Pulls the listed-company universe for HOSE/HNX/UPCOM through the public,
  permitted market-data APIs wrapped by the open-source `vnstock` library
  (SSI FastConnect / VCI public endpoints). Supports pagination, retry,
  timeout, and structured logging.
- `src/vietfin/ingestion/sources/hose.py`, `hnx.py`, `upcom.py` — connector
  **interfaces/placeholders** for pulling directly from the exchanges'
  own sites. They are intentionally left unimplemented
  (`raise NotImplementedError`) because direct scraping of these sites has
  not been verified against current Terms of Service / robots.txt from
  this environment. Per the project's data policy, no bypass of
  authentication, CAPTCHA, or access controls is implemented anywhere in
  this codebase. Implement these only after confirming legitimate access
  (an official open-data API, a licensed data feed, or confirmed
  permission), following the same interface as `VNStockUniverseCollector`.
- `src/vietfin/ingestion/universe.py` — orchestrator that calls collectors,
  deduplicates, assigns **stable `company_id`**, and builds/updates
  `company_master` + `company_identifier_history`.
- `src/vietfin/database/duckdb.py` — DuckDB + Parquet persistence layer.
- `sql/schema.sql`, `sql/company.sql` — table definitions (DuckDB & Postgres
  compatible DDL).
- `config/settings.yaml`, `config/sources.yaml` — configuration, no
  hard-coded secrets/URLs in code.
- `scripts/01_build_universe.py` — CLI entry point for this sprint.
- `tests/test_ingestion.py` — unit tests (collector contract, id
  stability, dedup, identifier-history logic) using a fake in-memory
  collector, so tests do not depend on network access.

### Data policy reminder (enforced in code)

Every record collected carries `source_name`, `source_url`, `retrieved_at`,
and (where applicable) `document_id` / `document_hash` — see
`CollectedCompany` in `base.py`. No collector performs authentication
bypass, CAPTCHA solving, or rate-limit evasion. `RateLimiter` in
`base.py` enforces polite request pacing.

## How to install

```bash
cd vietfin
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## How to run

```bash
python scripts/01_build_universe.py --source vnstock --out data/gold
```

Options:

```bash
python scripts/01_build_universe.py --help
```

## How to test

```bash
pytest tests/ -v --cov=src/vietfin
```

## Expected output

- `data/gold/company_master.parquet`
- `data/gold/company_identifier_history.parquet`
- `data/vietfin.duckdb` with tables `company_master`,
  `company_identifier_history`
- A console data-quality summary (row counts, duplicate tickers found,
  missing exchange, per-source counts)

## Go / No-Go criteria for Sprint 3

See the end of the assistant's message for the full checklist. In short:
`company_master` must have no duplicate `(ticker, exchange)` for currently
active rows, every row must have a non-null `company_id` and `source_name`,
and the golden-sample tickers (Section XXXII of the spec) must resolve
successfully before starting document discovery (Sprint 3).
