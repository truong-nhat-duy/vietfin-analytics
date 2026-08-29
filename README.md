<div align="center">

# VIETFIN ANALYTICS
### Enterprise Financial Intelligence & Analytical Data Platform for the Vietnamese Stock Market

[![Python Version](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![OLAP Engine](https://img.shields.io/badge/Engine-DuckDB-FFF000?style=flat&logo=duckdb&logoColor=black)](https://duckdb.org/)
[![Cloud Data Warehouse](https://img.shields.io/badge/Cloud-MotherDuck-FF6B00?style=flat)](https://motherduck.com/)
[![Architecture](https://img.shields.io/badge/Architecture-Medallion-blueviolet?style=flat)]()
[![Data Quality](https://img.shields.io/badge/Governance-Enforced-success?style=flat)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

### Enterprise Financial Intelligence & Analytical Data Platform for the Vietnamese Stock Market
<img src="logo.jpg" alt="VietFin Analytics Logo" width="200" style="border-radius: 50%;">

*An end-to-end, enterprise-grade ETL/ELT pipeline implementing Medallion Architecture to collect, clean, model, and serve financial statement data across all Vietnamese public exchanges (HOSE, HNX, UPCOM).*

</div>

---

## 📌 Executive Summary

**VietFin Analytics** is a scalable Data Engineering and Analytical Engine designed to standardize and process corporate financial statements from public markets in Vietnam. Built upon the **Medallion Data Lakehouse Architecture**, the platform ingests unstructured and semi-structured financial reporting metrics, standardizes them under the Vietnamese Accounting Standards (VAS) framework, and serves aggregated financial metrics via cloud-native OLAP engines (**MotherDuck / DuckDB**).

### Core Capabilities & Technical Metrics
* **Scale & Volume:** Manages **820,000+ granular financial statement line items** spanning balance sheets, income statements, and cash flow reports across the entire Vietnamese equity market.
* **Storage & Compute Paradigm:** Employs columnar local storage (**Apache Parquet**) for high-performance localized I/O, seamlessly integrated with **MotherDuck Cloud DW** for serverless, distributed analytical querying.
* **Feature Engineering (Gold Layer):** Automated calculation of corporate performance, liquidity, and leverage ratios (ROE, ROA, D/E, Profit Margins).
* **Data Governance & Provenance:** Full auditability via immutable tracking metadata, combined with strict execution rate-limiting.

---

## 🏗 Medallion Data Architecture

```text
              ┌─────────────────────────────────────────┐
              │     Data Ingestion (APIs/Exchanges)     │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ BRONZE LAYER (Raw Ingestion)                                              │
│ • Unprocessed JSON / Raw Data Lake persistent staging                     │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ SILVER LAYER (Modeled & Standardized)                                     │
│ • Normalized schema (`silver_financials`)                                 │
│ • Deduplicated, type-casted, standardized line-item mappings (VAS)        │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ GOLD LAYER (Analytical Feature Store)                                     │
│ • Curated metrics & financial ratios (`gold_financial_ratios`)            │
│ • Optimized for Machine Learning, Financial Modeling, and Streamlit Dash  │
└───────────────────────────────────────────────────────────────────────────┘
