-- VIETFIN core schema (Sprint 2 scope)
-- Compatible with both DuckDB and PostgreSQL (avoid engine-specific types).

CREATE TABLE IF NOT EXISTS company_master (
    company_id          VARCHAR PRIMARY KEY,   -- stable id, NOT the ticker
    ticker              VARCHAR NOT NULL,       -- current ticker
    company_name        VARCHAR NOT NULL,
    exchange            VARCHAR NOT NULL,       -- HOSE | HNX | UPCOM
    isin                VARCHAR,
    listing_date        DATE,
    delisting_date      DATE,
    status              VARCHAR NOT NULL,       -- active | delisted | suspended
    sector              VARCHAR,
    industry            VARCHAR,
    website             VARCHAR,
    source_name         VARCHAR NOT NULL,
    source_url          VARCHAR,
    retrieved_at        TIMESTAMP NOT NULL,
    created_at          TIMESTAMP NOT NULL,
    updated_at          TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS company_identifier_history (
    company_id          VARCHAR NOT NULL,
    ticker              VARCHAR NOT NULL,
    exchange            VARCHAR NOT NULL,
    valid_from          DATE NOT NULL,
    valid_to            DATE,                   -- NULL = still current
    status              VARCHAR NOT NULL,       -- active | delisted | ticker_changed | exchange_changed
    source_name         VARCHAR NOT NULL,
    retrieved_at        TIMESTAMP NOT NULL,
    PRIMARY KEY (company_id, ticker, exchange, valid_from)
);

CREATE INDEX IF NOT EXISTS idx_company_master_ticker
    ON company_master (ticker, exchange);

CREATE INDEX IF NOT EXISTS idx_identifier_history_company
    ON company_identifier_history (company_id);

-- Pipeline run metadata (Section XXIX Observability), used by every sprint.
CREATE TABLE IF NOT EXISTS pipeline_run (
    run_id              VARCHAR PRIMARY KEY,
    dag_name            VARCHAR NOT NULL,
    started_at          TIMESTAMP NOT NULL,
    finished_at         TIMESTAMP,
    status              VARCHAR NOT NULL,       -- running | success | failed
    records_in          INTEGER,
    records_out         INTEGER,
    errors              INTEGER,
    notes               VARCHAR
);
