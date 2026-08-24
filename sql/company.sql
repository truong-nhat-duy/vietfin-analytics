-- Useful queries against company_master / company_identifier_history

-- 1. Active companies per exchange
SELECT exchange, COUNT(*) AS n_companies
FROM company_master
WHERE status = 'active'
GROUP BY exchange
ORDER BY n_companies DESC;

-- 2. Duplicate (ticker, exchange) among active rows -- should return 0 rows
SELECT ticker, exchange, COUNT(*) AS n
FROM company_master
WHERE status = 'active'
GROUP BY ticker, exchange
HAVING COUNT(*) > 1;

-- 3. Companies with a ticker change (survivorship-bias check)
SELECT company_id, COUNT(DISTINCT ticker) AS n_tickers
FROM company_identifier_history
GROUP BY company_id
HAVING COUNT(DISTINCT ticker) > 1;

-- 4. Full identifier timeline for one company
SELECT *
FROM company_identifier_history
WHERE company_id = :company_id
ORDER BY valid_from;

-- 5. Companies missing required fields (data quality)
SELECT company_id, ticker, exchange
FROM company_master
WHERE company_name IS NULL
   OR exchange IS NULL
   OR source_name IS NULL;
