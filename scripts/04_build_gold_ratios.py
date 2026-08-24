import os
import duckdb
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('MOTHERDUCK_TOKEN')
if not token:
    raise ValueError('LỖI: Không tìm thấy MOTHERDUCK_TOKEN!')

con = duckdb.connect(f'md:vietfin_db?token={token}')
print('---> Kết nối MotherDuck thành công. Bắt đầu xử lý Tầng Gold...')

gold_sql = """
CREATE OR REPLACE TABLE gold_financial_ratios AS
WITH pivoted AS (
    SELECT 
        ticker,
        period,
        report_period,
        MAX(CASE WHEN lower(item) LIKE '%doanh thu thuần%' OR item_id IN ('10', '1') THEN value END) AS net_revenue,
        MAX(CASE WHEN lower(item) LIKE '%lợi nhuận gộp%' OR item_id IN ('20', '11') THEN value END) AS gross_profit,
        MAX(CASE WHEN lower(item) LIKE '%lợi nhuận sau thuế%' OR item_id IN ('60', '61', '21') THEN value END) AS net_income,
        MAX(CASE WHEN lower(item) LIKE '%tổng tài sản%' OR item_id IN ('100', '270') THEN value END) AS total_assets,
        MAX(CASE WHEN lower(item) LIKE '%nợ phải trả%' OR item_id IN ('300', '310') THEN value END) AS total_liabilities,
        MAX(CASE WHEN lower(item) LIKE '%vốn chủ sở hữu%' OR item_id IN ('400', '410') THEN value END) AS total_equity
    FROM silver_financials
    GROUP BY ticker, period, report_period
)
SELECT 
    ticker,
    period,
    report_period,
    net_revenue,
    gross_profit,
    net_income,
    total_assets,
    total_liabilities,
    total_equity,
    ROUND(net_income / NULLIF(total_equity, 0) * 100, 2) AS roe_pct,
    ROUND(net_income / NULLIF(total_assets, 0) * 100, 2) AS roa_pct,
    ROUND(total_liabilities / NULLIF(total_equity, 0), 2) AS debt_to_equity,
    ROUND(gross_profit / NULLIF(net_revenue, 0) * 100, 2) AS gross_margin_pct,
    ROUND(net_income / NULLIF(net_revenue, 0) * 100, 2) AS net_margin_pct
FROM pivoted
WHERE net_revenue IS NOT NULL OR total_assets IS NOT NULL
ORDER BY ticker, report_period DESC;
"""

con.execute(gold_sql)
count = con.execute('SELECT COUNT(*), COUNT(DISTINCT ticker) FROM gold_financial_ratios').fetchone()
print(f'---> ĐÃ TẠO BẢNG GOLD THÀNH CÔNG: {count[0]:,} dòng của {count[1]} mã cổ phiếu!')

os.makedirs('data/gold', exist_ok=True)
con.execute("COPY gold_financial_ratios TO 'data/gold/financial_ratios.parquet' (FORMAT PARQUET)")
print('---> Đã lưu bản sao Parquet tại: data/gold/financial_ratios.parquet')
