import os
import duckdb
from dotenv import load_dotenv

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN TỰ ĐỘNG (D:\vietfin)
# ==========================================
# Xác định thư mục gốc của dự án dựa trên vị trí của script hiện tại
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # D:\vietfin\scripts
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              # D:\vietfin
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')           # D:\vietfin\.env
GOLD_DIR = os.path.join(PROJECT_ROOT, 'data', 'gold')   # D:\vietfin\data\gold
PARQUET_FILE = os.path.join(GOLD_DIR, 'financial_ratios.parquet')

# Load biến môi trường từ file .env ở thư mục gốc
load_dotenv(ENV_PATH)
token = os.getenv('MOTHERDUCK_TOKEN')
if not token:
    raise ValueError('❌ LỖI: Không tìm thấy MOTHERDUCK_TOKEN! Vui lòng kiểm tra file .env')

# ==========================================
# 2. KẾT NỐI & XỬ LÝ DỮ LIỆU TẦNG GOLD
# ==========================================
print('---> Kết nối MotherDuck thành công. Bắt đầu xử lý Tầng Gold...')
con = duckdb.connect(f'md:vietfin_db?token={token}')

# 🛠️ Cập nhật: Thêm TRY_CAST để ép kiểu số và COALESCE để xử lý triệt để giá trị NULL
gold_sql = """
CREATE OR REPLACE TABLE gold_financial_ratios AS
WITH pivoted AS (
    SELECT 
        ticker,
        period,
        report_period,
        -- Ép kiểu dữ liệu về DOUBLE để tính toán chính xác, rỗng thì mặc định là 0
        COALESCE(MAX(CASE WHEN lower(item) LIKE '%doanh thu thuần%' OR item_id IN ('10', '1') THEN TRY_CAST(value AS DOUBLE) END), 0) AS net_revenue,
        COALESCE(MAX(CASE WHEN lower(item) LIKE '%lợi nhuận gộp%' OR item_id IN ('20', '11') THEN TRY_CAST(value AS DOUBLE) END), 0) AS gross_profit,
        COALESCE(MAX(CASE WHEN lower(item) LIKE '%lợi nhuận sau thuế%' OR item_id IN ('60', '61', '21') THEN TRY_CAST(value AS DOUBLE) END), 0) AS net_income,
        COALESCE(MAX(CASE WHEN lower(item) LIKE '%tổng tài sản%' OR item_id IN ('100', '270') THEN TRY_CAST(value AS DOUBLE) END), 0) AS total_assets,
        COALESCE(MAX(CASE WHEN lower(item) LIKE '%nợ phải trả%' OR item_id IN ('300', '310') THEN TRY_CAST(value AS DOUBLE) END), 0) AS total_liabilities,
        COALESCE(MAX(CASE WHEN lower(item) LIKE '%vốn chủ sở hữu%' OR item_id IN ('400', '410') THEN TRY_CAST(value AS DOUBLE) END), 0) AS total_equity
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
    -- Tính toán các chỉ số an toàn với NULLIF (tránh lỗi chia cho 0), nếu lỗi thì trả về 0
    COALESCE(ROUND(net_income / NULLIF(total_equity, 0) * 100, 2), 0) AS roe_pct,
    COALESCE(ROUND(net_income / NULLIF(total_assets, 0) * 100, 2), 0) AS roa_pct,
    COALESCE(ROUND(total_liabilities / NULLIF(total_equity, 0), 2), 0) AS debt_to_equity,
    COALESCE(ROUND(gross_profit / NULLIF(net_revenue, 0) * 100, 2), 0) AS gross_margin_pct,
    COALESCE(ROUND(net_income / NULLIF(net_revenue, 0) * 100, 2), 0) AS net_margin_pct
FROM pivoted
WHERE net_revenue != 0 OR total_assets != 0
ORDER BY ticker, report_period DESC;
"""

con.execute(gold_sql)

# Kiểm tra log số lượng dữ liệu
count = con.execute('SELECT COUNT(*), COUNT(DISTINCT ticker) FROM gold_financial_ratios').fetchone()
print(f'---> ĐÃ TẠO BẢNG GOLD THÀNH CÔNG: {count[0]:,} dòng của {count[1]} mã cổ phiếu!')

# ==========================================
# 3. EXPORT FILE PARQUET
# ==========================================
# Đảm bảo thư mục data/gold tồn tại trong dự án gốc
os.makedirs(GOLD_DIR, exist_ok=True)

# Đổi dấu \ thành / để tương thích với câu lệnh COPY của DuckDB trên Windows
parquet_path_sql = PARQUET_FILE.replace('\\', '/')

con.execute(f"COPY gold_financial_ratios TO '{parquet_path_sql}' (FORMAT PARQUET)")
print(f'---> Đã lưu bản sao Parquet tại: {PARQUET_FILE}')