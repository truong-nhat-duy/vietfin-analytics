import os
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
md_token = os.getenv("MOTHERDUCK_TOKEN")

if not md_token:
    print("❌ Lỗi: Chưa có token MotherDuck.")
else:
    con = duckdb.connect(f"md:vietfin_db?token={md_token}")
    print("⏳ Đang truy vấn dữ liệu từ MotherDuck...")
    
    query = """
        SELECT c.ticker
        FROM dim_company c
        LEFT JOIN (SELECT DISTINCT ticker FROM fact_financials) f ON c.ticker = f.ticker
        WHERE f.ticker IS NULL
        ORDER BY c.ticker
    """
    
    df_missing = con.execute(query).df()
    df_missing.to_parquet("missing_bctc.parquet", index=False)
    
    print(f"✅ Đã lưu danh sách {len(df_missing)} mã chứng khoán bị thiếu vào file: 'missing_bctc.parquet'")