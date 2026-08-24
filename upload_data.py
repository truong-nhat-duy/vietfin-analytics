import os
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("MOTHERDUCK_TOKEN")

if not token:
    raise ValueError("Thiếu MOTHERDUCK_TOKEN trong file .env")

# 1. Kết nối tới Cloud Data Warehouse
con = duckdb.connect(f"md:vietfin_db?token={token}")

# 2. Đọc dữ liệu đã cào (Thay bằng DataFrame hoặc file CSV/Parquet của bạn)
# Ví dụ dữ liệu sau khi cào và tính toán chỉ số:
df_scraped = pd.read_csv("gold_financial_ratios.csv") 

# 3. Ghi đè / Cập nhật bảng gold_financial_ratios trên Cloud
con.register("df_temp", df_scraped)
con.execute("""
    CREATE OR REPLACE TABLE gold_financial_ratios AS 
    SELECT * FROM df_temp
""")

print("✅ Đã đồng bộ thành công dữ liệu cào lên MotherDuck Cloud!")