import os
import duckdb
from dotenv import load_dotenv

# Tải biến môi trường từ file .env vừa tạo
load_dotenv()
token = os.getenv("MOTHERDUCK_TOKEN")

# Kết nối và kiểm tra dữ liệu
con = duckdb.connect(f"md:vietfin_db?token={token}")
count = con.execute("SELECT COUNT(*) FROM silver_financials").fetchone()[0]
print(f"---> Kết nối thành công! Tổng số bản ghi trên Cloud: {count:,}")