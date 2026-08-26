import duckdb

# Đường dẫn DB cục bộ và Token MotherDuck
LOCAL_DB_PATH = r"D:\vietfin\data\vietfin_gold.duckdb"
MOTHERDUCK_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InRydW9uZ25oYXRkdXlAZ21haWwuY29tIiwibWRSZWdpb24iOiJhd3MtYXAtbm9ydGhlYXN0LTEiLCJzZXNzaW9uIjoidHJ1b25nbmhhdGR1eS5nbWFpbC5jb20iLCJwYXQiOiJET0ZaRjRMVWJfTWIyWG1ScEJGd3ZvQU05RVljMEZlZ0dKUE5FQWIwbGc0IiwidXNlcklkIjoiYzAxMDM4NzItMjljOS00MDMyLTljZGMtNGQ3NzVjOWFjZmM3IiwiaXNzIjoibWRfcGF0IiwicmVhZE9ubHkiOmZhbHNlLCJ0b2tlblR5cGUiOiJyZWFkX3dyaXRlIiwiaWF0IjoxNzg3NTY4OTQ3fQ.vqbt25sKhTTLBmeffS-AXC57Yfs5ooInwVQZ9nXjgug"

print("--- Đang kết nối tới MotherDuck Cloud ---")
# Kết nối đồng thời DB local và DB MotherDuck
con = duckdb.connect(f"md:vietfin_db?token={MOTHERDUCK_TOKEN}")
con.execute(f"ATTACH '{LOCAL_DB_PATH}' AS local_db;")

# Danh sách các bảng Gold
tables = [
    "dim_company",
    "dim_officers",
    "dim_shareholders",
    "fact_daily_prices",
    "fact_ratio_summary",
    "fact_financials",
]

for t in tables:
    print(f"Đang đồng bộ bảng {t} lên MotherDuck...")
    con.execute(f"CREATE OR REPLACE TABLE vietfin_db.main.{t} AS SELECT * FROM local_db.main.{t};")

print("✅ ĐÃ ĐỒNG BỘ TOÀN BỘ CƠ SỞ DỮ LIỆU LÊN MOTHERDUCK CLOUD THÀNH CÔNG!")
con.close()