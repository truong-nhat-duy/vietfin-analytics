import duckdb
from pathlib import Path

GOLD_DIR = Path(r"D:\vietfin\data\gold")
DB_PATH = r"D:\vietfin\data\vietfin_gold.duckdb"

# Kết nối (tự động tạo mới file database)
con = duckdb.connect(DB_PATH)

print("--- Đang nạp toàn bộ dữ liệu Gold vào DuckDB ---")

# Nạp tự động từng file Parquet thành bảng trong Database
parquet_files = list(GOLD_DIR.glob("*.parquet"))

for file_path in parquet_files:
    table_name = file_path.stem  # Lấy tên file làm tên bảng
    print(f"Đang nạp bảng: {table_name}...")
    
    # Lệnh DuckDB đọc trực tiếp Parquet không qua RAM
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM '{file_path}';")

print(f"\n✅ ĐÃ ĐỒNG BỘ THÀNH CÔNG VÀO DATABASE: {DB_PATH}")

# Thống kê nhanh các bảng
tables = con.execute("SHOW TABLES;").fetchall()
print("Danh sách các bảng đã sẵn sàng cho App:")
for t in tables:
    count = con.execute(f"SELECT COUNT(*) FROM {t[0]};").fetchone()[0]
    print(f"  - {t[0]}: {count:,} dòng")

con.close()