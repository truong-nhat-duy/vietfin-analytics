import os
import duckdb
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("MOTHERDUCK_TOKEN")

if not token:
    print("❌ Thiếu MOTHERDUCK_TOKEN")
    exit()

print("🔌 Kết nối MotherDuck...")
con = duckdb.connect(f"md:vietfin_db?token={token}")

# Danh sách các dataset bạn đã cào
datasets = ['overview', 'officers', 'shareholders', 'ratio_summary', 'price_history']
# SỬA LẠI ĐƯỜNG DẪN Ở ĐÂY: Thêm '/corporate'
base_path = 'data/bronze/corporate'

for ds in datasets:
    # Dùng glob pattern của DuckDB để đọc tất cả file parquet của dataset đó
    parquet_path = f"{base_path}/{ds}/*.parquet" 
    table_name = f"gold_corporate_{ds}"
    
    try:
        print(f"🔄 Đang đồng bộ {ds} lên bảng {table_name}...")
        # Sử dụng ignore_errors=true để bỏ qua các file lỗi/rỗng nếu có
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS 
            SELECT * FROM read_parquet('{parquet_path}', hive_partitioning=1, union_by_name=true)
        """)
        
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"✅ Thành công: {count} bản ghi trong {table_name}")
    except Exception as e:
        print(f"⚠️ Bỏ qua {ds} (Lỗi: {e})")

print("🎉 Hoàn tất đồng bộ lên MotherDuck!")