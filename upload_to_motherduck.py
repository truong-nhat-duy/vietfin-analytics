import os
import duckdb

# 1. Điền MotherDuck Token của bạn vào đây (hoặc đặt biến môi trường MOTHERDUCK_TOKEN)
MOTHERDUCK_TOKEN = "CHEP_TOKEN_CUA_BAN_VAO_DAY"
DATABASE_NAME = "vietfin_db"  # Tên Database trên Cloud
PARQUET_PATH = "data/silver/financial_statements_long.parquet" # Đường dẫn file Parquet local

def upload_parquet_to_cloud():
    # Khởi tạo kết nối tới MotherDuck
    con = duckdb.connect(f"md:{DATABASE_NAME}?token={MOTHERDUCK_TOKEN}")

    print(f"Đang đẩy file {PARQUET_PATH} lên MotherDuck Cloud...")

    # Tạo bảng mới (hoặc ghi đè) trực tiếp từ file Parquet local
    con.execute(f"""
        CREATE OR REPLACE TABLE silver_financial_statements AS 
        SELECT * FROM read_parquet('{PARQUET_PATH}');
    """)

    # Kiểm tra số bản ghi đã tải lên
    result = con.execute("SELECT COUNT(*) FROM silver_financial_statements").fetchone()
    print(f"Thành công! Đã đẩy {result[0]:,} dòng lên bảng 'silver_financial_statements' trên Cloud.")

if __name__ == "__main__":
    upload_parquet_to_cloud()