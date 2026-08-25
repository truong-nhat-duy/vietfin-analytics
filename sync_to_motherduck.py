import duckdb
import os
import glob

# Thay token của bạn vào đây (cẩn thận không push file này lên Github)
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InRydW9uZ25oYXRkdXlAZ21haWwuY29tIiwibWRSZWdpb24iOiJhd3MtYXAtbm9ydGhlYXN0LTEiLCJzZXNzaW9uIjoidHJ1b25nbmhhdGR1eS5nbWFpbC5jb20iLCJwYXQiOiJET0ZaRjRMVWJfTWIyWG1ScEJGd3ZvQU05RVljMEZlZ0dKUE5FQWIwbGc0IiwidXNlcklkIjoiYzAxMDM4NzItMjljOS00MDMyLTljZGMtNGQ3NzVjOWFjZmM3IiwiaXNzIjoibWRfcGF0IiwicmVhZE9ubHkiOmZhbHNlLCJ0b2tlblR5cGUiOiJyZWFkX3dyaXRlIiwiaWF0IjoxNzg3NTY4OTQ3fQ.vqbt25sKhTTLBmeffS-AXC57Yfs5ooInwVQZ9nXjgug" 

print("Đang kết nối tới MotherDuck...")
con = duckdb.connect(f'md:?token={TOKEN}')

con.execute('CREATE DATABASE IF NOT EXISTS vietfin_db')
con.execute('USE vietfin_db')

# Danh sách các dataset cần tìm
datasets = {
    'gold_financial_ratios': 'ratio_summary',
    'gold_corporate_overview': 'overview',
    'gold_corporate_shareholders': 'shareholders',
    'gold_corporate_officers': 'officers',
    'gold_corporate_price_history': 'price_history'
}

for table_name, dataset_name in datasets.items():
    # Kịch bản quét tìm file thông minh ở cả bronze và silver
    possible_paths = [
        f'data/silver/{dataset_name}.parquet',
        f'data/bronze/{dataset_name}.parquet',
        f'data/silver/{dataset_name}/*.parquet',
        f'data/bronze/{dataset_name}/*.parquet'
    ]
    
    found = False
    for path in possible_paths:
        # Nếu đường dẫn có chứa file parquet
        if glob.glob(path):
            print(f"🔄 Đang tìm thấy dữ liệu tại: {path}")
            print(f"   -> Đang đẩy lên bảng {table_name}...")
            try:
                con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{path}')")
                count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                print(f"✅ THÀNH CÔNG! Bảng {table_name} có {count:,} dòng.\n")
                found = True
            except Exception as e:
                print(f"❌ LỖI khi đọc {path}: {e}")
            break # Tìm thấy và xử lý xong thì thoát vòng lặp path
            
    if not found:
        print(f"⚠️ KHÔNG TÌM THẤY dữ liệu cho {dataset_name}. Hãy kiểm tra lại tool cào.\n")

print("🎉 ĐÃ HOÀN TẤT QUÁ TRÌNH KIỂM TRA VÀ ĐỒNG BỘ!")