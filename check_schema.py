import os
import duckdb
from dotenv import load_dotenv

# 1. Tải biến môi trường và kết nối
load_dotenv()
token = os.getenv("MOTHERDUCK_TOKEN")

if not token:
    print("❌ Thiếu MOTHERDUCK_TOKEN trong file .env")
    exit(1)

print("🔌 Đang kết nối tới MotherDuck...")
con = duckdb.connect(f"md:vietfin_db?token={token}")

print("🔍 Đang quét cấu trúc và kiểm tra dữ liệu toàn diện...")

# Lấy danh sách các bảng người dùng trong schema 'main' (Lọc bỏ các bảng hệ thống)
tables = con.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'main' 
      AND table_type = 'BASE TABLE'
    ORDER BY table_name
""").fetchall()

report_lines = ["=== BÁO CÁO CHẨN ĐOÁN DỮ LIỆU VIETFIN (MOTHERDUCK) ==="]

for table in tables:
    table_name = table[0]
    
    # Bỏ qua các bảng hệ thống của MotherDuck nếu còn sót
    if table_name.startswith("md_") or table_name == "database_snapshots":
        continue

    report_lines.append(f"\n" + "="*50)
    report_lines.append(f"📦 BẢNG: {table_name}")
    report_lines.append("="*50)
    
    try:
        # 2. Lấy cấu trúc schema
        columns = con.execute(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}' AND table_schema = 'main'
            ORDER BY ordinal_position
        """).fetchall()
        
        col_names = [col[0].lower() for col in columns]
        
        report_lines.append("\n📌 CẤU TRÚC BẢNG:")
        for col in columns:
            report_lines.append(f"  ▪ {col[0]} ({col[1]})")

        # 3. Lấy tổng số dòng dữ liệu
        total_rows = con.execute(f'SELECT COUNT(*) FROM main."{table_name}"').fetchone()[0]
        report_lines.append(f"\n📊 TỔNG SỐ DÒNG DỮ LIỆU: {total_rows}")

        if total_rows == 0:
            report_lines.append("⚠️ Bảng không có dữ liệu, bỏ qua kiểm tra sâu.")
            continue

        # 4. Kiểm tra rủi ro trùng lặp đặc trưng (Multicollinearity)
        report_lines.append("\n⚠️ KIỂM TRA TRÙNG LẶP ĐẶC TRƯNG (ROE / ROA):")
        roe_cols = [c for c in col_names if 'roe' in c]
        roa_cols = [c for c in col_names if 'roa' in c]
        
        report_lines.append(f"  - Các cột chứa ROE: {roe_cols}")
        report_lines.append(f"  - Các cột chứa ROA: {roa_cols}")
        if len(roe_cols) > 1 or len(roa_cols) > 1:
            report_lines.append("  -> 🚨 CẢNH BÁO: Dữ liệu có nhiều định dạng ROE/ROA. Cần loại bỏ ở bước Machine Learning.")

        # 5. Đếm số lượng NULL cho TẤT CẢ các cột
        report_lines.append("\n📉 TỈ LỆ DỮ LIỆU RỖNG (NULL/NA):")
        for col in columns:
            col_name = col[0]
            null_count = con.execute(f'SELECT COUNT(*) FROM main."{table_name}" WHERE "{col_name}" IS NULL').fetchone()[0]
            if null_count > 0:
                null_pct = round((null_count / total_rows) * 100, 2)
                report_lines.append(f"  - {col_name}: Rỗng {null_count}/{total_rows} dòng ({null_pct}%)")

                # Cảnh báo riêng cho các biến trọng yếu
                if 'mcap' in col_name.lower() or 'market_cap' in col_name.lower():
                    report_lines.append("    -> 💡 Chú ý: Biến Vốn hóa có dữ liệu rỗng. Cần fallback trong Tab 1.")
                if 'net_income' in col_name.lower() or 'loi_nhuan' in col_name.lower():
                    report_lines.append("    -> 💡 Chú ý: Biến Lợi nhuận có dữ liệu rỗng. Cần xử lý kỹ cho mô hình SHAP và định giá.")
            else:
                report_lines.append(f"  - {col_name}: Full 100% dữ liệu")

    except Exception as e:
        report_lines.append(f"\n❌ Lỗi khi đọc bảng {table_name}: {str(e)}")

# 6. Lưu báo cáo
output_file = "motherduck_diagnostic_report.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\n✅ Đã quét xong! Toàn bộ báo cáo chẩn đoán đã được lưu vào: {output_file}")
print("👉 Bạn hãy mở file này, copy toàn bộ nội dung và dán vào đây để chúng ta bắt đầu vá lỗi app.py!")