import os
import duckdb
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("MOTHERDUCK_TOKEN")

if not token:
    print("❌ Thiếu MOTHERDUCK_TOKEN trong file .env")
    exit(1)

print("🔌 Đang kết nối tới MotherDuck...")
con = duckdb.connect(f"md:vietfin_db?token={token}")

# =========================================================================
# BƯỚC 1: ĐỒNG BỘ CÁC BẢNG RAW TỪ TẦNG BRONZE LÊN MOTHERDUCK
# =========================================================================
datasets = ['overview', 'officers', 'shareholders', 'ratio_summary', 'price_history']
base_path = 'data/bronze/corporate'

print("\n-----------------------------------------------------------------")
print("STEP 1: Nạp dữ liệu thô từ Bronze Parquet lên MotherDuck")
print("-----------------------------------------------------------------")

for ds in datasets:
    parquet_path = f"{base_path}/{ds}/*.parquet" 
    table_name = f"gold_corporate_{ds}"
    
    try:
        print(f"🔄 Đang đồng bộ dataset [{ds}] -> bảng [{table_name}]...")
        # union_by_name=true giúp tự động gộp các cột từ cả VCI và KBS
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS 
            SELECT * FROM read_parquet('{parquet_path}', hive_partitioning=1, union_by_name=true)
        """)
        
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"✅ Thành công: {count:,} bản ghi trong {table_name}")
    except Exception as e:
        print(f"⚠️ Bỏ qua {ds} (Lỗi: {e})")


# =========================================================================
# BƯỚC 2: BÓC TÁCH VÀ CHUẨN HÓA MÃ NGÀNH / CỤM NGÀNH CHO BẢNG `dim_company`
# =========================================================================
print("\n-----------------------------------------------------------------")
print("STEP 2: Trích xuất & Chuẩn hóa Danh mục Doanh nghiệp & Phân ngành ICB")
print("-----------------------------------------------------------------")

try:
    print("🔄 Đang phân loại ngành nghề (ICB) và lọc mã trùng lặp...")
    
    # Kỹ thuật kiểm tra an toàn danh sách cột thực tế đang có trong Parquet
    cols_df = con.execute("DESCRIBE gold_corporate_overview").df()
    existing_cols = set(cols_df['column_name'].str.lower().tolist())
    
    def safe_col(col_name, default_val="NULL"):
        """Trả về tên cột nếu tồn tại trong parquet, ngược lại trả về giá trị mặc định"""
        return col_name if col_name.lower() in existing_cols else default_val

    # Truy vấn tạo/cập nhật bảng dim_company chuẩn hóa (Đã mapping tên cột chuẩn xác 100%)
    query_dim_company = f"""
        CREATE OR REPLACE TABLE dim_company AS
        WITH ranked_overview AS (
            SELECT *,
                   -- Khử trùng lặp: Chỉ lấy bản ghi cào mới nhất cho mỗi mã chứng khoán
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker 
                       ORDER BY TRY_CAST(retrieved_at AS TIMESTAMP) DESC NULLS LAST
                   ) as rn
            FROM gold_corporate_overview
        )
        SELECT 
            ticker,
            
            -- Tên công ty (Ưu tiên nguồn KBS -> VCI -> Ticker)
            COALESCE(
                {safe_col('organ_name_kbs')}, 
                {safe_col('organ_name_vci')}, 
                ticker
            ) AS company_name,
            
            -- Thông tin doanh nghiệp
            {safe_col('tax_id_kbs')} AS tax_id,
            {safe_col('address_kbs')} AS address,
            
            -- ============================================================
            -- THÔNG TIN LIÊN HỆ & VỐN HÓA (ĐÃ KHỚP VỚI CẤU TRÚC THỰC TẾ)
            -- ============================================================
            {safe_col('phone_kbs')} AS phone,
            {safe_col('email_kbs')} AS email,
            {safe_col('website_kbs')} AS website,
            {safe_col('ceo_name_kbs')} AS ceo,
            TRY_CAST(COALESCE({safe_col('market_cap_vci')}, '0') AS DOUBLE) AS market_cap,

            -- ============================================================
            -- THÔNG TIN NGÀNH VÀ CỤM NGÀNH CHUẨN ICB (VCI)
            -- ============================================================
            {safe_col('icb_code_vci')} AS icb_code,
            {safe_col('sector_vci')} AS sector_level1,
            {safe_col('icb_code_lv2_vci')} AS industry_level2,
            {safe_col('icb_code_lv4_vci')} AS detail_industry_level4,
            
            -- Sàn giao dịch (HOSE, HNX, UPCOM)
            COALESCE({safe_col('exchange_kbs')}, {safe_col('exchange')}) AS exchange,
            
            retrieved_at
        FROM ranked_overview
        WHERE rn = 1
        ORDER BY ticker;
    """
    
    con.execute(query_dim_company)
    
    company_count = con.execute("SELECT COUNT(*) FROM dim_company").fetchone()[0]
    sector_count = con.execute("SELECT COUNT(DISTINCT sector_level1) FROM dim_company WHERE sector_level1 IS NOT NULL").fetchone()[0]
    
    print(f"✅ Đã tạo thành công bảng 'dim_company' với {company_count:,} doanh nghiệp!")
    print(f"📊 Đã phân loại thành công {sector_count} cụm ngành lớn (ICB Level 1).")

except Exception as e:
    print(f"❌ Lỗi khi khởi tạo dim_company: {e}")

print("\n🎉 Hoàn tất đồng bộ và phân loại ngành nghề lên MotherDuck!")