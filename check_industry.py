import os
import duckdb
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("MOTHERDUCK_TOKEN")
con = duckdb.connect(f"md:vietfin_db?token={token}")

print("\n📊 THỐNG KÊ SỐ LƯỢNG DOANH NGHIỆP THEO CỤM NGÀNH (ICB LEVEL 1):")
df_sector = con.execute("""
    SELECT 
        COALESCE(sector_level1, 'Chưa phân ngành') AS cum_nganh,
        COUNT(ticker) AS so_luong_doanh_nghiep
    FROM dim_company
    GROUP BY sector_level1
    ORDER BY so_luong_doanh_nghiep DESC
""").df()

print(df_sector.to_string(index=False))

print("\n🔍 SOI THỬ 5 DOANH NGHIỆP ĐẦU TIÊN:")
df_sample = con.execute("""
    SELECT ticker, company_name, icb_code, sector_level1, industry_level2 
    FROM dim_company 
    LIMIT 5
""").df()

print(df_sample.to_string(index=False))