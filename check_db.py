import os
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
md_token = os.getenv("MOTHERDUCK_TOKEN")

if not md_token:
    print("❌ Chưa tìm thấy MOTHERDUCK_TOKEN trong file .env!")
else:
    con = duckdb.connect(f"md:vietfin_db?token={md_token}")
    
    print("=" * 65)
    print("📊 BÁO CÁO KIỂM TRA DỮ LIỆU MOTHERDUCK (DATA AUDIT REPORT)")
    print("=" * 65)
    
    # 1. Thống kê tổng số dòng và mã cổ phiếu
    tables = ['dim_company', 'fact_financials', 'fact_ratio_summary']
    for tbl in tables:
        try:
            res = con.execute(f"SELECT COUNT(*) as total_rows, COUNT(DISTINCT ticker) as total_tickers FROM {tbl}").fetchone()
            print(f"🔹 Bảng `{tbl}`: {res[0]:,} dòng | {res[1]:,} mã cổ phiếu")
        except Exception as e:
            print(f"⚠️ Bảng `{tbl}` không tồn tại hoặc lỗi: {e}")

    # 2. Đối soát độ phủ mã cổ phiếu giữa các bảng
    print("\n🔍 ĐỐI SOÁT ĐỘ PHỦ MÃ CỔ PHIẾU GIỮA CÁC BẢNG")
    cross_check_query = """
        SELECT 
            COUNT(DISTINCT c.ticker) AS "Mã ở dim_company",
            COUNT(DISTINCT f.ticker) AS "Mã ở fact_financials",
            COUNT(DISTINCT r.ticker) AS "Mã ở fact_ratio_summary",
            COUNT(DISTINCT CASE WHEN f.ticker IS NULL THEN c.ticker END) AS "Mã thiếu BCTC",
            COUNT(DISTINCT CASE WHEN r.ticker IS NULL THEN c.ticker END) AS "Mã thiếu Ratios"
        FROM dim_company c
        LEFT JOIN (SELECT DISTINCT ticker FROM fact_financials) f ON c.ticker = f.ticker
        LEFT JOIN (SELECT DISTINCT ticker FROM fact_ratio_summary) r ON c.ticker = r.ticker
    """
    try:
        df_cross = con.execute(cross_check_query).df()
        print(df_cross.to_string(index=False))
    except Exception as e:
        print(f"⚠️ Lỗi đối soát: {e}")

    # 3. Kiểm tra chất lượng dữ liệu (Lấy tên cột an toàn qua Pandas)
    print("\n📉 KIỂM TRA CHẤT LƯỢNG DỮ LIỆU CÁC CỘT QUAN TRỌNG (fact_ratio_summary)")
    
    # Sửa lỗi tại đây: Dùng Pandas để lấy danh sách tên cột
    df_empty = con.execute("SELECT * FROM fact_ratio_summary LIMIT 0").df()
    cols = [str(c).lower() for c in df_empty.columns]
    
    roe_col = 'roe' if 'roe' in cols else ('roe_pct' if 'roe_pct' in cols else None)
    roa_col = 'roa' if 'roa' in cols else ('roa_pct' if 'roa_pct' in cols else None)
    rev_col = 'net_revenue' if 'net_revenue' in cols else ('revenue' if 'revenue' in cols else None)
    inc_col = 'net_income' if 'net_income' in cols else ('profit_after_tax' if 'profit_after_tax' in cols else None)

    select_exprs = ['COUNT(*) AS "Tổng số dòng"']
    if roe_col: select_exprs.append(f'ROUND(100.0 * SUM(CASE WHEN {roe_col} IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS "% Rỗng ROE"')
    if roa_col: select_exprs.append(f'ROUND(100.0 * SUM(CASE WHEN {roa_col} IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS "% Rỗng ROA"')
    if rev_col: select_exprs.append(f'ROUND(100.0 * SUM(CASE WHEN {rev_col} IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS "% Rỗng Doanh Thu"')
    if inc_col: select_exprs.append(f'ROUND(100.0 * SUM(CASE WHEN {inc_col} IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS "% Rỗng LNST"')

    quality_query = f"SELECT {', '.join(select_exprs)} FROM fact_ratio_summary"
    
    try:
        df_quality = con.execute(quality_query).df()
        print(df_quality.to_string(index=False))
    except Exception as e:
        print(f"⚠️ Lỗi kiểm tra chất lượng: {e}")
        
    print("=" * 65)