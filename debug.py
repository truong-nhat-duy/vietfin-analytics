import streamlit as st
import pandas as pd
import duckdb
import os
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()
md_token = os.getenv("MOTHERDUCK_TOKEN")
if not md_token:
    try:
        md_token = st.secrets["MOTHERDUCK_TOKEN"]
    except:
        pass

st.set_page_config(page_title="Debug VietFin", layout="wide")
st.title("🛠️ TRANG KIỂM TRA DỮ LIỆU GỐC (DEBUG)")

if not md_token:
    st.error("Chưa cấu hình MOTHERDUCK_TOKEN")
    st.stop()

# Kết nối database
@st.cache_resource
def get_con():
    return duckdb.connect(f"md:vietfin_db?token={md_token}")

con = get_con()

# --- BƯỚC 1: KIỂM TRA BẢNG CHỈ SỐ (Chứa ROE, ROA, Margin) ---
st.header("1. Bảng `fact_ratio_summary`")
try:
    # Lấy thử 5 dòng của VNM
    df_ratio = con.execute("SELECT * FROM fact_ratio_summary WHERE ticker = 'VNM' ORDER BY year DESC, period DESC LIMIT 5").df()
    st.write("**Danh sách các cột hiện có trong DB:**")
    st.code(df_ratio.columns.tolist())
    st.write("**Dữ liệu mẫu:**")
    st.dataframe(df_ratio)
except Exception as e:
    st.error(f"Lỗi khi tải fact_ratio_summary: {e}")

# --- BƯỚC 2: KIỂM TRA BẢNG BÁO CÁO TÀI CHÍNH (Chứa Doanh thu, LNST, Tài sản) ---
st.header("2. Bảng `fact_financials`")
try:
    df_fin = con.execute("SELECT * FROM fact_financials WHERE ticker = 'VNM' LIMIT 5").df()
    st.write("**Danh sách các cột hiện có trong DB:**")
    st.code(df_fin.columns.tolist())
    st.write("**Dữ liệu mẫu:**")
    st.dataframe(df_fin)
except Exception as e:
    st.error(f"Lỗi khi tải fact_financials: {e}")

# --- BƯỚC 3: KIỂM TRA THỬ TỔNG QUÁT ---
st.header("3. Kiểm tra số lượng mã cổ phiếu có Gross Margin hợp lệ")
try:
    # Tìm xem có cột gross margin nào không, thay tên cột nếu của bạn khác
    possible_gm_cols = [c for c in con.execute("SELECT * FROM fact_ratio_summary LIMIT 1").df().columns if 'gross' in c.lower() or 'margin' in c.lower()]
    st.write(f"Các cột có vẻ liên quan đến Margin: {possible_gm_cols}")
    
    if possible_gm_cols:
        col_to_check = possible_gm_cols[0]
        query = f"SELECT COUNT(*) as valid_count FROM fact_ratio_summary WHERE {col_to_check} IS NOT NULL AND {col_to_check} != 0"
        res = con.execute(query).df()
        st.write(f"Số dòng có `{col_to_check}` khác 0 và khác NULL: **{res.iloc[0]['valid_count']}** dòng")
except Exception as e:
    st.error(f"Lỗi kiểm tra tổng quát: {e}")