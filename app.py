import os
import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title="VietFin Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 VietFin Analytics — Financial Intelligence Dashboard")
st.caption("Nền tảng phân tích chỉ số tài chính doanh nghiệp niêm yết (HOSE, HNX, UPCOM)")

load_dotenv()
token = os.getenv("MOTHERDUCK_TOKEN")

if not token:
    st.error("❌ Không tìm thấy MOTHERDUCK_TOKEN trong file .env!")
    st.stop()

@st.cache_resource
def get_connection():
    return duckdb.connect(f"md:vietfin_db?token={token}")

con = get_connection()

@st.cache_data(ttl=3600)
def load_tickers():
    df_tickers = con.execute("SELECT DISTINCT ticker FROM gold_financial_ratios ORDER BY ticker").fetchall()
    return [r[0] for r in df_tickers]

tickers = load_tickers()

# Sidebar setup
st.sidebar.image("logo.jpg", width=120)
st.sidebar.header("🔍 Bộ lọc Tìm kiếm")
selected_ticker = st.sidebar.selectbox("Chọn mã cổ phiếu", tickers, index=tickers.index("VNM") if "VNM" in tickers else 0)

if selected_ticker:
    df = con.execute("""
        SELECT 
            report_period, 
            net_revenue, 
            net_income, 
            roe_pct, 
            roa_pct, 
            debt_to_equity, 
            gross_margin_pct,
            net_margin_pct
        FROM gold_financial_ratios
        WHERE ticker = ?
        ORDER BY report_period ASC
    """, [selected_ticker]).df()

    st.subheader(f"Doanh nghiệp: {selected_ticker}")

    if not df.empty:
        latest = df.iloc[-1]
        
        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROE (%)", f"{latest['roe_pct']:.2f}%" if pd.notnull(latest['roe_pct']) else "N/A")
        c2.metric("ROA (%)", f"{latest['roa_pct']:.2f}%" if pd.notnull(latest['roa_pct']) else "N/A")
        c3.metric("D/E (Nợ / VCSH)", f"{latest['debt_to_equity']:.2f}" if pd.notnull(latest['debt_to_equity']) else "N/A")
        c4.metric("Biên LN Rồng (%)", f"{latest['net_margin_pct']:.2f}%" if pd.notnull(latest['net_margin_pct']) else "N/A")

        st.divider()

        # Visual Tabs
        tab1, tab2, tab3 = st.tabs(["📉 Chỉ số Sinh lời (ROE/ROA)", "💰 Doanh thu & Lợi nhuận", "📋 Bảng Dữ liệu Chi tiết"])

        with tab1:
            st.markdown("#### Biến động ROE & ROA qua các kỳ báo cáo")
            st.line_chart(df.set_index("report_period")[["roe_pct", "roa_pct"]])

        with tab2:
            st.markdown("#### Quy mô Doanh thu & Lợi nhuận sau thuế")
            st.bar_chart(df.set_index("report_period")[["net_revenue", "net_income"]])

        with tab3:
            st.markdown("#### Dữ liệu Tài chính Tầng Gold")
            st.dataframe(df, width="stretch")
    else:
        st.warning("Chưa có dữ liệu tài chính cho mã cổ phiếu này.")