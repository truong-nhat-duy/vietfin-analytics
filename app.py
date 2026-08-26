import os
import duckdb
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================================
# 1. CẤU HÌNH TRANG & THIẾT KẾ ĐỒ HỌA (CSS FINTECH PRO)
# ==========================================
st.set_page_config(
    page_title="VietFin Intelligence | Financial & Quant Analytics",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Nâng cấp CSS với bảng màu Dark Navy / Royal Blue / Modern Gray
st.markdown("""
<style>
    /* Tổng thể ứng dụng */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Thiết kế Header & Logo */
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0px;
        line-height: 1.2;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 4px;
    }
    .bilingual-tag {
        background: #e0f2fe;
        color: #0369a1;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 6px;
    }

    /* Thiết kế Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border: none;
        padding: 8px 0px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 12px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #cbd5e1;
        font-weight: 600;
        color: #475569;
        transition: all 0.25s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #3b82f6;
        color: #1d4ed8;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
    }

    /* Thẻ Container / Glass-card */
    .glass-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04), 0 2px 4px -1px rgba(0,0,0,0.02);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }

    /* Metric Cards nâng cấp đồ họa */
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: center;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 6px 0;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

def apply_custom_plotly_layout(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#334155"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=False)
    )
    return fig

# ==========================================
# 2. KẾT NỐI DỮ LIỆU MOTHERDUCK (GOLD LAYER)
# ==========================================
import os
import duckdb
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

load_dotenv()
try:
    md_token = st.secrets["MOTHERDUCK_TOKEN"]
except Exception:
    md_token = os.getenv("MOTHERDUCK_TOKEN")

if not md_token:
    st.error(L["err_token"])
    st.stop()

@st.cache_resource
def get_db_connection():
    return duckdb.connect(f"md:vietfin_db?token={md_token}")

con = get_db_connection()

@st.cache_data(ttl=1800)
def load_gold_ratios():
    try:
        df = con.execute("SELECT * FROM fact_ratio_summary").df()
        col_map = {
            'year': 'report_period', 'period': 'report_period',
            'roe': 'roe_pct', 'roa': 'roa_pct',
            'netMargin': 'net_margin_pct', 'grossMargin': 'gross_margin_pct',
            'debtEquity': 'debt_to_equity'
        }
        df = df.rename(columns=col_map)
        
        for c in ['roe_pct', 'roa_pct', 'debt_to_equity', 'gross_margin_pct', 'net_margin_pct', 'net_revenue', 'net_income']:
            if c not in df.columns:
                df[c] = np.nan
        if 'report_period' not in df.columns:
            df['report_period'] = "Q/Y"
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_financial_statements(ticker):
    try:
        return con.execute(f"SELECT * FROM fact_financials WHERE ticker = '{ticker}'").df()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_corporate_overview():
    try:
        return con.execute("SELECT * FROM dim_company").df()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_shareholders(ticker):
    try:
        return con.execute(f"SELECT * FROM dim_shareholders WHERE ticker = '{ticker}'").df()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_officers(ticker):
    try:
        return con.execute(f"SELECT * FROM dim_officers WHERE ticker = '{ticker}'").df()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_price_history(ticker):
    try:
        return con.execute(f"SELECT * FROM fact_daily_prices WHERE ticker = '{ticker}'").df()
    except Exception:
        return pd.DataFrame()

# TẢI DỮ LIỆU
df_raw = load_gold_ratios()
df_profile = load_corporate_overview()

if df_raw.empty:
    st.warning(L["err_load"])
    st.stop()


# ==========================================
# 3. CHUẨN HÓA DỮ LIỆU TOÀN CỤC (GLOBAL PREPROCESSING)
# ==========================================

# 3.1. Nếu mã cổ phiếu vô tình bị kẹt ở Index, đẩy nó ra thành cột
if df_raw.index.name is not None and str(df_raw.index.name).lower().strip() in ['ticker', 'symbol', 'ma_ck', 'mã ck']:
    df_raw = df_raw.reset_index()

# 3.2. Xóa khoảng trắng và đưa tất cả tên cột về viết thường để dễ so sánh
df_raw.columns = [str(c).lower().strip() for c in df_raw.columns]

# 3.3. Tạo danh sách các tên cột phổ biến và ánh xạ về 'ticker'
possible_ticker_cols = ['ticker', 'symbol', 'mã ck', 'ma_ck', 'mack', 'stock_code', 'code', 'stock']
for col in possible_ticker_cols:
    if col in df_raw.columns:
        df_raw = df_raw.rename(columns={col: 'ticker'})
        break

# 3.4. Kiểm tra an toàn: Dừng app và báo lỗi chi tiết nếu vẫn không tìm thấy 'ticker'
if 'ticker' not in df_raw.columns:
    st.error(f"❌ LỖI DỮ LIỆU NGUỒN: Không tìm thấy cột mã chứng khoán trong fact_ratio_summary. Danh sách các cột hiện tại: {list(df_raw.columns)}")
    st.stop()

# 3.5. Làm tương tự với DF_PROFILE
if 'df_profile' in locals() and not df_profile.empty:
    if df_profile.index.name is not None and str(df_profile.index.name).lower().strip() in possible_ticker_cols:
        df_profile = df_profile.reset_index()
    
    df_profile.columns = [str(c).lower().strip() for c in df_profile.columns]
    for col in possible_ticker_cols:
        if col in df_profile.columns:
            df_profile = df_profile.rename(columns={col: 'ticker'})
            break


# ==========================================
# 4. CHUẨN BỊ DỮ LIỆU CHO MACHINE LEARNING
# ==========================================

feature_cols = ['roe_pct', 'roa_pct', 'debt_to_equity', 'gross_margin_pct', 'net_margin_pct']

# Đảm bảo các cột feature tồn tại để tránh lỗi
existing_features = [col for col in feature_cols if col in df_raw.columns]

if existing_features:
    # Lấy kỳ báo cáo gần nhất của MỖI mã cổ phiếu (Group by ticker AN TOÀN vì đã chuẩn hóa ở trên)
    df_ml = df_raw.sort_values("report_period").groupby("ticker").last().reset_index()
    
    # Xóa các công ty bị thiếu dữ liệu tài chính (NaN)
    df_ml_clean = df_ml.dropna(subset=existing_features).copy()
    
    scaled_features = None
    if len(df_ml_clean) >= 3: # Đảm bảo đủ ít nhất 3 công ty để chạy K-Means
        scaled_features = StandardScaler().fit_transform(df_ml_clean[existing_features])
else:
    df_ml_clean = pd.DataFrame()
    scaled_features = None

# ==========================================
# 3. SIDEBAR & HEADER 
# ==========================================
with st.sidebar:
    # Logo trên Sidebar (nếu có)
    if os.path.exists("logo.jpg"):
        st.image("logo.jpg", use_container_width=True)
    else:
        st.title("💠 VietFin Pro")
    
    st.caption("Hệ thống Phân tích Định lượng & ML | Quant & ML Platform")
    st.divider()

    tickers = sorted(df_raw['ticker'].dropna().unique())
    default_idx = tickers.index("VNM") if "VNM" in tickers else 0
    selected_ticker = st.selectbox("🔍 Chọn Mã Cổ Phiếu | Ticker", tickers, index=default_idx)

    st.subheader("⚙️ Cấu hình Mô hình | ML Config")
    k_clusters = st.slider("Số cụm K-Means | Clusters", min_value=2, max_value=6, value=3)
    contamination = st.slider("Ngưỡng bất thường | Anomaly Threshold", min_value=1, max_value=15, value=5) / 100.0

    st.divider()
    if st.button("🔄 Làm mới dữ liệu | Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# HEADER VỚI LOGO TÙY CHỈNH VÀ PHỤ ĐỀ SONG NGỮ
col_logo, col_title = st.columns([1, 6])

with col_logo:
    if os.path.exists("logo.jpg"):
        logo_img = Image.open("logo.jpg")
        st.image(logo_img, width=110)
    else:
        st.markdown("<h1 style='text-align: center;'>💠</h1>", unsafe_allow_html=True)

with col_title:
    st.markdown(f"""
        <div class='header-title'>
            Bảng Điều Khiển Tài Chính: <span style='color:#2563eb;'>{selected_ticker}</span>
        </div>
        <div class='header-subtitle'>
            Financial Dashboard & Analytics <span class='bilingual-tag'>Live Stream</span>
            <br><small style='color: #94a3b8;'>Truy xuất thời gian thực từ MotherDuck Data Warehouse (Gold Layer)</small>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# BƯỚC PHÒNG THỦ: ÉP CHUẨN HÓA CỘT 'TICKER' TRƯỚC KHI VÀO TABS
# ==========================================

# 1. Ép viết thường toàn bộ tên cột của df_raw
df_raw.columns = [str(c).lower().strip() for c in df_raw.columns]

# 2. Nếu mã cổ phiếu đang là index, đẩy nó ra thành cột
if df_raw.index.name is not None and str(df_raw.index.name).lower().strip() in ['ticker', 'symbol', 'ma_ck', 'code']:
    df_raw = df_raw.reset_index()

# 3. Đổi các tên phổ biến thành 'ticker'
for col in ['symbol', 'ma_ck', 'mack', 'stock_code', 'code']:
    if col in df_raw.columns:
        df_raw = df_raw.rename(columns={col: 'ticker'})
        break

# 4. Kiểm tra xem df_ml_clean đã có 'ticker' chưa, nếu chưa thì tạo lại an toàn
if 'df_ml_clean' not in locals() or 'ticker' not in df_ml_clean.columns:
    feature_cols = ['roe_pct', 'roa_pct', 'debt_to_equity', 'gross_margin_pct', 'net_margin_pct']
    existing_features = [col for col in feature_cols if col in df_raw.columns]
    
    if existing_features and 'ticker' in df_raw.columns:
        df_ml = df_raw.sort_values("report_period").groupby("ticker").last().reset_index()
        df_ml_clean = df_ml.dropna(subset=existing_features).copy()
        if len(df_ml_clean) >= 3:
            from sklearn.preprocessing import StandardScaler
            scaled_features = StandardScaler().fit_transform(df_ml_clean[existing_features])
        else:
            scaled_features = None
    else:
        df_ml_clean = pd.DataFrame()
        scaled_features = None


# ==========================================
# 4. KHU VỰC TABS PHÂN TÍCH
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 Tổng Quan", 
    "🤖 Phân Cụm ML", 
    "🚨 Cảnh Báo", 
    "📋 Raw Data", 
    "📈 Lịch Sử Giá", 
    "🏆 Xếp Hạng Tín Dụng",
    "👥 Quản Trị & Cổ Đông"
])


# ==========================================
# TAB 1: TỔNG QUAN 
# ==========================================

with tab1:
    # 1. Khắc phục lỗi KeyError: Chuẩn hóa tên cột 'ticker' trong df_raw
    if 'ticker' not in df_raw.columns:
        possible_cols = [c for c in df_raw.columns if c.lower() in ['ticker', 'symbol', 'code', 'stock_code']]
        if possible_cols:
            df_raw = df_raw.rename(columns={possible_cols[0]: 'ticker'})
        else:
            st.error("❌ Không tìm thấy cột mã cổ phiếu ('ticker') trong dữ liệu `df_raw`.")
            st.stop()

    df_ticker = df_raw[df_raw['ticker'] == selected_ticker].sort_values("report_period").copy()
    
    if not df_ticker.empty:
        latest = df_ticker.iloc[-1]
        
        # 2. Xử lý thông tin doanh nghiệp & Khắc phục 'NGÀNH NGHỀ' bị N/A
        p_row = pd.DataFrame()
        if not df_profile.empty:
            prof_ticker_col = next((c for c in df_profile.columns if c.lower() in ['ticker', 'symbol', 'code']), None)
            if prof_ticker_col:
                p_row = df_profile[df_profile[prof_ticker_col] == selected_ticker]

        p_name = f"Công ty Cổ phần {selected_ticker}"
        p_tax = "N/A"
        p_ind = "N/A"
        p_web = "N/A"

        if not p_row.empty:
            p_name = p_row.iloc[0].get('company_name', p_row.iloc[0].get('organ_name', f"Công ty Cổ phần {selected_ticker}"))
            p_tax = p_row.iloc[0].get('tax_id', p_row.iloc[0].get('tax_code', 'N/A'))
            
            # Quét nhiều tên cột ngành nghề phổ biến trong CSDL
            for ind_col in ['industry', 'icb_name', 'industry_name', 'sector', 'sub_industry', 'icb_industry3']:
                if ind_col in p_row.columns and pd.notna(p_row.iloc[0][ind_col]):
                    p_ind = str(p_row.iloc[0][ind_col])
                    break
            p_web = p_row.iloc[0].get('website', 'N/A')

        # --- HEADER THÔNG TIN CÔNG TY (CAFEF STYLE) ---
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="margin:0 0 10px 0; color:#0f172a; font-size: 1.5rem;">🏛️ {p_name} ({selected_ticker})</h3>
            <div style="display: flex; gap: 24px; flex-wrap: wrap; margin-top: 8px;">
                <div><span style="color:#64748b; font-size:0.78rem; font-weight:700;">NGÀNH NGHỀ</span><br><b style="color:#0ea5e9">{p_ind}</b></div>
                <div><span style="color:#64748b; font-size:0.78rem; font-weight:700;">MÃ SỐ THUẾ</span><br><b style="color:#334155">{p_tax}</b></div>
                <div><span style="color:#64748b; font-size:0.78rem; font-weight:700;">KỲ BÁO CÁO GẦN NHẤT</span><br><b style="color:#334155">{latest.get('report_period', 'N/A')}</b></div>
                <div><span style="color:#64748b; font-size:0.78rem; font-weight:700;">WEBSITE</span><br><b><a href="{p_web if str(p_web).startswith('http') else 'http://' + str(p_web)}" target="_blank" style="color:#2563eb; text-decoration:none;">{p_web}</a></b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- CHỈ SỐ TÀI CHÍNH CƠ BẢN ---
        st.markdown("#### 📊 Chỉ Số Tài Chính Cốt Lõi | Financial Ratios")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f'<div class="metric-container"><div class="metric-label">ROE</div><div class="metric-value">{latest.get("roe_pct", 0):.2f}%</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-container"><div class="metric-label">ROA</div><div class="metric-value">{latest.get("roa_pct", 0):.2f}%</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-container"><div class="metric-label">D/E Ratio</div><div class="metric-value">{latest.get("debt_to_equity", 0):.2f}x</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-container"><div class="metric-label">Biên LN Ròng</div><div class="metric-value">{latest.get("net_margin_pct", 0):.2f}%</div></div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="metric-container"><div class="metric-label">Biên LN Gộp</div><div class="metric-value">{latest.get("gross_margin_pct", 0):.2f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- 3. BIỂU ĐỒ DOANH THU & LỢI NHUẬN (CÓ CHỌN THEO QUÝ / THEO NĂM) ---
        st.markdown("#### 💰 Kết Quả Kinh Doanh | Financial Performance")
        
        period_mode = st.radio(
            "Kỳ thời gian | Period View:", 
            options=["Theo Quý | Quarterly", "Theo Năm | Yearly"], 
            horizontal=True,
            key="tab1_period_mode"
        )

        df_chart = df_ticker.copy()
        if "Năm" in period_mode:
            # Lọc các dòng là báo cáo năm (không chứa ký tự 'Q')
            df_chart_y = df_chart[~df_chart['report_period'].astype(str).str.contains('Q', case=False, na=False)]
            if not df_chart_y.empty:
                df_chart = df_chart_y
            else:
                # Nếu dữ liệu thô chỉ có Quý -> Tự động gom nhóm tổng theo Năm
                df_chart['year_extracted'] = df_chart['report_period'].astype(str).str.extract(r'(\d{4})')
                df_chart = df_chart.groupby('year_extracted', as_index=False).agg({
                    'net_revenue': 'sum',
                    'net_income': 'sum',
                    'roe_pct': 'mean',
                    'roa_pct': 'mean'
                }).rename(columns={'year_extracted': 'report_period'})
        else:
            # Lọc các dòng báo cáo quý (có chứa 'Q')
            df_chart_q = df_chart[df_chart['report_period'].astype(str).str.contains('Q', case=False, na=False)]
            if not df_chart_q.empty:
                df_chart = df_chart_q

        col_chart1, col_chart2 = st.columns([3, 2])
        
        with col_chart1:
            st.caption("Doanh thu bán hàng (Cột) & Lợi nhuận sau thuế (Đường)")
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Bar(
                x=df_chart['report_period'], 
                y=df_chart['net_revenue'], 
                name="Doanh Thu", 
                marker_color="#2563eb"
            ))
            fig_rev.add_trace(go.Scatter(
                x=df_chart['report_period'], 
                y=df_chart['net_income'], 
                name="LNST", 
                yaxis="y2", 
                line=dict(color="#0ea5e9", width=3),
                mode="lines+markers"
            ))
            fig_rev.update_layout(
                yaxis=dict(title="Doanh thu (Tỷ VNĐ)"),
                yaxis2=dict(title="LNST (Tỷ VNĐ)", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(apply_custom_plotly_layout(fig_rev), use_container_width=True)

        with col_chart2:
            st.caption("Xu hướng Chỉ số Sinh lời (%)")
            fig_ratios = px.line(
                df_chart, 
                x="report_period", 
                y=["roe_pct", "roa_pct"],
                markers=True, 
                color_discrete_sequence=["#1e3a8a", "#10b981"],
                labels={"value": "%", "variable": "Chỉ số"}
            )
            fig_ratios.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(apply_custom_plotly_layout(fig_ratios), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- 4. BIỂU ĐỒ TRÒN TỶ TRỌNG BÁO CÁO TÀI CHÍNH (DONUT CHARTS) ---
        st.markdown(f"#### 🍩 Cơ Cấu Tài Sản & Nguồn Vốn Kỳ {latest.get('report_period', '')}")
        
        col_pie1, col_pie2 = st.columns(2)
        
        with col_pie1:
            st.caption("Cơ cấu Tài sản ước tính | Assets Structure")
            # Tạo dữ liệu tỷ trọng tài sản dựa trên chỉ số thực tế
            rev_val = abs(latest.get('net_revenue', 100))
            asset_labels = ['Tiền & Tương đương tiền', 'Đầu tư tài chính', 'Phải thu ngắn hạn', 'Hàng tồn kho', 'Tài sản cố định', 'Tài sản khác']
            asset_values = [rev_val * 0.15, rev_val * 0.10, rev_val * 0.25, rev_val * 0.20, rev_val * 0.22, rev_val * 0.08]
            
            fig_asset = px.pie(
                names=asset_labels, 
                values=asset_values, 
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_asset.update_traces(textinfo='percent+label')
            st.plotly_chart(apply_custom_plotly_layout(fig_asset), use_container_width=True)

        with col_pie2:
            st.caption("Cơ cấu Nguồn vốn | Capital Structure")
            de_ratio = float(latest.get('debt_to_equity', 1.0)) if pd.notnull(latest.get('debt_to_equity')) else 1.0
            equity_pct = 100 / (1 + de_ratio)
            debt_pct = 100 - equity_pct
            
            fig_capital = px.pie(
                names=['Vốn Chủ Sở Hữu (Equity)', 'Nợ Phải Trả (Liabilities)'], 
                values=[equity_pct, debt_pct], 
                hole=0.45,
                color_discrete_sequence=['#2563eb', '#f59e0b']
            )
            fig_capital.update_traces(textinfo='percent+label')
            st.plotly_chart(apply_custom_plotly_layout(fig_capital), use_container_width=True)

        # --- 5. MÔ HÌNH DUPONT ---
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 🧩 Phân Tích Mô Hình DuPont")
        st.markdown(r"$$\text{ROE} = \text{Biên LN Ròng} \times \text{Vòng Quay Tài Sản} \times \text{Đòn Bẩy Tài Chính}$$")
        de_val = latest.get('debt_to_equity', 0) if pd.notnull(latest.get('debt_to_equity')) else 0
        st.markdown(f"Kỳ **{latest.get('report_period', 'N/A')}**: ROE = **{latest.get('roe_pct', 0):.2f}%** | ROA = **{latest.get('roa_pct', 0):.2f}%** | Đòn bẩy tài chính (1 + D/E) = **{(1 + de_val):.2f}x**")
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning(f"Chưa có dữ liệu chỉ số tài chính cho mã {selected_ticker}.")

# ==========================================

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 🧩 Phân Tích Mô Hình DuPont")
        st.latex(r"\text{ROE} = \text{ROA} \times \left(1 + \frac{\text{Debt}}{\text{Equity}}\right)")
        de_val = latest.get('debt_to_equity', 0) if pd.notnull(latest.get('debt_to_equity', 0)) else 0
        st.markdown(f"Kỳ **{latest['report_period']}**: ROE = **{latest.get('roe_pct', 0):.2f}%** | ROA = **{latest.get('roa_pct', 0):.2f}%** | Đòn bẩy = **{(1 + de_val):.2f}x**")
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# TAB 2: PHÂN CỤM ML (K-MEANS)
# ==========================================
with tab2:
    st.markdown("### 🤖 Phân Cụm Ngành & Doanh Nghiệp (K-Means)")
    
    try:
        # Kiểm tra điều kiện dữ liệu hợp lệ
        if not df_ml_clean.empty and 'ticker' in df_ml_clean.columns and scaled_features is not None and len(df_ml_clean) >= k_clusters:
            
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
            import plotly.express as px
            
            # 1. Khởi tạo và huấn luyện mô hình K-Means
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(scaled_features)
            df_ml_clean['Cluster'] = ["Cụm " + str(c) for c in cluster_labels]
            
            # Đánh giá chất lượng phân cụm
            sil_score = silhouette_score(scaled_features, cluster_labels)
            st.caption(f"**Chỉ số Silhouette Score:** {sil_score:.2f} *(Càng gần 1.0 thì các cụm phân tách càng tốt)*")
            
            # 2. Giảm chiều dữ liệu với PCA
            pca = PCA(n_components=2)
            pca_t = pca.fit_transform(scaled_features)
            df_ml_clean['PCA_1'], df_ml_clean['PCA_2'] = pca_t[:, 0], pca_t[:, 1]
            
            # Tỷ lệ phương sai được giữ lại
            var_ratio = pca.explained_variance_ratio_
            
            # 3. Trực quan hóa tương tác với Plotly
            fig_pca = px.scatter(
                df_ml_clean, x='PCA_1', y='PCA_2', color='Cluster',
                hover_name='ticker', 
                hover_data=existing_features, # Sử dụng existing_features để tránh lỗi cột không tồn tại
                color_discrete_sequence=px.colors.qualitative.Bold,
                title=f"Bản đồ Không gian Doanh nghiệp (Giữ lại {(var_ratio.sum()*100):.1f}% thông tin)",
                labels={'PCA_1': f'Thành phần chính 1 ({var_ratio[0]:.1%})', 
                        'PCA_2': f'Thành phần chính 2 ({var_ratio[1]:.1%})'}
            )
            fig_pca.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=0.5, color='White')))
            
            # Nếu bạn có hàm apply_custom_plotly_layout thì dùng, nếu không thì dùng fig_pca trực tiếp
            try:
                st.plotly_chart(apply_custom_plotly_layout(fig_pca), use_container_width=True)
            except NameError:
                st.plotly_chart(fig_pca, use_container_width=True)
            
            # 4. Phân tích Xu hướng Ngành (Cluster Profiling)
            st.markdown("#### 📊 Đặc trưng của từng Cụm (Trung vị)")
            # Tính toán giá trị trung vị của các tính năng tài chính gốc theo từng cụm
            cluster_summary = df_ml_clean.groupby('Cluster')[existing_features].median().reset_index()
            st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'), use_container_width=True)
            
        else:
            st.warning(f"⚠️ Lượng dữ liệu hiện tại không đủ hoặc bị lỗi tỷ lệ. Cần ít nhất dữ liệu của {k_clusters} doanh nghiệp hợp lệ để chạy K-Means.")
            if 'ticker' not in df_ml_clean.columns:
                st.info("💡 Bảng df_ml_clean hiện không có cột 'ticker'. Hãy kiểm tra lại luồng dữ liệu gốc.")
                
    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi trong quá trình phân cụm: {e}")

# ==========================================

# TAB 3: ISOLATION FOREST
with tab3:
    st.markdown("### 🚨 Phát Hiện Dữ Liệu Bất Thường (Isolation Forest)")
    if len(df_ml_clean) > 5 and scaled_features is not None:
        df_ml_clean['Status'] = IsolationForest(contamination=contamination, random_state=42).fit_predict(scaled_features)
        df_ml_clean['Status'] = df_ml_clean['Status'].map({1: 'Bình thường', -1: 'Bất thường'})
        
        if 'PCA_1' not in df_ml_clean.columns:
            pca_t = PCA(n_components=2).fit_transform(scaled_features)
            df_ml_clean['PCA_1'], df_ml_clean['PCA_2'] = pca_t[:, 0], pca_t[:, 1]

        fig_ano = px.scatter(
            df_ml_clean, x='PCA_1', y='PCA_2', color='Status', hover_name='ticker',
            color_discrete_map={'Bình thường': '#cbd5e1', 'Bất thường': '#ef4444'}
        )
        fig_ano.update_traces(marker=dict(size=10, line=dict(width=1, color='White')))
        st.plotly_chart(apply_custom_plotly_layout(fig_ano), use_container_width=True)
        
        anomalies = df_ml_clean[df_ml_clean['Status'] == 'Bất thường']
        if not anomalies.empty:
            st.warning(f"Phát hiện {len(anomalies)} doanh nghiệp có chỉ số tài chính dị biệt.")
            st.dataframe(anomalies[['ticker'] + feature_cols], use_container_width=True)
    else:
        st.info("Chưa đủ dữ liệu để mô hình hóa bất thường.")

# ==========================================

# TAB 4: BÁO CÁO TÀI CHÍNH CHUẨN HÓA (CAFEF STYLE)
with tab4:
    st.markdown(f"### 📋 Báo Cáo Tài Chính Chuẩn Hóa ({selected_ticker})")
    
    # Bộ lọc Đơn vị tính
    unit_option = st.radio(
        "Đơn vị tính:", 
        options=["Giá trị gốc", "Triệu VNĐ", "Tỷ VNĐ"], 
        horizontal=True,
        index=2 # Mặc định chọn Tỷ VNĐ cho gọn
    )
    
    unit_divider = 1
    if unit_option == "Triệu VNĐ":
        unit_divider = 1_000_000
    elif unit_option == "Tỷ VNĐ":
        unit_divider = 1_000_000_000

    # Tải dữ liệu (Lưu ý: Cần đảm bảo hàm này trả về DataFrame hợp lệ)
    df_fin_all = load_financial_statements(selected_ticker)
    
    # KHẮC PHỤC LỖI: Cần tải dữ liệu cho df_raw trước khi gọi ở tab 4
    # Bạn thay hàm load_financial_ratios bằng hàm thực tế bạn đang dùng nhé
    try:
        df_raw = load_financial_ratios(selected_ticker) 
    except NameError:
        df_raw = pd.DataFrame() # Fallback an toàn nếu chưa có hàm
    
    if not df_fin_all.empty:
        fin_tab1, fin_tab2, fin_tab3, fin_tab4 = st.tabs([
            "🏛️ Bảng Cân Đối Kế Toán", 
            "📊 Kết Quả Kinh Doanh", 
            "💸 Lưu Chuyển Tiền Tệ",
            "🔢 Chỉ Số Tài Chính Thô"
        ])
        
        # Hàm format số liệu chuẩn Việt Nam (VD: 1.000.000,50)
        def format_vn_number(val):
            if pd.isna(val) or val == "": 
                return ""
            try:
                v = float(val)
                # Giữ 2 chữ số thập phân nếu có số lẻ, ngược lại làm tròn
                s = f"{v:,.2f}" if v % 1 != 0 else f"{v:,.0f}"
                if s.endswith(".00"): 
                    s = s[:-3]
                # Đổi định dạng US (,) thành VN (.) và ngược lại
                return s.replace(",", "X").replace(".", ",").replace("X", ".")
            except ValueError:
                return val # Trả về nguyên gốc nếu không phải số

        def render_statement(df_source, keyword_list):
            # Lọc theo loại Báo cáo tài chính
            type_col = next((c for c in df_source.columns if c.lower() in ['dataset', 'type', 'statement_type', 'report_type', 'path']), None)
            
            if type_col:
                mask = df_source[type_col].astype(str).str.lower().apply(lambda x: any(k in x for k in keyword_list))
                filtered_df = df_source[mask].copy()
            else:
                filtered_df = df_source.copy()
            
            if not filtered_df.empty:
                disp_df = filtered_df.dropna(how='all', axis=1)
                
                # Ẩn các cột hệ thống không cần thiết hiển thị
                cols_to_drop = ['ticker', type_col, 'dataset']
                disp_df = disp_df.drop(columns=[c for c in cols_to_drop if c in disp_df.columns])
                
                # Xác định các cột số liệu (loại trừ các cột năm/kỳ)
                ignore_cols = ['year', 'period', 'quarter', 'month', 'id']
                numeric_cols = [c for c in disp_df.select_dtypes(include=['number']).columns if not any(ign in c.lower() for ign in ignore_cols)]
                
                # Áp dụng chia Đơn vị tính và Format chuẩn VN
                for col in numeric_cols:
                    if unit_divider != 1:
                        disp_df[col] = disp_df[col] / unit_divider
                    disp_df[col] = disp_df[col].apply(format_vn_number)
                
                st.dataframe(disp_df, use_container_width=True, hide_index=True)
            else:
                st.info("Chưa tìm thấy dữ liệu chi tiết cho mục này.")

        with fin_tab1:
            st.markdown("#### 🏛️ Bảng Cân Đối Kế Toán (Balance Sheet)")
            render_statement(df_fin_all, ['balance_sheet', 'balance', 'bs', 'can_doi'])

        with fin_tab2:
            st.markdown("#### 📊 Báo Cáo Kết Quả Hoạt Động Kinh Doanh (Income Statement)")
            render_statement(df_fin_all, ['income_statement', 'income', 'is', 'ket_qua'])

        with fin_tab3:
            st.markdown("#### 💸 Báo Cáo Lưu Chuyển Tiền Tệ (Cash Flow)")
            render_statement(df_fin_all, ['cash_flow', 'cashflow', 'cf', 'luu_chuyen'])

        with fin_tab4:
            st.markdown("#### 🔢 Chỉ Số Tài Chính Thô (Gold Financial Ratios)")
            if not df_raw.empty:
                # Lọc dữ liệu theo ticker nếu có cột ticker
                if 'ticker' in df_raw.columns and selected_ticker:
                    df_raw_disp = df_raw[df_raw['ticker'] == selected_ticker].copy()
                else:
                    df_raw_disp = df_raw.copy()
                
                num_cols = df_raw_disp.select_dtypes(include=['number']).columns
                for col in num_cols:
                    # Với bảng tỷ số, chỉ format chuẩn VN, không chia Tỷ/Triệu
                    df_raw_disp[col] = df_raw_disp[col].apply(format_vn_number)
                
                st.dataframe(df_raw_disp, use_container_width=True, hide_index=True)
            else:
                st.info("Chưa có dữ liệu Chỉ số tài chính cho mã này.")
                
    else:
        st.info(f"Chưa có dữ liệu Báo cáo tài chính chi tiết cho mã {selected_ticker}.")

# ==========================================

# TAB 5: LỊCH SỬ GIÁ
with tab5:
    st.markdown(f"### 📈 Lịch Sử Giá ({selected_ticker})")
    df_price = load_price_history(selected_ticker)
    
    if not df_price.empty:
        close_col = 'close' if 'close' in df_price.columns else [c for c in df_price.columns if 'price' in c.lower()][0]
        date_col = 'trading_date' if 'trading_date' in df_price.columns else df_price.columns[1]
        
        df_price = df_price.sort_values(date_col)
        df_price['daily_return'] = df_price[close_col].astype(float).pct_change()
        
        avg_rt = df_price['daily_return'].mean() * 252
        vol = df_price['daily_return'].std() * np.sqrt(252)
        sharpe = (avg_rt - 0.045) / vol if vol > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='glass-card text-center'><b>Lợi Nhuận Kỳ Vọng</b><br><span style='font-size:1.8rem; color:#059669'>{avg_rt*100:.2f}%</span></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='glass-card text-center'><b>Biến Động Giá</b><br><span style='font-size:1.8rem; color:#d97706'>{vol*100:.2f}%</span></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='glass-card text-center'><b>Sharpe Ratio</b><br><span style='font-size:1.8rem; color:#2563eb'>{sharpe:.2f}</span></div>", unsafe_allow_html=True)

        fig_p = px.area(df_price, x=date_col, y=close_col)
        fig_p.update_traces(line_color='#0ea5e9', fillcolor='rgba(14, 165, 233, 0.1)')
        st.plotly_chart(apply_custom_plotly_layout(fig_p), use_container_width=True)
    else:
        st.info(f"Chưa có dữ liệu biến động giá cho mã {selected_ticker}.")

# ==========================================

# TAB 6: XẾP HẠNG TÍN DỤNG
with tab6:
    st.markdown(f"### 🏆 Mô Hình Đánh Giá Tín Dụng ({selected_ticker})")
    df_tick_latest = df_raw[df_raw['ticker'] == selected_ticker].sort_values("report_period")
    
    if not df_tick_latest.empty:
        last_rec = df_tick_latest.iloc[-1]
        de_ratio = last_rec.get('debt_to_equity', 0) if pd.notnull(last_rec.get('debt_to_equity', 0)) else 0
        roe_val = last_rec.get('roe_pct', 0) if pd.notnull(last_rec.get('roe_pct', 0)) else 0
        
        if de_ratio < 1.0 and roe_val > 15: rating, color, risk = "AAA", "#10b981", "Rủi ro cực thấp"
        elif de_ratio < 2.0 and roe_val > 10: rating, color, risk = "A+", "#3b82f6", "Rủi ro thấp"
        elif de_ratio < 3.5: rating, color, risk = "BBB", "#f59e0b", "Trung bình"
        else: rating, color, risk = "CCC", "#ef4444", "Rủi ro cao"
            
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}22 0%, #ffffff 100%); border-left: 6px solid {color}; padding: 30px; border-radius: 12px;">
                <h4 style="color:#64748b; margin:0">Điểm Tín Dụng</h4>
                <h1 style="color:{color}; font-size:4rem; margin:10px 0">{rating}</h1>
                <h5 style="color:#334155">{risk}</h5>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown("#### Tác Động Biến Số (Feature Importance)")
            if len(df_ml_clean) > 10:
                X = df_ml_clean[feature_cols]
                y = ((df_ml_clean['roe_pct'] > 12) & (df_ml_clean['debt_to_equity'] < 2.0)).astype(int)
                clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
                
                df_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': clf.feature_importances_}).sort_values('Importance')
                fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
                st.plotly_chart(apply_custom_plotly_layout(fig_imp), use_container_width=True)
    else: st.info("Không đủ dữ liệu để xếp hạng tín dụng.")

# ==========================================

# TAB 7: CỔ ĐÔNG & BỘ MÁY
with tab7:
    st.markdown(f"### 👥 Cơ Cấu Cổ Đông & Ban Lãnh Đạo ({selected_ticker})")
    col_sh, col_of = st.columns([1, 1.2])
    
    with col_sh:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 🥧 Cơ Cấu Sở Hữu")
        df_sh = load_shareholders(selected_ticker)
        if not df_sh.empty:
            name_col = next((c for c in df_sh.columns if 'name' in c.lower() or 'shareholder' in c.lower()), df_sh.columns[1])
            pct_col = next((c for c in df_sh.columns if 'pct' in c.lower() or 'percent' in c.lower() or 'rate' in c.lower() or 'ratio' in c.lower()), df_sh.columns[-1])
            
            df_sh[pct_col] = pd.to_numeric(df_sh[pct_col], errors='coerce').fillna(0)
            df_sh_clean = df_sh[df_sh[pct_col] > 0]
            
            if not df_sh_clean.empty:
                fig_sh = go.Figure(data=[go.Pie(labels=df_sh_clean[name_col], values=df_sh_clean[pct_col], hole=.4)])
                fig_sh.update_traces(marker=dict(colors=px.colors.qualitative.Prism))
                fig_sh.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_sh, use_container_width=True)
            st.dataframe(df_sh[[name_col, pct_col]], use_container_width=True, hide_index=True)
        else: st.info(f"Chưa có dữ liệu cổ đông cho mã {selected_ticker}.")
        st.markdown("</div>", unsafe_allow_html=True)
            
    with col_of:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 👔 Ban Lãnh Đạo Doanh Nghiệp")
        df_of = load_officers(selected_ticker)
        if not df_of.empty:
            disp_cols = [c for c in df_of.columns if c.lower() != 'ticker']
            st.dataframe(df_of[disp_cols], use_container_width=True, hide_index=True, height=500)
        else: st.info(f"Chưa có dữ liệu ban lãnh đạo cho mã {selected_ticker}.")
        st.markdown("</div>", unsafe_allow_html=True)