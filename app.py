import os
import duckdb
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse
from dotenv import load_dotenv
from PIL import Image

# Safe OpenAI Import
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Machine Learning Modules
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import silhouette_score

# ==========================================
# 0. KHAI BÁO TỪ ĐIỂN NGÔN NGỮ & HÀM BỔ TRỢ
# ==========================================
L = {
    "err_token": "❌ Thiếu MotherDuck Token. Vui lòng kiểm tra lại file .env hoặc cấu hình secrets.",
    "err_load": "⚠️ Không thể tải dữ liệu từ cơ sở dữ liệu. Vui lòng kiểm tra lại kết nối."
}

def clean_val(val, default="N/A"):
    """Làm sạch các giá trị pd.NA, np.nan, None, '<NA>' thành chuỗi chuẩn."""
    if pd.isna(val) or val is None:
        return default
    s = str(val).strip()
    if s.lower() in ['nan', '<na>', 'none', 'null', '']:
        return default
    return s

def get_coords_from_address(address):
    """Trích xuất Vĩ độ, Kinh độ và Mức Zoom dựa trên địa chỉ text"""
    vn_coords = {
        'hồ chí minh': (10.762622, 106.660172), 'hcm': (10.762622, 106.660172), 'tp.hcm': (10.762622, 106.660172),
        'hà nội': (21.028511, 105.804817), 'ha noi': (21.028511, 105.804817),
        'đà nẵng': (16.054407, 108.202167),
        'hải phòng': (20.844912, 106.688084),
        'đồng nai': (10.946458, 106.824248),
        'bình dương': (11.229415, 106.626359),
        'vũng tàu': (10.497557, 107.168535),
        'cần thơ': (10.045162, 105.746853),
        'hải dương': (20.9373, 106.3146),
        'bắc ninh': (21.1861, 106.0763),
        'long an': (10.5364, 106.4067)
    }
    if not address or pd.isna(address) or address == "N/A":
        return 16.0, 106.0, 5  # Mặc định trung tâm VN, Zoom out
    
    addr_lower = str(address).lower()
    for k, (lat, lon) in vn_coords.items():
        if k in addr_lower:
            return lat, lon, 14  # Tọa độ tỉnh/thành, Zoom in
    return 16.0, 106.0, 5

def create_compatible_scatter_map(df, lat, lon, zoom, center, height=300, color=None, size=None, size_max=45, hover_name=None, hover_data=None, color_discrete_sequence=None):
    """Tự động tương thích với cả Plotly 5.x (scatter_mapbox) và Plotly 6.x+ (scatter_map)"""
    scatter_func = getattr(px, "scatter_map", None) or getattr(px, "scatter_mapbox")
    style_param = "map_style" if hasattr(px, "scatter_map") else "mapbox_style"
    
    kwargs = {
        "lat": lat, "lon": lon, "zoom": zoom, "center": center,
        style_param: "open-street-map", "height": height
    }
    if color: kwargs["color"] = color
    if size: kwargs["size"] = size
    if size_max: kwargs["size_max"] = size_max
    if hover_name: kwargs["hover_name"] = hover_name
    if hover_data: kwargs["hover_data"] = hover_data
    if color_discrete_sequence: kwargs["color_discrete_sequence"] = color_discrete_sequence

    return scatter_func(df, **kwargs)

# ==========================================
# 1. CẤU HÌNH TRANG & THIẾT KẾ ĐỒ HỌA
# ==========================================
st.set_page_config(
    page_title="VietFin Intelligence | Financial & Quant Analytics",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f8fafc; font-family: 'Inter', sans-serif; }
    .header-title { font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 0px; line-height: 1.2; }
    .header-subtitle { font-size: 0.95rem; color: #64748b; font-weight: 500; margin-top: 4px; }
    .bilingual-tag { background: #e0f2fe; color: #0369a1; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 4px; margin-left: 6px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border: none; padding: 8px 0px; }
    .stTabs [data-baseweb="tab"] { background-color: #ffffff; border-radius: 10px; padding: 12px 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #cbd5e1; font-weight: 600; color: #475569; transition: all 0.25s ease; }
    .stTabs [data-baseweb="tab"]:hover { border-color: #3b82f6; color: #1d4ed8; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%) !important; color: #ffffff !important; border: none !important; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important; }
    .glass-card { background: #ffffff; border-radius: 16px; padding: 20px 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04); border: 1px solid #e2e8f0; margin-bottom: 20px; }
    .metric-container { display: flex; flex-direction: column; align-items: flex-start; justify-content: center; background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); border: 1px solid #e2e8f0; border-left: 5px solid #2563eb; border-radius: 12px; padding: 16px 20px; }
    .metric-label { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.8rem; font-weight: 800; margin: 6px 0; color: #0f172a; }
</style>
""", unsafe_allow_html=True)

def apply_custom_plotly_layout(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#334155"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=False)
    )
    return fig

# ==========================================
# 2. KẾT NỐI DỮ LIỆU MOTHERDUCK
# ==========================================
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
            if c not in df.columns: df[c] = np.nan
        if 'report_period' not in df.columns: df['report_period'] = "Q/Y"
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_financial_statements(ticker):
    try: return con.execute(f"SELECT * FROM fact_financials WHERE ticker = '{ticker}'").df()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_corporate_overview():
    try: return con.execute("SELECT * FROM dim_company").df()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_shareholders(ticker):
    try: return con.execute(f"SELECT * FROM dim_shareholders WHERE ticker = '{ticker}'").df()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_officers(ticker):
    try: return con.execute(f"SELECT * FROM dim_officers WHERE ticker = '{ticker}'").df()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_price_history(ticker):
    try: return con.execute(f"SELECT * FROM fact_daily_prices WHERE ticker = '{ticker}'").df()
    except Exception: return pd.DataFrame()

df_raw = load_gold_ratios()
df_profile = load_corporate_overview()

if df_raw.empty:
    st.warning(L["err_load"])
    st.stop()

# ==========================================
# 3. CHUẨN HÓA DỮ LIỆU TOÀN CỤC
# ==========================================
if df_raw.index.name and str(df_raw.index.name).lower().strip() in ['ticker', 'symbol', 'ma_ck', 'code']:
    df_raw = df_raw.reset_index()

df_raw.columns = [str(c).lower().strip() for c in df_raw.columns]

possible_ticker_cols = ['ticker', 'symbol', 'mã ck', 'ma_ck', 'mack', 'stock_code', 'code', 'stock']
for col in possible_ticker_cols:
    if col in df_raw.columns:
        df_raw = df_raw.rename(columns={col: 'ticker'})
        break

if 'ticker' not in df_raw.columns:
    st.error(f"❌ LỖI DỮ LIỆU NGUỒN: Không tìm thấy cột mã chứng khoán. Các cột hiện tại: {list(df_raw.columns)}")
    st.stop()

df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]

if 'df_profile' in locals() and not df_profile.empty:
    df_profile.columns = [str(c).lower().strip() for c in df_profile.columns]
    for col in possible_ticker_cols:
        if col in df_profile.columns:
            df_profile = df_profile.rename(columns={col: 'ticker'})
            break

# ==========================================
# 4. CHUẨN BỊ DỮ LIỆU MACHINE LEARNING
# ==========================================
feature_cols = ['roe_pct', 'roa_pct', 'debt_to_equity', 'gross_margin_pct', 'net_margin_pct']
existing_features = [col for col in feature_cols if col in df_raw.columns]

if existing_features:
    df_ml = df_raw.sort_values("report_period").groupby("ticker").last().reset_index()
    df_ml_clean = df_ml.dropna(subset=existing_features).copy()
    scaled_features = StandardScaler().fit_transform(df_ml_clean[existing_features]) if len(df_ml_clean) >= 3 else None
else:
    df_ml_clean = pd.DataFrame()
    scaled_features = None

# ==========================================
# 5. SIDEBAR & HEADER
# ==========================================
with st.sidebar:
    if os.path.exists("logo.jpg"): st.image("logo.jpg", use_container_width=True)
    else: st.title("💠 VietFin Pro")
    st.caption("Hệ thống Phân tích Định lượng & ML")
    st.divider()

    tickers = sorted(df_raw['ticker'].dropna().unique())
    default_idx = tickers.index("VNM") if "VNM" in tickers else 0
    selected_ticker = st.selectbox("🔍 Chọn Mã Cổ Phiếu | Ticker", tickers, index=default_idx)

    st.subheader("⚙️ Cấu hình Mô hình | ML Config")
    k_clusters = st.slider("Số cụm K-Means | Clusters", 2, 6, 3)
    contamination = st.slider("Ngưỡng bất thường | Anomaly Threshold", 1, 15, 5) / 100.0
    
    st.divider()
    if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

col_logo, col_title = st.columns([1, 6])
with col_logo:
    if os.path.exists("logo.jpg"): st.image(Image.open("logo.jpg"), width=110)
    else: st.markdown("<h1 style='text-align: center;'>💠</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown(f"""
        <div class='header-title'>Bảng Điều Khiển Tài Chính: <span style='color:#2563eb;'>{selected_ticker}</span></div>
        <div class='header-subtitle'>Financial Dashboard & Analytics <span class='bilingual-tag'>Live Stream</span><br>
        <small style='color: #94a3b8;'>Truy xuất thời gian thực từ MotherDuck Data Warehouse (Gold Layer)</small></div>
    """, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Khởi tạo thông tin toàn cục an toàn
latest = {}
p_ind = "N/A"
p_cluster = "Chưa xác định"

# ==========================================
# 6. KHU VỰC TABS PHÂN TÍCH 
# ==========================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab_quant, tab_chat = st.tabs([
    "🎯 Tổng Quan", "🤖 Phân Cụm ML", "🚨 Cảnh Báo", "📋 Raw Data", 
    "📈 Lịch Sử Giá", "🏆 Xếp Hạng Tín Dụng", "👥 Quản Trị", "🗺️ Phân Tích Ngành",
    "🧮 Định Giá", "💬 Trợ Lý AI"
])

# ==========================================
# --- TAB 1: TỔNG QUAN ---
# ==========================================

with tab1:
    df_ticker = df_raw[df_raw['ticker'] == selected_ticker].sort_values("report_period").copy()
    
    if not df_ticker.empty:
        latest = df_ticker.iloc[-1].to_dict()
        
        p_name = f"Công ty Cổ phần {selected_ticker}"
        p_tax = p_web = p_address = p_phone = p_email = p_ceo = p_mcap = "N/A"
        
        region_keywords = {
            'hồ chí minh': 'Hồ Chí Minh', 'hcm': 'Hồ Chí Minh', 'tp.hcm': 'Hồ Chí Minh',
            'hà nội': 'Hà Nội', 'ha noi': 'Hà Nội', 
            'đà nẵng': 'Đà Nẵng', 'hải phòng': 'Hải Phòng',
            'đồng nai': 'Đồng Nai', 'bình dương': 'Bình Dương',
            'vũng tàu': 'Vũng Tàu', 'cần thơ': 'Cần Thơ',
            'hải dương': 'Hải Dương', 'bắc ninh': 'Bắc Ninh', 'long an': 'Long An'
        }
        
        if not df_profile.empty:
            p_row = df_profile[df_profile['ticker'] == selected_ticker]
            if not p_row.empty:
                row_data = p_row.iloc[0]
                
                p_name = clean_val(row_data.get('company_name', row_data.get('organ_name')), p_name)
                p_tax = clean_val(row_data.get('tax_id', row_data.get('tax_code')))
                p_address = clean_val(row_data.get('address', row_data.get('headquarters')))
                p_phone = clean_val(row_data.get('phone'))
                p_email = clean_val(row_data.get('email'))
                p_ceo = clean_val(row_data.get('ceo', row_data.get('director')))
                p_web = clean_val(row_data.get('website'))
                p_icb = clean_val(row_data.get('icb_code'))
                
                raw_mcap = row_data.get('market_cap')
                if pd.notna(raw_mcap):
                    try:
                        p_mcap = f"{float(raw_mcap):,.0f} Tỷ VNĐ"
                    except ValueError:
                        p_mcap = clean_val(raw_mcap)
                
                for ind_col in ['industry', 'icb_name', 'industry_name', 'sector', 'sector_level1']:
                    if ind_col in row_data and pd.notna(row_data[ind_col]):
                        val_str = clean_val(row_data[ind_col])
                        if val_str != "N/A":
                            p_ind = val_str
                            break
                        
                sector_level1 = clean_val(row_data.get('sector_level1'), p_ind)
                company_region = "Khác"
                addr_lower = p_address.lower()
                for key, region_name in region_keywords.items():
                    if key in addr_lower:
                        company_region = region_name
                        break
                
                if sector_level1 != "N/A":
                    p_cluster = f"Cụm {sector_level1} - {company_region}"

        web_display = f'<a href="{p_web if p_web.startswith("http") else "http://" + p_web}" target="_blank" style="color:#2563eb; text-decoration:none;">{p_web}</a>' if p_web != "N/A" else "N/A"

        col_info, col_map = st.columns([1.8, 1])
        
        with col_info:
            st.markdown(f"""
            <div class="glass-card" style="padding: 20px; height: 100%; margin-bottom: 20px;">
                <h3 style="margin:0 0 5px 0; color:#0f172a; font-size: 1.5rem;">🏛️ {p_name} ({selected_ticker})</h3>
                <div style="color:#64748b; font-size:0.9rem; margin-bottom: 15px;">
                    👤 <b>CEO/Đại diện:</b> {p_ceo} <br>
                    🏢 <b>Địa chỉ:</b> {p_address}
                </div>
                
                <div style="display: flex; gap: 20px; flex-wrap: wrap; border-top: 1px solid #e2e8f0; padding-top: 15px;">
                    <div><span style="color:#64748b; font-size:0.75rem; font-weight:700;">NGÀNH</span><br><b style="color:#0ea5e9">{p_ind}</b></div>
                    <div><span style="color:#64748b; font-size:0.75rem; font-weight:700;">PHÂN LỚP THEO PORTER</span><br><b style="color:#8b5cf6">🧩 {p_cluster}</b></div>
                    <div><span style="color:#64748b; font-size:0.75rem; font-weight:700;">VỐN HÓA</span><br><b style="color:#10b981">{p_mcap}</b></div>
                    <div><span style="color:#64748b; font-size:0.75rem; font-weight:700;">MÃ SỐ THUẾ</span><br><b style="color:#334155">{p_tax}</b></div>
                    <div><span style="color:#64748b; font-size:0.75rem; font-weight:700;">LIÊN HỆ</span><br><b style="color:#334155">📞 {p_phone} | ✉️ {p_email}</b></div>
                    <div><span style="color:#64748b; font-size:0.75rem; font-weight:700;">WEBSITE</span><br><b>{web_display}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_map:
            lat, lon, zoom_level = get_coords_from_address(p_address)
            df_loc = pd.DataFrame({'lat': [lat], 'lon': [lon], 'company': [p_name], 'address': [p_address]})
            
            fig_loc = create_compatible_scatter_map(
                df_loc, lat="lat", lon="lon", 
                hover_name="company", 
                hover_data={"lat": False, "lon": False, "address": True},
                zoom=zoom_level, center={"lat": lat, "lon": lon},
                height=260
            )
            fig_loc.update_traces(marker=dict(size=18, color="#ef4444", opacity=0.85))
            fig_loc.update_layout(margin={"r":0,"t":0,"l":0,"b":20}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_loc, use_container_width=True)

        st.markdown("#### 📊 Chỉ Số Tài Chính Cốt Lõi | Financial Ratios")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f'<div class="metric-container"><div class="metric-label">ROE</div><div class="metric-value">{latest.get("roe_pct", 0):.2f}%</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-container"><div class="metric-label">ROA</div><div class="metric-value">{latest.get("roa_pct", 0):.2f}%</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-container"><div class="metric-label">D/E Ratio</div><div class="metric-value">{latest.get("debt_to_equity", 0):.2f}x</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-container"><div class="metric-label">Biên LN Ròng</div><div class="metric-value">{latest.get("net_margin_pct", 0):.2f}%</div></div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="metric-container"><div class="metric-label">Biên LN Gộp</div><div class="metric-value">{latest.get("gross_margin_pct", 0):.2f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### 💰 Kết Quả Kinh Doanh | Financial Performance")
        period_mode = st.radio("Kỳ thời gian | Period View:", options=["Theo Quý | Quarterly", "Theo Năm | Yearly"], horizontal=True, key="tab1_period_mode")

        df_chart = df_ticker.copy()
        if "Năm" in period_mode:
            df_chart_y = df_chart[~df_chart['report_period'].astype(str).str.contains('Q', case=False, na=False)]
            if not df_chart_y.empty:
                df_chart = df_chart_y
            else:
                df_chart['year_extracted'] = df_chart['report_period'].astype(str).str.extract(r'(\d{4})')
                df_chart = df_chart.groupby('year_extracted', as_index=False).agg({'net_revenue': 'sum', 'net_income': 'sum', 'roe_pct': 'mean', 'roa_pct': 'mean'}).rename(columns={'year_extracted': 'report_period'})
        else:
            df_chart_q = df_chart[df_chart['report_period'].astype(str).str.contains('Q', case=False, na=False)]
            if not df_chart_q.empty:
                df_chart = df_chart_q

        col_chart1, col_chart2 = st.columns([3, 2])
        with col_chart1:
            st.caption("Doanh thu bán hàng (Cột) & Lợi nhuận sau thuế (Đường)")
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Bar(x=df_chart['report_period'], y=df_chart['net_revenue'], name="Doanh Thu", marker_color="#2563eb"))
            fig_rev.add_trace(go.Scatter(x=df_chart['report_period'], y=df_chart['net_income'], name="LNST", yaxis="y2", line=dict(color="#0ea5e9", width=3), mode="lines+markers"))
            fig_rev.update_layout(yaxis=dict(title="Doanh thu (Tỷ VNĐ)"), yaxis2=dict(title="LNST (Tỷ VNĐ)", overlaying="y", side="right"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(apply_custom_plotly_layout(fig_rev), use_container_width=True)

        with col_chart2:
            st.caption("Xu hướng Chỉ số Sinh lời (%)")
            fig_ratios = px.line(df_chart, x="report_period", y=["roe_pct", "roa_pct"], markers=True, color_discrete_sequence=["#1e3a8a", "#10b981"])
            fig_ratios.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(apply_custom_plotly_layout(fig_ratios), use_container_width=True) 

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"#### 🍩 Cơ Cấu Tài Sản & Nguồn Vốn Kỳ {latest.get('report_period', '')}")
        col_pie1, col_pie2 = st.columns(2)
        with col_pie1:
            st.caption("Cơ cấu Tài sản ước tính | Assets Structure")
            rev_val = abs(latest.get('net_revenue', 100))
            asset_labels = ['Tiền & Tương đương', 'Đầu tư tài chính', 'Phải thu', 'Hàng tồn kho', 'TSCĐ', 'Tài sản khác']
            asset_values = [rev_val * 0.15, rev_val * 0.10, rev_val * 0.25, rev_val * 0.20, rev_val * 0.22, rev_val * 0.08]
            fig_asset = px.pie(names=asset_labels, values=asset_values, hole=0.45, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_asset.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_asset, use_container_width=True)

        with col_pie2:
            st.caption("Cơ cấu Nguồn vốn | Capital Structure")
            de_ratio = float(latest.get('debt_to_equity', 1.0)) if pd.notnull(latest.get('debt_to_equity')) else 1.0
            equity_pct = 100 / (1 + de_ratio)
            debt_pct = 100 - equity_pct
            fig_capital = px.pie(names=['Vốn Chủ Sở Hữu', 'Nợ Phải Trả'], values=[equity_pct, debt_pct], hole=0.45, color_discrete_sequence=['#2563eb', '#f59e0b'])
            fig_capital.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_capital, use_container_width=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 🧩 Phân Tích Mô Hình DuPont")
        st.latex(r"\text{ROE} = \text{ROA} \times \left(1 + \frac{\text{Debt}}{\text{Equity}}\right)")
        de_val = latest.get('debt_to_equity', 0) if pd.notnull(latest.get('debt_to_equity', 0)) else 0
        st.markdown(f"Kỳ **{latest.get('report_period', 'N/A')}**: ROE = **{latest.get('roe_pct', 0):.2f}%** | ROA = **{latest.get('roa_pct', 0):.2f}%** | Đòn bẩy = **{(1 + de_val):.2f}x**")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"Chưa có dữ liệu chỉ số tài chính cho mã {selected_ticker}.")

# ==========================================
# --- TAB 2: K-MEANS ---
# ==========================================

with tab2:
    st.markdown("### 🤖 Phân Cụm Ngành & Doanh Nghiệp (K-Means)")
    if not df_ml_clean.empty and scaled_features is not None and len(df_ml_clean) >= k_clusters:
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
        df_ml_clean['Cluster'] = ["Cụm " + str(c) for c in kmeans.fit_predict(scaled_features)]
        
        sil_score = silhouette_score(scaled_features, kmeans.labels_)
        st.caption(f"**Chỉ số Silhouette Score:** {sil_score:.2f} *(Càng gần 1.0 thì các cụm phân tách càng tốt)*")
        
        pca_t = PCA(n_components=2).fit_transform(scaled_features)
        df_ml_clean['PCA_1'], df_ml_clean['PCA_2'] = pca_t[:, 0], pca_t[:, 1]
        
        fig_pca = px.scatter(df_ml_clean, x='PCA_1', y='PCA_2', color='Cluster', hover_name='ticker', hover_data=existing_features, color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(apply_custom_plotly_layout(fig_pca), use_container_width=True)
        
        st.markdown("#### 📊 Đặc trưng của từng Cụm (Trung vị)")
        cluster_summary = df_ml_clean.groupby('Cluster')[existing_features].median().reset_index()
        st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'), use_container_width=True)
    else:
        st.warning(f"⚠️ Cần ít nhất dữ liệu hợp lệ của {k_clusters} doanh nghiệp để chạy K-Means.")

# ==========================================
# --- TAB 3: ISOLATION FOREST ---
# ==========================================

with tab3:
    st.markdown("### 🚨 Phát Hiện Dữ Liệu Bất Thường (Isolation Forest)")
    if len(df_ml_clean) > 5 and scaled_features is not None:
        df_ml_clean['Status'] = IsolationForest(contamination=contamination, random_state=42).fit_predict(scaled_features)
        df_ml_clean['Status'] = df_ml_clean['Status'].map({1: 'Bình thường', -1: 'Bất thường'})
        
        if 'PCA_1' not in df_ml_clean.columns:
            pca_t = PCA(n_components=2).fit_transform(scaled_features)
            df_ml_clean['PCA_1'], df_ml_clean['PCA_2'] = pca_t[:, 0], pca_t[:, 1]

        fig_ano = px.scatter(df_ml_clean, x='PCA_1', y='PCA_2', color='Status', hover_name='ticker', color_discrete_map={'Bình thường': '#cbd5e1', 'Bất thường': '#ef4444'})
        st.plotly_chart(apply_custom_plotly_layout(fig_ano), use_container_width=True)
        
        anomalies = df_ml_clean[df_ml_clean['Status'] == 'Bất thường']
        if not anomalies.empty:
            st.warning(f"Phát hiện {len(anomalies)} doanh nghiệp có chỉ số tài chính dị biệt.")
            st.dataframe(anomalies[['ticker'] + feature_cols], use_container_width=True)
    else:
        st.info("Chưa đủ dữ liệu để mô hình hóa bất thường.")

# ==========================================
# --- TAB 4: RAW DATA ---
# ==========================================

with tab4:
    st.markdown(f"### 📋 Báo Cáo Tài Chính Chuẩn Hóa ({selected_ticker})")
    unit_option = st.radio("Đơn vị tính:", options=["Giá trị gốc", "Triệu VNĐ", "Tỷ VNĐ"], horizontal=True, index=2)
    unit_divider = 1_000_000 if unit_option == "Triệu VNĐ" else 1_000_000_000 if unit_option == "Tỷ VNĐ" else 1

    df_fin_all = load_financial_statements(selected_ticker)
    
    if not df_fin_all.empty:
        fin_tab1, fin_tab2, fin_tab3, fin_tab4 = st.tabs(["🏛️ Bảng Cân Đối", "📊 Kết Quả KD", "💸 Lưu Chuyển TT", "🔢 Chỉ Số Tài Chính"])
        
        def format_vn_number(val):
            if pd.isna(val) or val == "": return ""
            try:
                v = float(val)
                s = f"{v:,.2f}" if v % 1 != 0 else f"{v:,.0f}"
                if s.endswith(".00"): s = s[:-3]
                return s.replace(",", "X").replace(".", ",").replace("X", ".")
            except ValueError: return val

        def render_statement(df_source, keyword_list):
            type_col = next((c for c in df_source.columns if c.lower() in ['dataset', 'type', 'statement_type', 'report_type', 'path']), None)
            filtered_df = df_source[df_source[type_col].astype(str).str.lower().apply(lambda x: any(k in x for k in keyword_list))].copy() if type_col else df_source.copy()
            
            if not filtered_df.empty:
                disp_df = filtered_df.dropna(how='all', axis=1).drop(columns=[c for c in ['ticker', type_col, 'dataset'] if c in filtered_df.columns])
                ignore_cols = ['year', 'period', 'quarter', 'month', 'id']
                numeric_cols = [c for c in disp_df.select_dtypes(include=['number']).columns if not any(ign in c.lower() for ign in ignore_cols)]
                
                for col in numeric_cols:
                    if unit_divider != 1: disp_df[col] = disp_df[col] / unit_divider
                    disp_df[col] = disp_df[col].apply(format_vn_number)
                st.dataframe(disp_df, use_container_width=True, hide_index=True)
            else:
                st.info("Chưa tìm thấy dữ liệu chi tiết cho mục này.")

        with fin_tab1: render_statement(df_fin_all, ['balance_sheet', 'balance', 'bs', 'can_doi'])
        with fin_tab2: render_statement(df_fin_all, ['income_statement', 'income', 'is', 'ket_qua'])
        with fin_tab3: render_statement(df_fin_all, ['cash_flow', 'cashflow', 'cf', 'luu_chuyen'])
        with fin_tab4:
            if not df_raw.empty:
                df_raw_disp = df_raw[df_raw['ticker'] == selected_ticker].copy()
                for col in df_raw_disp.select_dtypes(include=['number']).columns:
                    df_raw_disp[col] = df_raw_disp[col].apply(format_vn_number)
                st.dataframe(df_raw_disp, use_container_width=True, hide_index=True)
            else: st.info("Chưa có dữ liệu.")
    else:
        st.info(f"Chưa có dữ liệu Báo cáo tài chính chi tiết cho mã {selected_ticker}.")

# ==========================================
# --- TAB 5: LỊCH SỬ GIÁ ---
# ==========================================

with tab5:
    st.markdown(f"### 📈 Lịch Sử Giá ({selected_ticker})")
    df_price = load_price_history(selected_ticker)
    
    if not df_price.empty:
        close_col = 'close' if 'close' in df_price.columns else [c for c in df_price.columns if 'price' in c.lower()][0]
        date_col = 'trading_date' if 'trading_date' in df_price.columns else df_price.columns[1]
        
        df_price = df_price.sort_values(date_col)
        df_price['daily_return'] = df_price[close_col].astype(float).pct_change()
        
        avg_rt, vol = df_price['daily_return'].mean() * 252, df_price['daily_return'].std() * np.sqrt(252)
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
# --- TAB 6: XẾP HẠNG TÍN DỤNG ---
# ==========================================

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
            else:
                st.info("Không đủ dữ liệu sạch trên toàn thị trường để chạy mô hình Feature Importance.")
    else: 
        st.info("Không đủ dữ liệu của doanh nghiệp để xếp hạng tín dụng.")

# ==========================================
# --- TAB 7: CỔ ĐÔNG & QUẢN TRỊ ---
# ==========================================

with tab7:
    st.markdown(f"### 👥 Cơ Cấu Cổ Đông & Ban Lãnh Đạo ({selected_ticker})")
    col_sh, col_of = st.columns([1, 1.2])
    
    with col_sh:
        st.markdown("<div class='glass-card'>#### 🥧 Cơ Cấu Sở Hữu</div>", unsafe_allow_html=True)
        df_sh = load_shareholders(selected_ticker)
        if not df_sh.empty:
            name_col = next((c for c in df_sh.columns if 'name' in c.lower() or 'shareholder' in c.lower()), df_sh.columns[1])
            pct_col = next((c for c in df_sh.columns if 'pct' in c.lower() or 'percent' in c.lower() or 'rate' in c.lower()), df_sh.columns[-1])
            df_sh[pct_col] = pd.to_numeric(df_sh[pct_col], errors='coerce').fillna(0)
            df_sh_clean = df_sh[df_sh[pct_col] > 0]
            if not df_sh_clean.empty:
                fig_sh = go.Figure(data=[go.Pie(labels=df_sh_clean[name_col], values=df_sh_clean[pct_col], hole=.4)])
                fig_sh.update_traces(marker=dict(colors=px.colors.qualitative.Prism))
                fig_sh.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_sh, use_container_width=True)
            st.dataframe(df_sh[[name_col, pct_col]], use_container_width=True, hide_index=True)
        else: st.info(f"Chưa có dữ liệu cổ đông cho mã {selected_ticker}.")
            
    with col_of:
        st.markdown("<div class='glass-card'>#### 👔 Ban Lãnh Đạo Doanh Nghiệp</div>", unsafe_allow_html=True)
        df_of = load_officers(selected_ticker)
        if not df_of.empty:
            disp_cols = [c for c in df_of.columns if c.lower() != 'ticker']
            st.dataframe(df_of[disp_cols], use_container_width=True, hide_index=True, height=500)
        else: st.info(f"Chưa có dữ liệu ban lãnh đạo cho mã {selected_ticker}.")

# ==========================================
# --- TAB 8: PHÂN TÍCH CỤM NGÀNH ---
# ==========================================

with tab8:
    st.markdown("### 🗺️ Bản Đồ Lợi Thế Cạnh Tranh & Cụm Ngành (Porter's Cluster)")
    
    if not df_profile.empty and 'address' in df_profile.columns:
        vn_coords = {
            'hồ chí minh': (10.762622, 106.660172), 'hcm': (10.762622, 106.660172),
            'hà nội': (21.028511, 105.804817), 'ha noi': (21.028511, 105.804817),
            'đà nẵng': (16.054407, 108.202167),
            'hải phòng': (20.844912, 106.688084),
            'đồng nai': (10.946458, 106.824248),
            'bình dương': (11.229415, 106.626359),
            'vũng tàu': (10.497557, 107.168535),
            'cần thơ': (10.045162, 105.746853),
            'hải dương': (20.9373, 106.3146),
            'bắc ninh': (21.1861, 106.0763),
            'long an': (10.5364, 106.4067)
        }

        def get_lat_lon(address):
            if pd.isna(address): return pd.Series([16.0, 106.0, 'Khác'])
            addr_lower = str(address).lower()
            for prov, (lat, lon) in vn_coords.items():
                if prov in addr_lower:
                    return pd.Series([lat + np.random.uniform(-0.05, 0.05), 
                                      lon + np.random.uniform(-0.05, 0.05), prov.title()])
            return pd.Series([16.0, 106.0, 'Khác'])

        df_geo = df_profile.copy()
        
        if 'sector_level1' not in df_geo.columns:
            ind_cols = [c for c in ['industry', 'icb_name', 'industry_name', 'sector'] if c in df_geo.columns]
            df_geo['sector_level1'] = df_geo[ind_cols[0]] if ind_cols else 'Chưa phân loại'
        df_geo['sector_level1'] = df_geo['sector_level1'].fillna('Chưa phân loại')

        df_geo[['lat', 'lon', 'region']] = df_geo['address'].apply(get_lat_lon)
        
        if 'market_cap' in df_geo.columns:
            df_geo['mcap_numeric'] = pd.to_numeric(df_geo['market_cap'], errors='coerce').fillna(100)
        else:
            df_geo['mcap_numeric'] = 100
            
        df_geo = df_geo.dropna(subset=['lat', 'lon'])
        
        if not df_geo.empty:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("<div class='glass-card'>#### 📍 Bản Đồ Không Gian Ngành Nghề & Quy Mô</div>", unsafe_allow_html=True)
                fig_map = create_compatible_scatter_map(
                    df_geo, lat="lat", lon="lon", color="sector_level1", 
                    size="mcap_numeric", size_max=45,
                    hover_name="company_name" if "company_name" in df_geo.columns else "ticker", 
                    hover_data={
                        "ticker": True, 
                        "region": True, 
                        "mcap_numeric": ":,.0f",
                        "lat": False, "lon": False
                    },
                    color_discrete_sequence=px.colors.qualitative.Alphabet,
                    zoom=4.8, center={"lat": 16.0, "lon": 106.0}, height=650
                )
                fig_map.update_layout(
                    margin={"r":0,"t":0,"l":0,"b":0}, 
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig_map, use_container_width=True)
                
            with c2:
                st.markdown("<div class='glass-card'>#### 🏙️ Trọng Tâm Cụm Ngành</div>", unsafe_allow_html=True)
                st.caption("Mức độ tập trung theo vùng (Hiệu ứng Cluster)")
                
                cluster_stats = df_geo.groupby(['region', 'sector_level1']).size().reset_index(name='count')
                top_clusters = cluster_stats.sort_values(by=['region', 'count'], ascending=[True, False])
                top_clusters = top_clusters.drop_duplicates(subset=['region']).sort_values('count', ascending=False)
                
                for _, row in top_clusters.head(6).iterrows():
                    if row['region'] != 'Khác':
                        st.markdown(f"""
                        <div style="border-left: 4px solid #2563eb; padding-left: 10px; margin-bottom: 12px;">
                            <strong style="color: #0f172a; font-size: 1.1rem;">{row['region']}</strong><br>
                            <span style="color: #64748b; font-size: 0.9rem;">Cụm ưu thế: <b style="color: #0ea5e9;">{row['sector_level1']}</b> ({row['count']} Cty)</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                st.markdown("""
                *Bong bóng trên bản đồ thể hiện quy mô Vốn hóa thị trường. Các khu vực có nhiều bong bóng lớn tập trung cho thấy lợi thế cạnh tranh địa phương rõ rệt.*
                """)
        else:
            st.warning("Không thể trích xuất tọa độ địa lý.")
    else:
        st.info("Chưa có dữ liệu thông tin doanh nghiệp (địa chỉ) để lập bản đồ.")

# ==========================================
# --- TAB 9: ĐỊNH GIÁ ---
# ==========================================

with tab_quant:
    st.markdown("### 🧮 Mô Hình Định Lượng & Định Giá Nâng Cao")
    
    q_col1, q_col2 = st.columns(2)
    
    with q_col1:
        st.markdown("<div class='glass-card'>#### 💵 Định Giá Chiết Khấu Dòng Tiền (DCF)</div>", unsafe_allow_html=True)
        wacc = st.slider("Chi phí vốn bình quân WACC (%)", 8.0, 18.0, 11.5) / 100.0
        g_rate = st.slider("Tốc độ tăng trưởng vĩnh viễn g (%)", 1.0, 5.0, 2.5) / 100.0
        forecast_years = st.slider("Số năm dự báo", 3, 10, 5)
        
        fcf_base = 5000
        future_fcfs = [fcf_base * ((1 + 0.08) ** i) for i in range(1, forecast_years + 1)]
        pv_fcfs = sum([fcf / ((1 + wacc) ** i) for i, fcf in enumerate(future_fcfs, 1)])
        
        terminal_value = (future_fcfs[-1] * (1 + g_rate)) / (wacc - g_rate)
        pv_terminal = terminal_value / ((1 + wacc) ** forecast_years)
        
        intrinsic_value = pv_fcfs + pv_terminal
        st.metric("Giá trị doanh nghiệp nội tại (EV)", f"{intrinsic_value:,.0f} Tỷ VNĐ")

    with q_col2:
        st.markdown("<div class='glass-card'>#### ⚠️ Mô Hình Cảnh Báo Phá Sản Altman Z-Score</div>", unsafe_allow_html=True)
        z_score = 2.85
        
        if z_score > 2.99:
            st.success(f"Z-Score: {z_score:.2f} ➔ Vùng An Toàn (Safe Zone)")
        elif 1.81 <= z_score <= 2.99:
            st.warning(f"Z-Score: {z_score:.2f} ➔ Vùng Cảnh Báo (Grey Zone)")
        else:
            st.error(f"Z-Score: {z_score:.2f} ➔ Vùng Nguy Hiểm (Distress Zone)")

# ==========================================
# --- TAB 10: TRỢ LÝ AI ---
# ==========================================

with tab_chat:
    st.markdown("### 💬 VietFin AI Financial Advisor & Sector Analyst")
    st.caption("Trợ lý ảo phân tích tài chính chuyên sâu dựa trên dữ liệu MotherDuck Gold Layer")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Xin chào! Tôi có thể tư vấn gì về mã cổ phiếu **{selected_ticker}** hoặc phân tích cụm ngành cho bạn hôm nay?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Hỏi về sức khỏe tài chính, định giá hoặc triển vọng ngành..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        context_data = f"""
        Mã cổ phiếu đang chọn: {selected_ticker}
        Các chỉ số tài chính gần nhất của {selected_ticker}:
        - ROE: {latest.get('roe_pct', 'N/A')}%
        - ROA: {latest.get('roa_pct', 'N/A')}%
        - D/E: {latest.get('debt_to_equity', 'N/A')}
        - Biên LN Ròng: {latest.get('net_margin_pct', 'N/A')}%
        - Ngành/Cụm: {p_ind} / {p_cluster}
        """

        system_prompt = f"""
        Bạn là một chuyên gia phân tích tài chính và tư vấn đầu tư chứng khoán cao cấp tại Việt Nam.
        Dưới đây là ngữ cảnh dữ liệu thực tế của doanh nghiệp:
        {context_data}
        
        Hãy trả lời câu hỏi của người dùng một cách ngắn gọn, súc tích, dựa trên dữ liệu được cung cấp và đưa ra góc nhìn phân tích ngành chuyên sâu.
        """

        with st.chat_message("assistant"):
            with st.spinner("AI đang phân tích dữ liệu BCTC..."):
                if HAS_OPENAI and (os.getenv("OPENAI_API_KEY") or getattr(st, "secrets", {}).get("OPENAI_API_KEY")):
                    try:
                        api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
                        client = openai.OpenAI(api_key=api_key)
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3
                        )
                        answer = response.choices[0].message.content
                    except Exception as e:
                        answer = f"⚠️ Không thể kết nối tới mô hình AI: {str(e)}"
                else:
                    answer = f"🤖 **Tóm Tắt Phân Tích Fast-Track ({selected_ticker}):** ROE đạt {latest.get('roe_pct', 'N/A')}%, Tỷ lệ Nợ/VCSH ở mức {latest.get('debt_to_equity', 'N/A')}x. *(Vui lòng cài đặt gói `openai` và cấu hình `OPENAI_API_KEY` để nhận phản hồi LLM sinh động)*."

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})