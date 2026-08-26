import os
import duckdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import streamlit as st

# ==========================================
# 1. CẤU HÌNH TRANG & THIẾT KẾ ĐỒ HỌA (CSS)
# ==========================================

st.set_page_config(
    page_title="VietFin Intelligence | Quant & Corporate",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Khởi tạo trạng thái ngôn ngữ mặc định
if "lang" not in st.session_state:
    st.session_state.lang = "VI"

# Giao diện FinTech Premium: Phối hợp dải màu Indigo - Slate & Emerald Cyan
st.markdown("""
<style>
    @import url('https://googleapis.com');
    
    /* Toàn bộ nền ứng dụng */
    .stApp { 
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%); 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Thiết kế thanh Sidebar đậm chất chuyên nghiệp */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        box-shadow: 4px 0 25px rgba(15, 23, 42, 0.15);
    }
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }
    
    /* Tùy biến thanh chọn và thanh trượt trong Sidebar bớt đơn điệu */
    [data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #1e293b !important;
        border-radius: 8px;
    }
    
    /* Thiết kế thanh Tabs dạng 'Pills' chuyển động mềm mại */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 10px; 
        border: none; 
        padding: 8px;
        background: rgba(226, 232, 240, 0.7);
        border-radius: 14px;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] { 
        background-color: transparent; 
        border-radius: 10px; 
        padding: 10px 22px; 
        border: none !important;
        font-weight: 700; 
        color: #475569; 
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stTabs [data-baseweb="tab"]:hover { 
        color: #1e3a8a;
        background-color: rgba(255, 255, 255, 0.6);
    }
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important; 
        color: #ffffff !important; 
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.25) !important; 
    }
    
    /* Thẻ Container có chiều sâu đồ họa (Card) */
    .glass-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.03), 0 1px 3px rgba(0, 0, 0, 0.01);
        border: 1px solid rgba(226, 232, 240, 0.8);
        margin-bottom: 24px;
        transition: transform 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
    }
    
    /* Hộp hiển thị Chỉ số (Metric) có viền Gradient */
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 4px 6px rgba(15, 23, 42, 0.01);
        position: relative;
        overflow: hidden;
    }
    /* Tạo dải trang trí màu sắc ở cạnh trên hộp metric */
    .metric-container::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, #3b82f6, #0ea5e9);
    }
    .metric-container:hover {
        border-color: #cbd5e1;
        box-shadow: 0 16px 24px -8px rgba(15, 23, 42, 0.06);
    }
    .metric-label { 
        font-size: 0.75rem; 
        font-weight: 700; 
        text-transform: uppercase; 
        color: #64748b; 
        letter-spacing: 0.08em; 
    }
    .metric-value { 
        font-size: 2.2rem; 
        font-weight: 800; 
        margin: 8px 0 0 0; 
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
    }
    
    /* Thiết kế lại nút bấm chính đẹp hơn */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.2s !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

def apply_custom_plotly_layout(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#475569"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color='#64748b')),
        yaxis=dict(showgrid=True, gridcolor='#e2e8f0', zeroline=False, tickfont=dict(color='#64748b')),
        hovermode="x unified"
    )
    return fig

# Từ điển nhãn ngữ cảnh đa ngôn ngữ
DICT = {
    "VI": {
        "err_token": "❌ Chưa cấu hình MOTHERDUCK_TOKEN trong secrets hoặc .env",
        "err_load": "⚠️ Chưa nạp được dữ liệu từ kho MotherDuck (fact_ratio_summary). Vui lòng kiểm tra lại kết nối đồng bộ.",
        "sys_caption": "Hệ thống Phân tích Định lượng & ML",
        "lbl_lang": "🌐 Ngôn ngữ / Language",
        "lbl_ticker": "🔍 Mã Cổ Phiếu Phân Tích",
        "lbl_ml_cfg": "⚙️ Cấu hình Machine Learning",
        "lbl_k": "Số cụm K-Means",
        "lbl_anomaly": "Ngưỡng bất thường (Anomaly)",
        "btn_refresh": "🔄 Làm mới kho dữ liệu",
        "header_title": "📊 Bảng Điều Khiển Tài Chính",
        "header_caption": "Truy xuất thời gian thực từ MotherDuck Data Warehouse (Gold Layer)"
    },
    "EN": {
        "err_token": "❌ MOTHERDUCK_TOKEN is missing in secrets or .env file",
        "err_load": "⚠️ Failed to fetch data from MotherDuck (fact_ratio_summary). Please check connections.",
        "sys_caption": "Quantitative & ML Analytics System",
        "lbl_lang": "🌐 Language / Ngôn ngữ",
        "lbl_ticker": "🔍 Select Ticker Symbol",
        "lbl_ml_cfg": "⚙️ Machine Learning Config",
        "lbl_k": "K-Means Cluster Count",
        "lbl_anomaly": "Anomaly Contamination Threshold",
        "btn_refresh": "🔄 Refresh Data Warehouse",
        "header_title": "📊 Financial Intelligence Dashboard",
        "header_caption": "Real-time query pipelines powered by MotherDuck (Gold Layer)"
    }
}

L = DICT[st.session_state.lang]

# ==========================================
# 2. KẾT NỐI DỮ LIỆU MOTHERDUCK (GOLD LAYER)
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

df_raw = load_gold_ratios()
df_profile = load_corporate_overview()

if df_raw.empty:
    st.warning(L["err_load"])
    st.stop()

# ==========================================
# 3. SIDEBAR & HEADER
# ==========================================
with st.sidebar:
    # 3.1. Chèn Logo cục bộ "logo.jpg" vào vị trí đầu tiên của Sidebar kèm tiêu đề cạnh nhau
    logo_path = "logo.jpg"
    if os.path.exists(logo_path):
        col_logo, col_title = st.columns([1, 2.5])
        with col_logo:
            st.image(logo_path, use_container_width=True)
        with col_title:
            st.markdown("<h2 style='margin-top:0px; color:#ffffff; font-weight:800;'>VietFin Pro</h2>", unsafe_allow_html=True)
    else:
        st.title("VietFin Pro")
        st.warning("⚠️ File logo.jpg không tìm thấy tại thư mục app.")
        
    st.caption(L["sys_caption"])
    st.divider()

    # 3.2. Bộ chuyển đổi cấu hình song ngữ
    lang_choice = st.selectbox(
        L["lbl_lang"], 
        ["Tiếng Việt (VI)", "English (EN)"], 
        index=0 if st.session_state.lang == "VI" else 1
    )
    new_lang = "VI" if "Tiếng Việt" in lang_choice else "EN"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    st.divider()

    # 3.3. Các bộ lọc tham số

# ==========================================
# 4. KHU VỰC TABS PHÂN TÍCH
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 Tổng Quan & DuPont", 
    "🤖 Phân Cụm ML", 
    "🚨 Cảnh Báo", 
    "📋 Raw Data", 
    "📈 Lịch Sử Giá", 
    "🏆 Xếp Hạng Tín Dụng",
    "👥 Quản Trị & Cổ Đông"
])

# ==========================================

# TAB 1: TỔNG QUAN & DUPONT
with tab1:
    df_ticker = df_raw[df_raw['ticker'] == selected_ticker].sort_values("report_period").copy()
    
    if not df_ticker.empty:
        latest = df_ticker.iloc[-1]
        
        p_row = df_profile[df_profile['ticker'] == selected_ticker] if not df_profile.empty and 'ticker' in df_profile.columns else pd.DataFrame()
        p_name = p_row.iloc[0].get('company_name', f"Công ty CP {selected_ticker}") if not p_row.empty else f"Công ty CP {selected_ticker}"
        p_tax = p_row.iloc[0].get('tax_id', p_row.iloc[0].get('tax_code', 'N/A')) if not p_row.empty else "N/A"
        p_ind = p_row.iloc[0].get('industry', 'N/A') if not p_row.empty else "N/A"
        p_web = p_row.iloc[0].get('website', 'N/A') if not p_row.empty else "N/A"

        st.markdown(f"""
        <div class="glass-card">
            <h3 style="margin:0 0 12px 0; color:#0f172a; font-size: 1.6rem;">🏛️ {p_name}</h3>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div><span style="color:#64748b; font-size:0.85rem">NGÀNH NGHỀ</span><br><b style="color:#0ea5e9">{p_ind}</b></div>
                <div><span style="color:#64748b; font-size:0.85rem">MÃ SỐ THUẾ</span><br><b style="color:#334155">{p_tax}</b></div>
                <div><span style="color:#64748b; font-size:0.85rem">KỲ BÁO CÁO GẦN NHẤT</span><br><b style="color:#334155">{latest['report_period']}</b></div>
                <div><span style="color:#64748b; font-size:0.85rem">WEBSITE</span><br><b><a href="{p_web}" target="_blank" style="color:#1e3a8a; text-decoration:none;">{p_web}</a></b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-container"><div class="metric-label">ROE</div><div class="metric-value">{latest.get("roe_pct", 0):.2f}%</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-container"><div class="metric-label">ROA</div><div class="metric-value">{latest.get("roa_pct", 0):.2f}%</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-container"><div class="metric-label">D/E Ratio</div><div class="metric-value">{latest.get("debt_to_equity", 0):.2f}x</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-container"><div class="metric-label">Biên LN Ròng</div><div class="metric-value">{latest.get("net_margin_pct", 0):.2f}%</div></div>', unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("<h4 style='color:#334155'>💰 Doanh Thu & Lợi Nhuận</h4>", unsafe_allow_html=True)
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Bar(x=df_ticker['report_period'], y=df_ticker['net_revenue'], name="Doanh thu", marker_color="#bae6fd"))
            fig_rev.add_trace(go.Scatter(x=df_ticker['report_period'], y=df_ticker['net_income'], name="LNST", yaxis="y2", line=dict(color="#0ea5e9", width=4)))
            fig_rev.update_layout(yaxis=dict(title="Doanh thu"), yaxis2=dict(title="LNST", overlaying="y", side="right"), showlegend=False)
            st.plotly_chart(apply_custom_plotly_layout(fig_rev), use_container_width=True)

        with col_r:
            st.markdown("<h4 style='color:#334155'>📉 Chỉ Số Lợi Nhuận</h4>", unsafe_allow_html=True)
            fig_ratios = px.line(
                df_ticker, x="report_period", y=["roe_pct", "roa_pct", "gross_margin_pct", "net_margin_pct"],
                markers=True, color_discrete_sequence=["#1e3a8a", "#0ea5e9", "#10b981", "#f59e0b"]
            )
            st.plotly_chart(apply_custom_plotly_layout(fig_ratios), use_container_width=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 🧩 Phân Tích Mô Hình DuPont")
        st.latex(r"\text{ROE} = \text{ROA} \times \left(1 + \frac{\text{Debt}}{\text{Equity}}\right)")
        de_val = latest.get('debt_to_equity', 0) if pd.notnull(latest.get('debt_to_equity', 0)) else 0
        st.markdown(f"Kỳ **{latest['report_period']}**: ROE = **{latest.get('roe_pct', 0):.2f}%** | ROA = **{latest.get('roa_pct', 0):.2f}%** | Đòn bẩy = **{(1 + de_val):.2f}x**")
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================

# LÀM SẠCH DỮ LIỆU CHO ML
feature_cols = ['roe_pct', 'roa_pct', 'debt_to_equity', 'gross_margin_pct', 'net_margin_pct']
df_ml = df_raw.sort_values("report_period").groupby("ticker").last().reset_index()
df_ml_clean = df_ml.dropna(subset=feature_cols).copy()

scaled_features = None
if len(df_ml_clean) > 0:
    scaled_features = StandardScaler().fit_transform(df_ml_clean[feature_cols])

# ==========================================

# TAB 2: PHÂN CỤM ML (K-MEANS)
with tab2:
    st.markdown("### 🤖 Phân Cụm Ngành & Doanh Nghiệp (K-Means)")
    if len(df_ml_clean) >= k_clusters and scaled_features is not None:
        df_ml_clean['Cluster'] = "Cụm " + KMeans(n_clusters=k_clusters, random_state=42, n_init=10).fit_predict(scaled_features).astype(str)
        pca_t = PCA(n_components=2).fit_transform(scaled_features)
        df_ml_clean['PCA_1'], df_ml_clean['PCA_2'] = pca_t[:, 0], pca_t[:, 1]
        
        fig_pca = px.scatter(
            df_ml_clean, x='PCA_1', y='PCA_2', color='Cluster',
            hover_name='ticker', hover_data=feature_cols,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pca.update_traces(marker=dict(size=12, line=dict(width=1, color='White')))
        st.plotly_chart(apply_custom_plotly_layout(fig_pca), use_container_width=True)
    else:
        st.info("Chưa đủ số lượng dữ liệu doanh nghiệp để phân cụm.")

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