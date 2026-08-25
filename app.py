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

# ==========================================
# 1. CẤU HÌNH TRANG & THIẾT KẾ ĐỒ HỌA (CSS)
# ==========================================
st.set_page_config(
    page_title="VietFin Intelligence | Quant & Corporate",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Tổng thể nền và Font chữ */
    .stApp { 
        background-color: #f4f7f9; 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Thiết kế các Tab dạng khối hiện đại (Pills) */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 12px; 
        border: none; 
        padding-bottom: 5px; 
    }
    .stTabs [data-baseweb="tab"] { 
        background-color: #ffffff; 
        border-radius: 8px; 
        padding: 10px 20px; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); 
        border: 1px solid #e2e8f0; 
        font-weight: 600; 
        color: #64748b; 
        transition: all 0.3s ease; 
    }
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #94a3b8;
    }
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important; 
        color: white !important; 
        border: none !important; 
        box-shadow: 0 4px 10px rgba(59,130,246,0.3) !important; 
    }

    /* Glass Card cho Thông tin doanh nghiệp */
    .glass-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        border: 1px solid rgba(226, 232, 240, 0.8);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 24px;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
    }

    /* Metric Cards Đẹp mắt */
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        transition: all 0.3s;
    }
    .metric-container:hover {
        border-color: #bae6fd;
        box-shadow: 0 8px 12px rgba(14, 165, 233, 0.08);
    }
    .metric-label { 
        font-size: 0.8rem; 
        font-weight: 700; 
        text-transform: uppercase; 
        color: #64748b; 
        letter-spacing: 0.05em; 
    }
    .metric-value { 
        font-size: 2.2rem; 
        font-weight: 900; 
        margin: 8px 0; 
        background: -webkit-linear-gradient(45deg, #1e3a8a, #0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Tùy chỉnh thanh cuộn (Scrollbar) */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# Hàm chuẩn hóa layout đồ thị Plotly
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
# 2. KẾT NỐI DỮ LIỆU THỰC TẾ
# ==========================================
load_dotenv()
try:
    md_token = st.secrets["MOTHERDUCK_TOKEN"]
except Exception:
    md_token = os.getenv("MOTHERDUCK_TOKEN")

if not md_token:
    st.error("❌ Chưa cấu hình MOTHERDUCK_TOKEN")
    st.stop()

@st.cache_resource
def get_db_connection():
    return duckdb.connect(f"md:vietfin_db?token={md_token}")

con = get_db_connection()

@st.cache_data(ttl=1800)
def load_gold_ratios():
    try:
        return con.execute("""
            SELECT ticker, report_period, net_revenue, net_income, roe_pct, roa_pct, 
                   debt_to_equity, gross_margin_pct, net_margin_pct
            FROM gold_financial_ratios
        """).df()
    except: return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_corporate_overview():
    try: return con.execute("SELECT * FROM gold_corporate_overview").df()
    except: return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_shareholders(ticker):
    try: return con.execute(f"SELECT * FROM gold_corporate_shareholders WHERE ticker = '{ticker}'").df()
    except: return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_officers(ticker):
    try: return con.execute(f"SELECT * FROM gold_corporate_officers WHERE ticker = '{ticker}'").df()
    except: return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_price_history(ticker):
    try: return con.execute(f"SELECT * FROM gold_corporate_price_history WHERE ticker = '{ticker}'").df()
    except: return pd.DataFrame()

df_raw = load_gold_ratios()
df_profile = load_corporate_overview()

if df_raw.empty:
    st.warning("⚠️ Không có dữ liệu tài chính trong kho MotherDuck.")
    st.stop()

# ==========================================
# 3. SIDEBAR & HEADER
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3256/3256094.png", width=60)
    st.title("VietFin Pro")
    st.caption("Hệ thống Định lượng & Học máy Tài chính")
    st.divider()

    tickers = sorted(df_raw['ticker'].dropna().unique())
    default_idx = tickers.index("VNM") if "VNM" in tickers else 0
    selected_ticker = st.selectbox("🔍 Mã Cổ Phiếu Phân Tích", tickers, index=default_idx)

    st.subheader("⚙️ Cấu hình Machine Learning")
    k_clusters = st.slider("Số cụm K-Means", min_value=2, max_value=6, value=3)
    contamination = st.slider("Ngưỡng bất thường (Anomaly)", min_value=1, max_value=15, value=5) / 100.0

    st.divider()
    if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown(f"<h2>📊 Bảng Điều Khiển Tài Chính: <span style='color:#1e3a8a'>{selected_ticker}</span></h2>", unsafe_allow_html=True)
st.caption("Truy xuất thời gian thực từ MotherDuck Data Warehouse")
st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 4. KHU VỰC TABS (NỘI DUNG CHÍNH)
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

# ----------------------------------------------------
# TAB 1: TỔNG QUAN DOANH NGHIỆP & CHỈ SỐ
# ----------------------------------------------------
with tab1:
    df_ticker = df_raw[df_raw['ticker'] == selected_ticker].sort_values("report_period").copy()
    
    if not df_ticker.empty:
        latest = df_ticker.iloc[-1]
        
        # Profile Data
        if not df_profile.empty and 'ticker' in df_profile.columns:
            p_row = df_profile[df_profile['ticker'] == selected_ticker]
        else: p_row = pd.DataFrame()
            
        p_name = p_row.iloc[0].get('company_name', f"Công ty CP {selected_ticker}") if not p_row.empty else f"Công ty CP {selected_ticker}"
        p_tax = p_row.iloc[0].get('tax_code', 'N/A') if not p_row.empty else "N/A"
        p_ind = p_row.iloc[0].get('industry', 'N/A') if not p_row.empty else "N/A"
        p_web = p_row.iloc[0].get('website', 'N/A') if not p_row.empty else "N/A"

        # Hiển thị Card Thông tin
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
        
        # Chỉ số Tài chính Nổi bật (Metrics)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-container"><div class="metric-label">ROE (Tỷ suất LN/Vốn CS)</div><div class="metric-value">{latest.get("roe_pct", 0):.2f}%</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-container"><div class="metric-label">ROA (Tỷ suất LN/Tài sản)</div><div class="metric-value">{latest.get("roa_pct", 0):.2f}%</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-container"><div class="metric-label">Đòn bẩy (D/E Ratio)</div><div class="metric-value">{latest.get("debt_to_equity", 0):.2f}x</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-container"><div class="metric-label">Biên LN Ròng</div><div class="metric-value">{latest.get("net_margin_pct", 0):.2f}%</div></div>', unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)

        # Biểu đồ
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("<h4 style='color:#334155'>💰 Tăng Trưởng Doanh Thu & Lợi Nhuận</h4>", unsafe_allow_html=True)
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Bar(x=df_ticker['report_period'], y=df_ticker['net_revenue'], name="Doanh thu", marker_color="#bae6fd", hovertemplate="DT: %{y:,.0f} <extra></extra>"))
            fig_rev.add_trace(go.Scatter(x=df_ticker['report_period'], y=df_ticker['net_income'], name="LNST", yaxis="y2", line=dict(color="#0ea5e9", width=4), marker=dict(size=8, color="#0284c7")))
            fig_rev.update_layout(yaxis=dict(title="Doanh thu"), yaxis2=dict(title="LNST", overlaying="y", side="right"), showlegend=False, hovermode="x unified")
            st.plotly_chart(apply_custom_plotly_layout(fig_rev), use_container_width=True)

        with col_r:
            st.markdown("<h4 style='color:#334155'>📉 Lịch Sử Biên Lợi Nhuận & Hiệu Quả</h4>", unsafe_allow_html=True)
            fig_ratios = px.line(
                df_ticker, x="report_period", y=["roe_pct", "roa_pct", "gross_margin_pct", "net_margin_pct"],
                markers=True, color_discrete_sequence=["#1e3a8a", "#0ea5e9", "#10b981", "#f59e0b"]
            )
            fig_ratios.update_layout(legend_title_text="Chỉ số", hovermode="x unified")
            st.plotly_chart(apply_custom_plotly_layout(fig_ratios), use_container_width=True)

        # DuPont
        st.markdown("<div class='glass-card' style='padding:20px'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:0 0 10px 0; color:#0f172a;'>🧩 Phân Tích Mô Hình DuPont</h4>", unsafe_allow_html=True)
        st.latex(r"\text{ROE} = \text{ROA} \times \left(1 + \frac{\text{Debt}}{\text{Equity}}\right)")
        de_val = latest.get('debt_to_equity', 0) if pd.notnull(latest.get('debt_to_equity', 0)) else 0
        lev_factor = 1 + de_val
        st.markdown(f"<p style='color:#475569'>Tại kỳ <b>{latest['report_period']}</b>, tỷ suất ROE <b>{latest.get('roe_pct', 0):.2f}%</b> được cấu thành từ khả năng sinh lời trên tài sản (ROA: <b>{latest.get('roa_pct', 0):.2f}%</b>) nhân với đòn bẩy tài chính (<b>{lev_factor:.2f} lần</b>).</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Chuẩn bị dữ liệu ML chung
feature_cols = ['roe_pct', 'roa_pct', 'debt_to_equity', 'gross_margin_pct', 'net_margin_pct']
df_ml = df_raw.sort_values("report_period").groupby("ticker").last().reset_index()
df_ml_clean = df_ml.dropna(subset=feature_cols).copy()
scaled_features = StandardScaler().fit_transform(df_ml_clean[feature_cols]) if len(df_ml_clean) > 0 else None

# ----------------------------------------------------
# TAB 2 & 3: MACHINE LEARNING (PCA, KMEANS, ISOLATION FOREST)
# ----------------------------------------------------
with tab2:
    st.markdown("### 🤖 Phân Cụm Ngành & Doanh Nghiệp (K-Means)")
    if len(df_ml_clean) >= k_clusters and scaled_features is not None:
        df_ml_clean['Cluster'] = "Cụm " + KMeans(n_clusters=k_clusters, random_state=42, n_init=10).fit_predict(scaled_features).astype(str)
        pca_t = PCA(n_components=2).fit_transform(scaled_features)
        df_ml_clean['PCA_1'], df_ml_clean['PCA_2'] = pca_t[:, 0], pca_t[:, 1]
        
        fig_pca = px.scatter(
            df_ml_clean, x='PCA_1', y='PCA_2', color='Cluster', size='roe_pct',
            hover_name='ticker', hover_data=feature_cols,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(apply_custom_plotly_layout(fig_pca), use_container_width=True)

with tab3:
    st.markdown("### 🚨 Phát Hiện Dữ Liệu Bất Thường (Isolation Forest)")
    if len(df_ml_clean) > 5 and scaled_features is not None:
        df_ml_clean['Status'] = IsolationForest(contamination=contamination, random_state=42).fit_predict(scaled_features)
        df_ml_clean['Status'] = df_ml_clean['Status'].map({1: 'Bình thường', -1: 'Bất thường'})
        
        fig_ano = px.scatter(
            df_ml_clean, x='PCA_1', y='PCA_2', color='Status', hover_name='ticker',
            color_discrete_map={'Bình thường': '#cbd5e1', 'Bất thường': '#ef4444'}
        )
        fig_ano.update_traces(marker=dict(size=10, line=dict(width=1, color='White')))
        st.plotly_chart(apply_custom_plotly_layout(fig_ano), use_container_width=True)
        
        anomalies = df_ml_clean[df_ml_clean['Status'] == 'Bất thường']
        if not anomalies.empty:
            st.error(f"Phát hiện {len(anomalies)} doanh nghiệp có chỉ số tài chính dị biệt so với thị trường chung.")
            st.dataframe(anomalies[['ticker'] + feature_cols].style.background_gradient(cmap='Reds'), use_container_width=True)

# ----------------------------------------------------
# TAB 4: RAW DATA
# ----------------------------------------------------
with tab4:
    st.markdown("### 📋 Kho Dữ Liệu Thô (Gold Financial Ratios)")
    st.dataframe(df_raw[df_raw['ticker'] == selected_ticker] if selected_ticker else df_raw, use_container_width=True)

# ----------------------------------------------------
# TAB 5: PRICE & SHARPE
# ----------------------------------------------------
with tab5:
    st.markdown(f"### 📈 Lịch Sử Giá & Khẩu Vị Rủi Ro ({selected_ticker})")
    df_price = load_price_history(selected_ticker)
    
    if not df_price.empty and 'close' in df_price.columns:
        date_col = 'trading_date' if 'trading_date' in df_price.columns else df_price.columns[1]
        df_price = df_price.sort_values(date_col)
        df_price['daily_return'] = df_price['close'].pct_change()
        
        avg_rt, vol = df_price['daily_return'].mean() * 252, df_price['daily_return'].std() * np.sqrt(252)
        sharpe = (avg_rt - 0.045) / vol if vol > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='glass-card text-center'><b>Lợi Nhuận Kỳ Vọng (Năm)</b><br><span style='font-size:1.8rem; color:#059669'>{avg_rt*100:.2f}%</span></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='glass-card text-center'><b>Biến Động Giá (Volatility)</b><br><span style='font-size:1.8rem; color:#d97706'>{vol*100:.2f}%</span></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='glass-card text-center'><b>Hệ Số Sharpe Ratio</b><br><span style='font-size:1.8rem; color:#2563eb'>{sharpe:.2f}</span></div>", unsafe_allow_html=True)

        fig_p = px.area(df_price, x=date_col, y='close')
        fig_p.update_traces(line_color='#0ea5e9', fillcolor='rgba(14, 165, 233, 0.1)')
        st.plotly_chart(apply_custom_plotly_layout(fig_p), use_container_width=True)
    else:
        st.info("Chưa có dữ liệu biến động giá.")

# ----------------------------------------------------
# TAB 6: CREDIT RATING (AI)
# ----------------------------------------------------
with tab6:
    st.markdown(f"### 🏆 Mô Hình Đánh Giá Tín Dụng ({selected_ticker})")
    df_tick_latest = df_raw[df_raw['ticker'] == selected_ticker].sort_values("report_period")
    
    if not df_tick_latest.empty:
        last_rec = df_tick_latest.iloc[-1]
        de_ratio, roe_val = last_rec.get('debt_to_equity', 0), last_rec.get('roe_pct', 0)
        
        if de_ratio < 1.0 and roe_val > 15: rating, color, risk = "AAA", "#10b981", "Rủi ro cực thấp"
        elif de_ratio < 2.0 and roe_val > 10: rating, color, risk = "A+", "#3b82f6", "Rủi ro thấp"
        elif de_ratio < 3.5: rating, color, risk = "BBB", "#f59e0b", "Trung bình"
        else: rating, color, risk = "CCC", "#ef4444", "Rủi ro cao"
            
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}22 0%, #ffffff 100%); border-left: 6px solid {color}; padding: 30px; border-radius: 12px; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <h4 style="color:#64748b; margin:0">Điểm Tín Dụng Hiện Tại</h4>
                <h1 style="color:{color}; font-size:4rem; margin:10px 0">{rating}</h1>
                <h5 style="color:#334155">{risk}</h5>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown("#### Mức độ ảnh hưởng (Random Forest Feature Importance)")
            if len(df_ml_clean) > 10:
                X = df_ml_clean[feature_cols]
                y = ((df_ml_clean['roe_pct'] > 12) & (df_ml_clean['debt_to_equity'] < 2.0)).astype(int)
                clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
                
                df_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': clf.feature_importances_}).sort_values('Importance')
                fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
                st.plotly_chart(apply_custom_plotly_layout(fig_imp), use_container_width=True)
    else: st.info("Không đủ dữ liệu.")

# ----------------------------------------------------
# TAB 7: QUẢN TRỊ DOANH NGHIỆP
# ----------------------------------------------------
with tab7:
    st.markdown(f"### 👥 Cơ Cấu Cổ Đông & Ban Lãnh Đạo ({selected_ticker})")
    col_sh, col_of = st.columns([1, 1.2])
    
    with col_sh:
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#0f172a;'>🥧 Cơ Cấu Sở Hữu</h4>", unsafe_allow_html=True)
        df_sh = load_shareholders(selected_ticker)
        if not df_sh.empty:
            name_col = next((c for c in df_sh.columns if 'name' in c.lower() or 'shareholder' in c.lower()), df_sh.columns[1])
            pct_col = next((c for c in df_sh.columns if 'pct' in c.lower() or 'percent' in c.lower() or 'rate' in c.lower()), df_sh.columns[2])
            df_sh_clean = df_sh[df_sh[pct_col] > 0]
            
            if not df_sh_clean.empty:
                fig_sh = go.Figure(data=[go.Pie(labels=df_sh_clean[name_col], values=df_sh_clean[pct_col], hole=.4)])
                fig_sh.update_traces(marker=dict(colors=px.colors.qualitative.Prism))
                fig_sh.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_sh, use_container_width=True)
            st.dataframe(df_sh[[name_col, pct_col]], use_container_width=True, hide_index=True)
        else: st.info("Chưa có dữ liệu cổ đông.")
        st.markdown("</div>", unsafe_allow_html=True)
            
    with col_of:
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#0f172a;'>👔 Thành Viên Hội Đồng Quản Trị / Ban Giám Đốc</h4>", unsafe_allow_html=True)
        df_of = load_officers(selected_ticker)
        if not df_of.empty:
            disp_cols = [c for c in df_of.columns if c.lower() != 'ticker']
            st.dataframe(df_of[disp_cols], use_container_width=True, hide_index=True, height=500)
        else: st.info("Chưa có dữ liệu ban lãnh đạo.")
        st.markdown("</div>", unsafe_allow_html=True)