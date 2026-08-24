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
from sklearn.ensemble import IsolationForest

# 1. Cấu hình Trang & Injected CSS (Institutional Light Theme)
st.set_page_config(
    page_title="VietFin Intelligence | Financial Analytics Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Professional Light Theme CSS
st.markdown("""
<style>
    /* Main Background & Base Typography */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Financial Metric Cards (White Slate Style) */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.05), 0 1px 2px -1px rgba(15, 23, 42, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.08);
        border-color: #cbd5e1;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a;
        margin: 4px 0;
    }
    .metric-badge-pos {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        color: #047857;
        background-color: #ecfdf5;
        padding: 2px 8px;
        border-radius: 9999px;
    }

    /* Tabs Layout Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e2e8f0;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 10px 18px;
        background-color: #f1f5f9;
        color: #475569;
        font-weight: 600;
        font-size: 0.9rem;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a8a !important;
        color: #ffffff !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Thêm nút bấm clear cache trên Sidebar app.py
st.sidebar.divider()
if st.sidebar.button("🔄 Cập nhật dữ liệu mới"):
    st.cache_data.clear()
    st.rerun()

# 2. Dictionary Đa Ngôn Ngữ (Bilingual Dictionary)
T = {
    "VI": {
        "title": "VietFin Analytics — Nền Tảng Phân Tích Tài Chính & ML",
        "subtitle": "Hệ thống định lượng & Học máy tài chính tích hợp Cloud MotherDuck DW (HOSE, HNX, UPCOM)",
        "sidebar_header": "⚡ VietFin Quant Suite",
        "sidebar_sub": "Định Lượng & Kiến Trúc Dữ Liệu",
        "lang_select": "🌐 Ngôn ngữ / Language",
        "ticker_select": "🔍 Mã cổ phiếu phân tích",
        "ml_settings": "⚙️ Cấu hình Học Máy (ML)",
        "k_clusters": "Số cụm K-Means (k)",
        "anomaly_thresh": "Ngưỡng lọc bất thường (%)",
        "tab1": "📊 Phân Tích Doanh Nghiệp & DuPont",
        "tab2": "🤖 Học Máy: Phân Cụm (K-Means & PCA)",
        "tab3": "🚨 Tầm Soát Biến Động Bất Thường",
        "tab4": "📋 Kho Dữ Liệu Tầng Gold (OLAP)",
        "roe_label": "ROE (Tỷ suất LN/VCSH)",
        "roa_label": "ROA (Tỷ suất LN/Tài sản)",
        "de_label": "Đòn Bẩy D/E (Nợ/VCSH)",
        "margin_label": "Biên Lợi Nhuận Ròng",
        "dupont_title": "🧮 Mô Hình Phân Tách DuPont (2 Nhân Tố)",
        "dupont_desc": "ROE thực tế đạt **{roe:.2f}%**, tạo thành từ ROA nền tảng (**{roa:.2f}%**) khuếch đại qua Đòn bẩy Tài chính (**{lev:.2f}x**).",
        "chart_rev": "Doanh Thu Thuần & Lợi Nhuận Sau Thuế",
        "chart_ratios": "Xu Hướng Các Chỉ Số Sinh Lời",
        "pca_title": "Bản Đồ Phân Cụm Cổ Phiếu Toàn Thị Trường (PCA & K-Means)",
        "cluster_profile": "📋 Đặc Tính Trung Bình Theo Cụm",
        "anomaly_title": "Tầm Soát Cổ Phiếu Có Chỉ Số Tài Chính Đột Biến",
        "anomaly_count": "Số cổ phiếu phát hiện bất thường",
        "data_console_title": "📋 Dữ Liệu Tầng Gold (MotherDuck Cloud DW)",
        "filter_ticker": "Lọc theo danh sách mã",
        "total_records": "Tổng số bản ghi:"
    },
    "EN": {
        "title": "VietFin Analytics — Quant Financial & ML Platform",
        "subtitle": "Quantitative Financial Analytics & Machine Learning Platform powered by MotherDuck Cloud DW",
        "sidebar_header": "⚡ VietFin Quant Suite",
        "sidebar_sub": "Quant Architecture & Data Science",
        "lang_select": "🌐 Language / Ngôn ngữ",
        "ticker_select": "🔍 Select Ticker Symbol",
        "ml_settings": "⚙️ Machine Learning Config",
        "k_clusters": "K-Means Clusters (k)",
        "anomaly_thresh": "Anomaly Threshold (%)",
        "tab1": "📊 Corporate Deep-Dive & DuPont",
        "tab2": "🤖 ML: Clustering (K-Means & PCA)",
        "tab3": "🚨 Financial Anomaly Detection",
        "tab4": "📋 Gold Layer Data Warehouse",
        "roe_label": "ROE (Return on Equity)",
        "roa_label": "ROA (Return on Assets)",
        "de_label": "Financial Leverage (D/E)",
        "margin_label": "Net Profit Margin",
        "dupont_title": "🧮 DuPont Analysis Model (2-Factor)",
        "dupont_desc": "Actual ROE reaches **{roe:.2f}%**, driven by baseline ROA (**{roa:.2f}%**) amplified through Financial Leverage (**{lev:.2f}x**).",
        "chart_rev": "Net Revenue & Net Income Trend",
        "chart_ratios": "Profitability Ratios Trend",
        "pca_title": "Market-Wide Stock Clustering Map (PCA & K-Means)",
        "cluster_profile": "📋 Cluster Feature Profile",
        "anomaly_title": "Financial Outlier & Anomaly Screener",
        "anomaly_count": "Anomalies Detected",
        "data_console_title": "📋 Gold Layer Warehouse Console (MotherDuck DW)",
        "filter_ticker": "Filter by ticker list",
        "total_records": "Total Records Returned:"
    }
}

# 3. Kết nối Cloud DW MotherDuck
load_dotenv()
token = os.getenv("MOTHERDUCK_TOKEN")

if not token:
    st.error("❌ MOTHERDUCK_TOKEN not found in environment settings!")
    st.stop()

@st.cache_resource
def get_connection():
    return duckdb.connect(f"md:vietfin_db?token={token}")

con = get_connection()

@st.cache_data(ttl=3600)
def load_gold_data():
    return con.execute("""
        SELECT 
            ticker, 
            report_period, 
            net_revenue, 
            net_income, 
            roe_pct, 
            roa_pct, 
            debt_to_equity, 
            gross_margin_pct,
            net_margin_pct
        FROM gold_financial_ratios
    """).df()

df_raw = load_gold_data()

# 4. Thanh Điều Hướng (Sidebar)
if os.path.exists("logo.jpg"):
    st.sidebar.image("logo.jpg", width=110)

# Language Selector
lang_choice = st.sidebar.selectbox("🌐 Language / Ngôn ngữ", ["Tiếng Việt", "English"], index=0)
lang = "VI" if lang_choice == "Tiếng Việt" else "EN"
txt = T[lang]

st.sidebar.title(txt["sidebar_header"])
st.sidebar.caption(txt["sidebar_sub"])
st.sidebar.divider()

tickers = sorted(df_raw['ticker'].dropna().unique())
selected_ticker = st.sidebar.selectbox(txt["ticker_select"], tickers, index=tickers.index("VNM") if "VNM" in tickers else 0)

st.sidebar.subheader(txt["ml_settings"])
k_clusters = st.sidebar.slider(txt["k_clusters"], min_value=2, max_value=6, value=3)
contamination = st.sidebar.slider(txt["anomaly_thresh"], min_value=1, max_value=15, value=5) / 100.0

# Header
st.title(txt["title"])
st.caption(txt["subtitle"])
st.divider()

# 5. Các Tab Nội Dung Chính
tab1, tab2, tab3, tab4 = st.tabs([txt["tab1"], txt["tab2"], txt["tab3"], txt["tab4"]])

# ----------------------------------------------------
# TAB 1: CORPORATE DEEP-DIVE & DUPONT ANALYSIS
# ----------------------------------------------------
with tab1:
    df_ticker = df_raw[df_raw['ticker'] == selected_ticker].sort_values("report_period").copy()
    
    if not df_ticker.empty:
        latest = df_ticker.iloc[-1]
        
        # Financial Cards (High-Contrast White Design)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">{txt["roe_label"]}</div>
                    <div class="metric-value">{latest["roe_pct"]:.2f}%</div>
                    <div class="metric-badge-pos">Profitability</div>
                </div>
            ''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">{txt["roa_label"]}</div>
                    <div class="metric-value">{latest["roa_pct"]:.2f}%</div>
                    <div class="metric-badge-pos">Efficiency</div>
                </div>
            ''', unsafe_allow_html=True)
        with c3:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">{txt["de_label"]}</div>
                    <div class="metric-value">{latest["debt_to_equity"]:.2f}x</div>
                    <div class="metric-badge-pos">Capital Structure</div>
                </div>
            ''', unsafe_allow_html=True)
        with c4:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">{txt["margin_label"]}</div>
                    <div class="metric-value">{latest["net_margin_pct"]:.2f}%</div>
                    <div class="metric-badge-pos">Margin</div>
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plotly Charts (Plotly White Theme)
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown(f"#### 💰 {txt['chart_rev']}")
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Bar(x=df_ticker['report_period'], y=df_ticker['net_revenue'], name="Revenue", marker_color="#1e3a8a"))
            fig_rev.add_trace(go.Scatter(x=df_ticker['report_period'], y=df_ticker['net_income'], name="Net Income", yaxis="y2", line=dict(color="#059669", width=3)))
            
            fig_rev.update_layout(
                template="plotly_white",
                yaxis=dict(title="Revenue (VND)"),
                yaxis2=dict(title="Net Income (VND)", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_rev, use_container_width=True)

        with col_right:
            st.markdown(f"#### 📉 {txt['chart_ratios']}")
            fig_ratios = px.line(
                df_ticker, x="report_period", y=["roe_pct", "roa_pct", "gross_margin_pct", "net_margin_pct"],
                markers=True, template="plotly_white",
                color_discrete_sequence=["#1e3a8a", "#059669", "#d97706", "#2563eb"]
            )
            fig_ratios.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_ratios, use_container_width=True)

        # DuPont Model
        st.markdown(f"#### {txt['dupont_title']}")
        st.latex(r"\text{ROE} = \text{ROA} \times \left(1 + \frac{\text{Debt}}{\text{Equity}}\right)")
        
        leverage_factor = 1 + (latest['debt_to_equity'] if pd.notnull(latest['debt_to_equity']) else 0)
        st.info(txt['dupont_desc'].format(
            roe=latest['roe_pct'] if pd.notnull(latest['roe_pct']) else 0,
            roa=latest['roa_pct'] if pd.notnull(latest['roa_pct']) else 0,
            lev=leverage_factor
        ))
    else:
        st.warning("No financial data available for this ticker.")

# ----------------------------------------------------
# TAB 2: UNSUPERVISED ML (K-MEANS & PCA)
# ----------------------------------------------------
with tab2:
    st.markdown(f"### 🤖 {txt['pca_title']}")

    feature_cols = ['roe_pct', 'roa_pct', 'debt_to_equity', 'gross_margin_pct', 'net_margin_pct']
    df_ml = df_raw.sort_values("report_period").groupby("ticker").last().reset_index()
    df_ml_clean = df_ml.dropna(subset=feature_cols).copy()

    if len(df_ml_clean) >= k_clusters:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_ml_clean[feature_cols])

        # K-Means
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        df_ml_clean['Cluster'] = "Cluster " + kmeans.fit_predict(scaled_features).astype(str)

        # PCA 2D
        pca = PCA(n_components=2)
        pca_transformed = pca.fit_transform(scaled_features)
        df_ml_clean['PCA_1'] = pca_transformed[:, 0]
        df_ml_clean['PCA_2'] = pca_transformed[:, 1]
        var_exp = pca.explained_variance_ratio_ * 100

        fig_pca = px.scatter(
            df_ml_clean, x='PCA_1', y='PCA_2', color='Cluster',
            hover_name='ticker', hover_data=feature_cols,
            labels={'PCA_1': f'PCA Component 1 ({var_exp[0]:.1f}%)', 'PCA_2': f'PCA Component 2 ({var_exp[1]:.1f}%)'},
            template="plotly_white", color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_pca.update_traces(marker=dict(size=11, opacity=0.85))
        st.plotly_chart(fig_pca, use_container_width=True)

        st.markdown(f"#### {txt['cluster_profile']}")
        cluster_profile = df_ml_clean.groupby('Cluster')[feature_cols].mean().reset_index()
        st.dataframe(cluster_profile.style.highlight_max(axis=0, color="#dbeafe"), width="stretch")

# ----------------------------------------------------
# TAB 3: ANOMALY DETECTION (ISOLATION FOREST)
# ----------------------------------------------------
with tab3:
    st.markdown(f"### 🚨 {txt['anomaly_title']}")

    if len(df_ml_clean) > 10:
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        df_ml_clean['Anomaly_Score'] = iso_forest.fit_predict(scaled_features)
        df_anomalies = df_ml_clean[df_ml_clean['Anomaly_Score'] == -1]
        
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            st.metric("Total Screened Stocks", len(df_ml_clean))
            st.metric(txt["anomaly_count"], len(df_anomalies), delta=f"{len(df_anomalies)/len(df_ml_clean)*100:.1f}%", delta_color="inverse")
        
        with col_m2:
            fig_anomaly = px.scatter(
                df_ml_clean, x='roe_pct', y='debt_to_equity',
                color=df_ml_clean['Anomaly_Score'].map({1: 'Normal', -1: 'Anomaly'}),
                color_discrete_map={'Normal': '#1e3a8a', 'Anomaly': '#dc2626'},
                hover_name='ticker', hover_data=['net_margin_pct'],
                template="plotly_white"
            )
            st.plotly_chart(fig_anomaly, use_container_width=True)

        st.dataframe(
            df_anomalies[['ticker', 'roe_pct', 'roa_pct', 'debt_to_equity', 'gross_margin_pct', 'net_margin_pct']],
            width="stretch"
        )

# ----------------------------------------------------
# TAB 4: DATA WAREHOUSE CONSOLE
# ----------------------------------------------------
with tab4:
    st.markdown(f"### {txt['data_console_title']}")
    
    col_f1, _ = st.columns(2)
    with col_f1:
        search_ticker = st.multiselect(txt["filter_ticker"], tickers, default=[selected_ticker])
    
    df_filtered = df_raw[df_raw['ticker'].isin(search_ticker)] if search_ticker else df_raw
    st.dataframe(df_filtered, width="stretch")
    st.caption(f"{txt['total_records']} **{len(df_filtered):,}**")