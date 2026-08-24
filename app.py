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

# 1. Cấu hình giao diện chuẩn Financial Terminal (Dark Mode)
st.set_page_config(
    page_title="VietFin Intelligence | Quant & ML Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Institutional Dark Theme CSS */
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 0.75rem; text-transform: uppercase; color: #9ca3af; font-weight: 600; letter-spacing: 0.05em; }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: #38bdf8; margin-top: 2px; }
    .metric-sub { font-size: 0.75rem; color: #10b981; font-weight: 500; }
    .metric-sub-neg { font-size: 0.75rem; color: #ef4444; font-weight: 500; }

    /* Custom Streamlit Tabs UI */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #374151; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        background-color: #1f2937;
        color: #9ca3af;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. Kết nối MotherDuck DW Engine
load_dotenv()
token = os.getenv("MOTHERDUCK_TOKEN")

if not token:
    st.error("❌ Không tìm thấy MOTHERDUCK_TOKEN trong cấu hình file .env hoặc Secrets!")
    st.stop()

@st.cache_resource
def get_connection():
    return duckdb.connect(f"md:vietfin_db?token={token}")

con = get_connection()

@st.cache_data(ttl=3600)
def load_gold_data():
    """Tải và tối ưu hóa toàn bộ dữ liệu Gold từ DW bằng DuckDB OLAP Engine"""
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

# 3. Thanh điều hướng Sidebar
if os.path.exists("logo.jpg"):
    st.sidebar.image("logo.jpg", width=120)

st.sidebar.title("⚡ VietFin Quant Suite")
st.sidebar.caption("Big Data Architecture & Financial Analytics")

tickers = sorted(df_raw['ticker'].dropna().unique())
selected_ticker = st.sidebar.selectbox("🔍 Mã cổ phiếu phân tích", tickers, index=tickers.index("VNM") if "VNM" in tickers else 0)

st.sidebar.divider()
st.sidebar.subheader("🤖 Tham số Học Máy (ML)")
k_clusters = st.sidebar.slider("Số cụm K-Means (k)", min_value=2, max_value=6, value=3)
contamination = st.sidebar.slider("Ngưỡng phát hiện bất thường (%)", min_value=1, max_value=15, value=5) / 100.0

# Header
st.title("📈 VietFin Analytics — Financial Intelligence Dashboard")
st.caption("Nền tảng phân tích tài chính định lượng & ML tích hợp kho dữ liệu Cloud MotherDuck (HOSE/HNX/UPCOM)")

# 4. Các Tab Chức Năng Chuyên Sâu
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Phân Tích Chuyên Sâu Doanh Nghiệp",
    "🤖 Học Máy: Phân Cụm (K-Means & PCA)",
    "🚨 Phát Hiện Biến Động Bất Thường (Outliers)",
    "📋 Kho Dữ Liệu Tầng Gold (Big Data Console)"
])

# ----------------------------------------------------
# TAB 1: FINANCIAL DEEP-DIVE & DUPONT ANALYSIS
# ----------------------------------------------------
with tab1:
    df_ticker = df_raw[df_raw['ticker'] == selected_ticker].sort_values("report_period").copy()
    
    if not df_ticker.empty:
        # Tính toán YoY growth rate
        df_ticker['rev_growth'] = df_ticker['net_revenue'].pct_change() * 100
        df_ticker['income_growth'] = df_ticker['net_income'].pct_change() * 100
        latest = df_ticker.iloc[-1]
        
        # Financial Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">ROE (Tỷ suất sinh lời VCSH)</div>
                    <div class="metric-val">{latest["roe_pct"]:.2f}%</div>
                    <div class="metric-sub">Hiệu quả sử dụng vốn</div>
                </div>
            ''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">ROA (Tỷ suất sinh lời Tài sản)</div>
                    <div class="metric-val">{latest["roa_pct"]:.2f}%</div>
                    <div class="metric-sub">Hiệu quả quản trị tài sản</div>
                </div>
            ''', unsafe_allow_html=True)
        with c3:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Đòn bẩy Tài chính (D/E)</div>
                    <div class="metric-val">{latest["debt_to_equity"]:.2f}x</div>
                    <div class="metric-sub">Mức độ rủi ro cấu trúc nợ</div>
                </div>
            ''', unsafe_allow_html=True)
        with c4:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Biên Lợi Nhuận Ròng</div>
                    <div class="metric-val">{latest["net_margin_pct"]:.2f}%</div>
                    <div class="metric-sub">Khả năng sinh lời thuần</div>
                </div>
            ''', unsafe_allow_html=True)

        st.divider()

        # Visual Plots
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 💰 Doanh Thu & Lợi Nhuận Sau Thuế")
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Bar(x=df_ticker['report_period'], y=df_ticker['net_revenue'], name="Doanh Thu Thuần", marker_color="#0284c7"))
            fig_rev.add_trace(go.Scatter(x=df_ticker['report_period'], y=df_ticker['net_income'], name="LNST", yaxis="y2", line=dict(color="#34d399", width=3)))
            
            fig_rev.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Doanh Thu (VND)"),
                yaxis2=dict(title="LNST (VND)", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_rev, use_container_width=True)

        with col_right:
            st.markdown("#### 📉 Phân Tích Xu Hướng Chỉ Số Sinh Lời")
            fig_ratios = px.line(
                df_ticker, x="report_period", y=["roe_pct", "roa_pct", "gross_margin_pct", "net_margin_pct"],
                markers=True, template="plotly_dark",
                labels={"value": "Phần trăm (%)", "report_period": "Kỳ báo cáo"}
            )
            fig_ratios.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_ratios, use_container_width=True)

        # Mô hình Phân tích DuPont 2 Nhân Tố
        st.markdown("#### 🧮 Mô Hình Phân Tách DuPont Analysis")
        st.latex(r"ROE = ROA \times \left(1 + \frac{\text{Debt}}{\text{Equity}}\right)")
        
        leverage_factor = 1 + (latest['debt_to_equity'] if pd.notnull(latest['debt_to_equity']) else 0)
        calculated_roe = (latest['roa_pct'] if pd.notnull(latest['roa_pct']) else 0) * leverage_factor
        
        st.info(f"**Giải thích DuPont cho {selected_ticker}:** ROE thực tế đạt **{latest['roe_pct']:.2f}%**, được tạo thành từ ROA nền tảng (**{latest['roa_pct']:.2f}%**) khuếch đại qua Hệ số Đòn bẩy Tài chính (**{leverage_factor:.2f}x**).")
    else:
        st.warning("Chưa có dữ liệu tài chính cho mã cổ phiếu này.")

# ----------------------------------------------------
# TAB 2: UNSUPERVISED MACHINE LEARNING (K-MEANS & PCA)
# ----------------------------------------------------
with tab2:
    st.markdown("### 🤖 Không Gian Giảm Chiều PCA & Phân Cụm K-Means")
    st.caption("Thuật toán mã hóa các chỉ số đa chiều (ROE, ROA, D/E, Gross Margin, Net Margin) thành không gian 2D để xác định cấu trúc nhóm cổ phiếu tương đồng.")

    feature_cols = ['roe_pct', 'roa_pct', 'debt_to_equity', 'gross_margin_pct', 'net_margin_pct']
    
    # Lấy bản ghi báo cáo mới nhất của từng doanh nghiệp
    df_ml = df_raw.sort_values("report_period").groupby("ticker").last().reset_index()
    df_ml_clean = df_ml.dropna(subset=feature_cols).copy()

    if len(df_ml_clean) >= k_clusters:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_ml_clean[feature_cols])

        # K-Means Clustering
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        df_ml_clean['Cluster'] = kmeans.fit_predict(scaled_features)
        df_ml_clean['Cluster'] = "Cụm " + df_ml_clean['Cluster'].astype(str)

        # PCA 2D Dimensionality Reduction
        pca = PCA(n_components=2)
        pca_transformed = pca.fit_transform(scaled_features)
        df_ml_clean['PCA_1'] = pca_transformed[:, 0]
        df_ml_clean['PCA_2'] = pca_transformed[:, 1]
        
        var_exp = pca.explained_variance_ratio_ * 100

        # Plot PCA Scatter Chart
        fig_pca = px.scatter(
            df_ml_clean, x='PCA_1', y='PCA_2', color='Cluster',
            hover_name='ticker', hover_data=feature_cols,
            labels={'PCA_1': f'Thành phần PCA 1 ({var_exp[0]:.1f}%)', 'PCA_2': f'Thành phần PCA 2 ({var_exp[1]:.1f}%)'},
            template="plotly_dark", title=f"Bản Đồ Phân Cụm Cổ Phiếu Toàn Thị Trường (K={k_clusters})"
        )
        fig_pca.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='White')))
        fig_pca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pca, use_container_width=True)

        # Bảng đặc tính kỹ thuật từng cụm
        st.markdown("#### 📋 Đặc Tính Chỉ Số Trung Bình Từng Cụm ML")
        cluster_profile = df_ml_clean.groupby('Cluster')[feature_cols].mean().reset_index()
        st.dataframe(cluster_profile.style.highlight_max(axis=0, color="#0284c7"), width="stretch")
    else:
        st.warning("Dữ liệu không đủ để thực hiện thuật toán phân cụm.")

# ----------------------------------------------------
# TAB 3: ANOMALY DETECTION (ISOLATION FOREST)
# ----------------------------------------------------
with tab3:
    st.markdown("### 🚨 Phát Hiện Điểm Bất Thường Báo Cáo Tài Chính")
    st.caption("Ứng dụng mô hình **Isolation Forest** nhằm tự động tầm soát các cổ phiếu có cấu trúc chỉ số tài chính đột biến/bất thường so với toàn thị trường.")

    if len(df_ml_clean) > 10:
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        df_ml_clean['Anomaly_Score'] = iso_forest.fit_predict(scaled_features)
        
        # -1 indicates anomaly, 1 indicates normal
        df_anomalies = df_ml_clean[df_ml_clean['Anomaly_Score'] == -1]
        
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            st.metric("Tổng Số Cổ Phiếu Tầm Soát", len(df_ml_clean))
            st.metric("Số Cổ Phiếu Bất Thường Phát Hiện", len(df_anomalies), delta=f"{len(df_anomalies)/len(df_ml_clean)*100:.1f}%", delta_color="inverse")
        
        with col_m2:
            fig_anomaly = px.scatter(
                df_ml_clean, x='roe_pct', y='debt_to_equity',
                color=df_ml_clean['Anomaly_Score'].map({1: 'Bình thường', -1: 'Bất thường'}),
                color_discrete_map={'Bình thường': '#0284c7', 'Bất thường': '#ef4444'},
                hover_name='ticker', hover_data=['net_margin_pct'],
                template="plotly_dark", title="Phân Phối Bất Thường: ROE vs. Đòn Bẩy D/E"
            )
            fig_anomaly.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_anomaly, use_container_width=True)

        st.markdown("#### ⚠️ Danh Sách Mã Cổ Phiếu Có Chỉ Số Bất Thường")
        st.dataframe(
            df_anomalies[['ticker', 'roe_pct', 'roa_pct', 'debt_to_equity', 'gross_margin_pct', 'net_margin_pct']],
            width="stretch"
        )

# ----------------------------------------------------
# TAB 4: DATA WAREHOUSE BIG DATA CONSOLE
# ----------------------------------------------------
with tab4:
    st.markdown("### 📋 Dữ Liệu Tầng Gold (MotherDuck Cloud Data Warehouse)")
    st.caption("Truy xuất dữ liệu trực tiếp bằng kho dữ liệu DuckDB OLAP Engine.")
    
    # Filter Controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        search_ticker = st.multiselect("Lọc theo mã cổ phiếu", tickers, default=[selected_ticker])
    
    df_filtered = df_raw[df_raw['ticker'].isin(search_ticker)] if search_ticker else df_raw
    
    st.dataframe(df_filtered, width="stretch")
    st.caption(f"Tổng số bản ghi truy xuất: **{len(df_filtered):,}** dòng dữ liệu.")