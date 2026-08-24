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
    .stApp { background-color: #f8fafc; color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    .metric-card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px 20px; box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.05); transition: transform 0.15s ease; }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.08); border-color: #cbd5e1; }
    .metric-label { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.75rem; font-weight: 800; color: #0f172a; margin: 4px 0; }
    .metric-badge-pos { display: inline-block; font-size: 0.75rem; font-weight: 600; color: #047857; background-color: #ecfdf5; padding: 2px 8px; border-radius: 9999px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid #e2e8f0; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; padding: 10px 18px; background-color: #f1f5f9; color: #475569; font-weight: 600; font-size: 0.9rem; border: none; }
    .stTabs [aria-selected="true"] { background-color: #1e3a8a !important; color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.divider()
if st.sidebar.button("🔄 Cập nhật dữ liệu mới"):
    st.cache_data.clear()
    st.rerun()

# 2. Dictionary Đa Ngôn Ngữ
T = {
    "VI": {
        "title": "VietFin Analytics — Nền Tảng Phân Tích Tài Chính & ML",
        "subtitle": "Hệ thống định lượng & Học máy tài chính tích hợp Cloud MotherDuck DW",
        "sidebar_header": "⚡ VietFin Quant Suite",
        "ticker_select": "🔍 Mã cổ phiếu phân tích",
        "tab1": "📊 Phân Tích & DuPont",
        "tab2": "🤖 Phân Cụm (K-Means/PCA)",
        "tab3": "🚨 Cảnh Báo Bất Thường",
        "tab4": "📋 Kho Dữ Liệu (OLAP)",
        "tab5": "🌍 Vĩ Mô, Tỷ Giá & Bản Đồ",
        "tab6": "📈 BCTC 5 Năm & Xếp Hạng Tín Dụng"
    }
}
lang = "VI"
txt = T[lang]


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


# 6. Sidebar
st.sidebar.title(txt["sidebar_header"])
tickers = sorted(df_raw['ticker'].dropna().unique())
selected_ticker = st.sidebar.selectbox(txt["ticker_select"], tickers, index=0)

st.title(txt["title"])
st.caption(txt["subtitle"])
st.divider()

# 7. Khởi tạo các Tabs

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([txt["tab1"], txt["tab2"], txt["tab3"], txt["tab4"], txt["tab5"], txt["tab6"]])

# --- Các Tab 1, 2, 3, 4 giữ nguyên logic cơ bản của bạn (rút gọn để tập trung Tab 5, 6) ---

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

# ----------------------------------------------------
# TAB 5: MACRO, FX, RATES & SPATIAL MAP
# ----------------------------------------------------
with tab5:
    st.markdown("### 🌍 Thông tin Vĩ mô, Tỷ giá & Hiệu suất rủi ro (Sharpe)")
    
    col_macro1, col_macro2 = st.columns(2)
    
    with col_macro1:
        st.markdown("#### 💱 Tỷ giá & Lãi suất Ngân hàng (Real-time Simulation)")
        # Bảng giả lập lãi suất và tỷ giá
        rates_data = pd.DataFrame({
            "Ngân hàng": ["Vietcombank", "BIDV", "Techcombank", "MBBank"],
            "Lãi suất Huy động (12T)": ["4.7%", "4.7%", "4.9%", "5.1%"],
            "Lãi suất Vay (Bình quân)": ["7.5%", "7.8%", "8.2%", "8.0%"]
        })
        st.table(rates_data)
        
        fx_data = pd.DataFrame({
            "Ngoại tệ": ["USD/VND", "EUR/VND", "JPY/VND"],
            "Mua vào": ["25,230", "27,150", "165.4"],
            "Bán ra": ["25,450", "27,800", "172.1"]
        })
        st.table(fx_data)
        
    with col_macro2:
        st.markdown("#### 📈 Biểu đồ Sharpe Ratio (Risk-Adjusted Return)")
        # Giả lập Sharpe Ratio = (Return - RiskFree) / Volatility
        df_ticker = df_raw[df_raw['ticker'] == selected_ticker].copy()
        df_ticker['Return'] = df_ticker['net_income'].pct_change().fillna(0)
        df_ticker['Sharpe'] = (df_ticker['Return'] - df_ticker['risk_free_rate']) / df_ticker['daily_returns_volatility']
        
        fig_sharpe = px.bar(
            df_ticker, x="report_period", y="Sharpe", 
            title=f"Sharpe Ratio của {selected_ticker} qua các kỳ",
            template="plotly_white", color="Sharpe", color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)

    st.divider()
    st.markdown(f"#### 📍 Vị trí địa lý Trụ sở doanh nghiệp ({selected_ticker})")
    # Định vị giả lập cho các doanh nghiệp
    location_dict = {
        "VNM": [10.7297, 106.7190], # Q7, HCM
        "FPT": [21.0278, 105.8342], # HN
        "VCB": [21.0250, 105.8520], # HN
        "HPG": [20.9400, 106.0600], # Hung Yen
        "VIC": [21.0333, 105.9220], # Long Bien, HN
    }
    coords = location_dict.get(selected_ticker, [10.7626, 106.6601])
    map_data = pd.DataFrame({'lat': [coords[0]], 'lon': [coords[1]]})
    st.map(map_data, zoom=12)


# ----------------------------------------------------
# TAB 6: 5-YEAR FINANCIALS & CREDIT RATING
# ----------------------------------------------------
with tab6:
    st.markdown(f"### 📋 Báo Cáo Tài Chính 5 Năm & Xếp Hạng Tín Dụng ({selected_ticker})")
    
    # 1. Bảng Dữ liệu BCTC 5 Năm (Giả lập theo template của bạn)
    st.markdown("#### Dữ liệu BCTC (Income Statement, Balance Sheet, Ratios)")
    
    fin_data = {
        "Chỉ tiêu": [
            "Revenue (Doanh thu)", "Gross Profit (LN Gộp)", "Operating income (LN HĐKD)", "Net profit after tax (LNST)",
            "Total Assets (Tổng tài sản)", "Current Liabilities (Nợ ngắn hạn)", "Long-Term Borrowings (Nợ dài hạn)", "Owner's equity (VCSH)",
            "ROE (%)", "ROA (%)", "Debt Ratio (Hệ số nợ)", "ICR (Khả năng trả lãi)"
        ],
        "202212": ["11,337,481", "1,311,290", "177,430", "74,597", "4,915,959", "2,397,494", "1,824,549", "1,123,894", "6.64", "1.52", "337.40", "1.80"],
        "202312": ["10,982,579", "1,756,675", "193,800", "31,457", "4,651,290", "2,580,830", "1,445,925", "1,124,488", "2.80", "0.66", "313.64", "1.39"],
        "202412": ["16,428,686", "2,193,979", "218,979", "114,149", "4,246,724", "3,055,431", "934,882", "1,191,293", "9.86", "2.57", "256.48", "2.15"],
        "202512": ["14,666,386", "2,647,966", "631,632", "441,826", "4,213,818", "2,621,504", "923,937", "1,592,314", "31.74", "10.44", "164.63", "7.57"]
    }
    df_fin5 = pd.DataFrame(fin_data)
    st.dataframe(df_fin5, use_container_width=True, hide_index=True)

    st.divider()

    # 2. Xếp Hạng Tín Dụng & Học Máy Thông Minh (Random Forest Feature Importance)
    col_credit1, col_credit2 = st.columns([1, 1])
    
    with col_credit1:
        st.markdown("#### 🏆 Xếp Hạng Tín Dụng Doanh Nghiệp (Credit Trend)")
        
        # Logic tính điểm tín dụng (Scoring Model)
        def get_credit_rating(icr, debt_ebitda):
            if icr > 5 and debt_ebitda < 2: return "AAA", "#16a34a"
            elif icr > 3 and debt_ebitda < 4: return "A+", "#2563eb"
            elif icr > 1.5 and debt_ebitda < 6: return "BBB", "#d97706"
            else: return "CCC", "#dc2626"
            
        # Dữ liệu giả định năm gần nhất cho model
        current_icr = 7.57
        current_debt_ebitda = 1.46
        rating, color = get_credit_rating(current_icr, current_debt_ebitda)
        
        st.markdown(f"""
        <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; border: 2px solid {color}; text-align: center;">
            <h3 style="color: #64748b; margin: 0;">Mức Xếp Hạng Hiện Tại (2025)</h3>
            <h1 style="color: {color}; font-size: 3.5rem; margin: 10px 0;">{rating}</h1>
            <p style="color: #475569; font-size: 1rem;">Mức độ rủi ro tín dụng: <b>An toàn cao</b></p>
            <hr style="border-top: 1px solid #e2e8f0;"/>
            <p>Khả năng thanh toán lãi vay (ICR): <b>{current_icr}x</b> | Total Debt / EBITDA: <b>{current_debt_ebitda}x</b></p>
        </div>
        """, unsafe_allow_html=True)

    with col_credit2:
        st.markdown("#### 🧠 ML Model: Đánh giá Tầm quan trọng của Biến (Feature Importance)")
        st.caption("Ứng dụng Random Forest Classifier để xác định biến tài chính nào ảnh hưởng lớn nhất đến Xếp hạng Tín dụng (Investment Grade).")
        
        # Khởi tạo thuật toán học máy phân loại
        features = df_raw[['roe_pct', 'roa_pct', 'debt_to_equity', 'gross_margin_pct', 'net_margin_pct']].dropna()
        # Tạo nhãn mục tiêu giả (Target) - Doanh nghiệp Tốt (1) vs Xấu (0) dựa vào ROE > 15 và Debt < 1.5
        target = ((features['roe_pct'] > 15) & (features['debt_to_equity'] < 1.5)).astype(int)
        
        if len(features) > 10:
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(features, target)
            
            importances = rf_model.feature_importances_
            feature_names = features.columns
            df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=True)
            
            fig_rf = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                            template='plotly_white', color='Importance', color_continuous_scale="Blues")
            fig_rf.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=250)
            st.plotly_chart(fig_rf, use_container_width=True)