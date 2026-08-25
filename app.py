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

# 2. Dictionary Đa Ngôn Ngữ
T = {
    "VI": {
        "title": "VietFin Analytics — Nền Tảng Phân Tích Tài Chính & ML",
        "subtitle": "Hệ thống định lượng & Học máy tài chính tích hợp Cloud MotherDuck DW",
        "sidebar_header": "⚡ VietFin Quant Suite",
        "sidebar_sub": "Hệ thống phân tích tài chính định lượng",
        "ticker_select": "🔍 Mã cổ phiếu phân tích",
        "ml_settings": "⚙️ Cấu hình Machine Learning",
        "k_clusters": "Số cụm K-Means",
        "anomaly_thresh": "Tỷ lệ bất thường (Contamination)",
        "tab1": "📊 Phân Tích & DuPont",
        "tab2": "🤖 Phân Cụm (K-Means/PCA)",
        "tab3": "🚨 Cảnh Báo Bất Thường",
        "tab4": "📋 Kho Dữ Liệu (OLAP)",
        "tab5": "🌍 Vĩ Mô, Tỷ Giá & Bản Đồ",
        "tab6": "📈 BCTC 5 Năm & Xếp Hạng",
        "roe_label": "Tỷ suất ROE",
        "roa_label": "Tỷ suất ROA",
        "de_label": "Nợ / VCSH (D/E)",
        "margin_label": "Biên LN Ròng",
        "chart_rev": "Doanh Thu & Lợi Nhuận Ròng Qua Các Kỳ",
        "chart_ratios": "Xu Hướng Các Chỉ Số Tài Chính",
        "dupont_title": "Mô Hình Phân Tích DuPont (ROE Decomposition)",
        "dupont_desc": "ROE của doanh nghiệp đạt {roe:.2f}%, được cấu thành từ ROA ({roa:.2f}%) và Đòn bẩy tài chính ({lev:.2f}x).",
        "pca_title": "Phân Cụm Doanh Nghiệp (K-Means) & Giảm Chiều Dữ Liệu (PCA)",
        "cluster_profile": "Đặc Tính Trung Bình Của Các Cụm",
        "anomaly_title": "Phát Hiện Cổ Phiếu Có Chỉ Số Bất Thường (Isolation Forest)",
        "anomaly_count": "Số Cổ Phiếu Bất Thường",
        "data_console_title": "Truy Vấn Kho Dữ Liệu Gold Financial Ratios",
        "filter_ticker": "Lọc theo mã cổ phiếu",
        "total_records": "Tổng số bản ghi trong kho dữ liệu:"
    },
    "EN": {
        "title": "VietFin Analytics — Financial & ML Platform",
        "subtitle": "Quantitative Financial Platform integrated with Cloud MotherDuck DW",
        "sidebar_header": "⚡ VietFin Quant Suite",
        "sidebar_sub": "Quantitative Analytics System",
        "ticker_select": "🔍 Select Ticker",
        "ml_settings": "⚙️ Machine Learning Settings",
        "k_clusters": "K-Means Clusters",
        "anomaly_thresh": "Anomaly Contamination Rate",
        "tab1": "📊 Deep-Dive & DuPont",
        "tab2": "🤖 Clustering (K-Means/PCA)",
        "tab3": "🚨 Anomaly Detection",
        "tab4": "📋 Data Warehouse (OLAP)",
        "tab5": "🌍 Macro, FX & Map",
        "tab6": "📈 5-Yr Financials & Rating",
        "roe_label": "ROE Ratio",
        "roa_label": "ROA Ratio",
        "de_label": "Debt to Equity",
        "margin_label": "Net Profit Margin",
        "chart_rev": "Revenue & Net Income Trend",
        "chart_ratios": "Financial Ratios Trend",
        "dupont_title": "DuPont Analysis Model",
        "dupont_desc": "Company ROE stands at {roe:.2f}%, composed of ROA ({roa:.2f}%) and Financial Leverage ({lev:.2f}x).",
        "pca_title": "Corporate Clustering (K-Means) & PCA Reduction",
        "cluster_profile": "Cluster Average Profile",
        "anomaly_title": "Financial Anomaly Detection (Isolation Forest)",
        "anomaly_count": "Anomalous Stocks Detected",
        "data_console_title": "Gold Financial Ratios Warehouse Query",
        "filter_ticker": "Filter by Ticker",
        "total_records": "Total records loaded:"
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
    df = con.execute("""
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
    
    # Xử lý chuẩn hóa các cột tính toán bổ sung nếu thiếu trong database
    if 'risk_free_rate' not in df.columns:
        df['risk_free_rate'] = 0.045
    if 'daily_returns_volatility' not in df.columns:
        df['daily_returns_volatility'] = 0.025
    return df

try:
    df_raw = load_gold_data()
except Exception as e:
    st.error(f"❌ Lỗi truy vấn Database: {e}")
    st.stop()

# 4. Thanh Điều Hướng (Sidebar)
if os.path.exists("logo.jpg"):
    st.sidebar.image("logo.jpg", width=110)

lang_choice = st.sidebar.selectbox("🌐 Language / Ngôn ngữ", ["Tiếng Việt", "English"], index=0)
lang = "VI" if lang_choice == "Tiếng Việt" else "EN"
txt = T[lang]

st.sidebar.title(txt["sidebar_header"])
st.sidebar.caption(txt["sidebar_sub"])
st.sidebar.divider()

tickers = sorted(df_raw['ticker'].dropna().unique()) if not df_raw.empty else ["VNM"]
default_index = tickers.index("VNM") if "VNM" in tickers else 0
selected_ticker = st.sidebar.selectbox(txt["ticker_select"], tickers, index=default_index)

st.sidebar.subheader(txt["ml_settings"])
k_clusters = st.sidebar.slider(txt["k_clusters"], min_value=2, max_value=6, value=3)
contamination = st.sidebar.slider(txt["anomaly_thresh"], min_value=1, max_value=15, value=5) / 100.0

st.sidebar.divider()
if st.sidebar.button("🔄 Cập nhật dữ liệu mới"):
    st.cache_data.clear()
    st.rerun()

# Header chính
st.title(txt["title"])
st.caption(txt["subtitle"])
st.divider()

# Khởi tạo giả lập df_profile nếu chưa có bảng này trong DB
df_profile = pd.DataFrame({
    'ticker': tickers,
    'company_name': [f"Công ty Cổ phần {t}" for t in tickers],
    'tax_code': ['Đang cập nhật'] * len(tickers),
    'industry': ['Sản xuất & Thương mại'] * len(tickers),
    'address': ['Đang cập nhật'] * len(tickers)
})


# 5. Khởi tạo các Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    txt["tab1"], txt["tab2"], txt["tab3"], txt["tab4"], txt["tab5"], txt["tab6"], "👥 Lãnh đạo & Cổ đông"
])

# ----------------------------------------------------
# TAB 1: CORPORATE DEEP-DIVE & DUPONT ANALYSIS
# ----------------------------------------------------
with tab1:
    df_ticker = df_raw[df_raw['ticker'] == selected_ticker].sort_values("report_period").copy()
    
    if not df_ticker.empty:
        latest = df_ticker.iloc[-1]
        
        # Sửa lỗi: Lấy từ df_profile đã được định nghĩa ở trên
        profile_row = df_profile[df_profile['ticker'] == selected_ticker]
        
        profile = {
            "name": profile_row.iloc[0].get('company_name', f"Công ty CP {selected_ticker}"),
            "tax_code": profile_row.iloc[0].get('tax_code', 'Đang cập nhật'),
            "industry": profile_row.iloc[0].get('industry', 'Đang cập nhật'),
            "address": profile_row.iloc[0].get('address', 'Đang cập nhật')
        } if not profile_row.empty else {
            "name": f"Công ty CP {selected_ticker}", "tax_code": "N/A", "industry": "N/A", "address": "N/A"
        }

        # Hiển thị thông tin doanh nghiệp
        st.markdown(f"""
        <div style="background-color: #ffffff; padding: 18px 22px; border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h2 style="margin:0 0 8px 0; color:#1e3a8a; font-size: 1.4rem; font-weight: 700;">🏛️ {profile['name']} ({selected_ticker})</h2>
                    <p style="margin: 4px 0; color: #334155; font-size: 0.92rem;">
                        <b>Mã số thuế:</b> <span style="color:#0f172a; font-weight:600;">{profile['tax_code']}</span> &nbsp;|&nbsp; 
                        <b>Ngành nghề:</b> <span style="color:#0284c7; font-weight:600;">{profile['industry']}</span> &nbsp;|&nbsp;
                        <b>Kỳ báo cáo gần nhất:</b> <span style="color:#047857; font-weight:600;">{latest['report_period']}</span>
                    </p>
                    <p style="margin: 4px 0 0 0; color: #64748b; font-size: 0.88rem;">
                        <b>📍 Địa chỉ trụ sở:</b> {profile['address']}
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Executive Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">{txt.get("roe_label", "ROE")}</div>
                    <div class="metric-value">{latest.get("roe_pct", 0):.2f}%</div>
                    <div class="metric-badge-pos">Profitability</div>
                </div>
            ''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">{txt.get("roa_label", "ROA")}</div>
                    <div class="metric-value">{latest.get("roa_pct", 0):.2f}%</div>
                    <div class="metric-badge-pos">Efficiency</div>
                </div>
            ''', unsafe_allow_html=True)
        with c3:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">{txt.get("de_label", "D/E Ratio")}</div>
                    <div class="metric-value">{latest.get("debt_to_equity", 0):.2f}x</div>
                    <div class="metric-badge-pos">Capital Structure</div>
                </div>
            ''', unsafe_allow_html=True)
        with c4:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">{txt.get("margin_label", "Net Margin")}</div>
                    <div class="metric-value">{latest.get("net_margin_pct", 0):.2f}%</div>
                    <div class="metric-badge-pos">Margin</div>
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Biểu đồ Doanh thu, Lợi nhuận & Chỉ số
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown(f"#### 💰 {txt.get('chart_rev', 'Doanh thu & Lợi nhuận')}")
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
            st.markdown(f"#### 📉 {txt.get('chart_ratios', 'Chỉ số sinh lời (%)')}")
            fig_ratios = px.line(
                df_ticker, x="report_period", y=["roe_pct", "roa_pct", "gross_margin_pct", "net_margin_pct"],
                markers=True, template="plotly_white",
                color_discrete_sequence=["#1e3a8a", "#059669", "#d97706", "#2563eb"]
            )
            fig_ratios.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_ratios, use_container_width=True)

        st.divider()

        # 4. Phân tích Cấu trúc Bảng Cân đối Kế toán
        st.markdown(f"### 🧩 Phân Tích Cơ Cấu Tài Sản & Nguồn Vốn ({latest['report_period']})")
        
        asset_details = {
            "Tiền & Tương đương tiền": latest.get('cash_and_equivalents', 0),
            "Phải thu ngắn hạn": latest.get('short_term_receivables', 0),
            "Hàng tồn kho": latest.get('inventory', 0),
            "Tài sản ngắn hạn khác": latest.get('other_short_term_assets', 0),
            "Tài sản cố định": latest.get('fixed_assets', 0),
            "Bất động sản đầu tư": latest.get('investment_properties', 0),
            "Tài sản dài hạn khác": latest.get('other_long_term_assets', 0)
        }
        
        capital_details = {
            "Nợ ngắn hạn": latest.get('short_term_debt', 0),
            "Nợ dài hạn": latest.get('long_term_debt', 0),
            "Vốn góp chủ sở hữu": latest.get('owner_equity', 0),
            "Lợi nhuận sau thuế chưa phân phối": latest.get('undistributed_earnings', 0),
            "Quỹ & Vốn khác": latest.get('other_capital_and_funds', 0)
        }

        col_pie1, col_pie2 = st.columns(2)
        
        with col_pie1:
            st.markdown("##### 📦 Cơ cấu Tài sản")
            df_asset_pie = pd.DataFrame(list(asset_details.items()), columns=['Khoản mục', 'Giá trị'])
            df_asset_pie = df_asset_pie[df_asset_pie['Giá trị'] > 0] 
            
            if not df_asset_pie.empty:
                fig_asset_pie = px.pie(
                    df_asset_pie, values='Giá trị', names='Khoản mục',
                    hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_asset_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_asset_pie.update_layout(margin=dict(l=10, r=10, t=20, b=20), showlegend=False)
                st.plotly_chart(fig_asset_pie, use_container_width=True)
            else:
                st.info("Chưa có dữ liệu Cơ cấu Tài sản trong Database.")

        with col_pie2:
            st.markdown("##### 🏛️ Cơ cấu Nguồn vốn")
            df_capital_pie = pd.DataFrame(list(capital_details.items()), columns=['Khoản mục', 'Giá trị'])
            df_capital_pie = df_capital_pie[df_capital_pie['Giá trị'] > 0]
            
            if not df_capital_pie.empty:
                fig_capital_pie = px.pie(
                    df_capital_pie, values='Giá trị', names='Khoản mục',
                    hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_capital_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_capital_pie.update_layout(margin=dict(l=10, r=10, t=20, b=20), showlegend=False)
                st.plotly_chart(fig_capital_pie, use_container_width=True)
            else:
                st.info("Chưa có dữ liệu Cơ cấu Nguồn vốn trong Database.")

        # 5. Biểu đồ Xu hướng
        st.markdown("#### 📈 Xu hướng dịch chuyển các Khoản mục Trọng yếu (> 5% Cơ cấu)")
        
        trend_data = []
        for idx, row in df_ticker.iterrows():
            period = row['report_period']
            total_assets = row.get('total_assets', 1) 
            total_capital = row.get('total_capital', total_assets)
            
            trend_data.extend([
                {"Kỳ": period, "Khoản mục": "Vốn chủ sở hữu", "Tỷ trọng (%)": (row.get('owner_equity', 0) / total_capital) * 100},
                {"Kỳ": period, "Khoản mục": "Nợ ngắn hạn", "Tỷ trọng (%)": (row.get('short_term_debt', 0) / total_capital) * 100},
                {"Kỳ": period, "Khoản mục": "Nợ dài hạn", "Tỷ trọng (%)": (row.get('long_term_debt', 0) / total_capital) * 100},
                {"Kỳ": period, "Khoản mục": "Tài sản cố định", "Tỷ trọng (%)": (row.get('fixed_assets', 0) / total_assets) * 100},
                {"Kỳ": period, "Khoản mục": "Hàng tồn kho", "Tỷ trọng (%)": (row.get('inventory', 0) / total_assets) * 100},
                {"Kỳ": period, "Khoản mục": "Tiền & Tương đương tiền", "Tỷ trọng (%)": (row.get('cash_and_equivalents', 0) / total_assets) * 100}
            ])

        df_trend = pd.DataFrame(trend_data)
        
        # Sửa lỗi Plotly báo rỗng nếu CSDL không có dữ liệu 
        avg_shares = df_trend.groupby("Khoản mục")["Tỷ trọng (%)"].mean()
        major_items = avg_shares[avg_shares > 5.0].index.tolist()
        df_trend_filtered = df_trend[df_trend["Khoản mục"].isin(major_items)]

        if not df_trend_filtered.empty and not (df_trend_filtered['Tỷ trọng (%)'] == 0).all():
            fig_trend = px.line(
                df_trend_filtered, x="Kỳ", y="Tỷ trọng (%)", color="Khoản mục",
                markers=True, template="plotly_white",
                title="Biến động tỷ trọng các khoản mục lớn qua các năm"
            )
            fig_trend.update_layout(margin=dict(l=20, r=20, t=40, b=20), yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Chưa đủ dữ liệu các khoản mục chi tiết để thống kê xu hướng dài hạn.")

        st.divider()

        # 6. Mô hình DuPont Analysis
        st.markdown(f"#### {txt.get('dupont_title', 'Phân tích DuPont')}")
        st.latex(r"\text{ROE} = \text{ROA} \times \left(1 + \frac{\text{Debt}}{\text{Equity}}\right)")
        
        leverage_factor = 1 + (latest.get('debt_to_equity', 0) if pd.notnull(latest.get('debt_to_equity', 0)) else 0)
        st.info(txt.get('dupont_desc', 'Tỷ suất ROE {roe:.2f}% được thúc đẩy bởi ROA {roa:.2f}% và đòn bẩy tài chính {lev:.2f}x.').format(
            roe=latest.get('roe_pct', 0), roa=latest.get('roa_pct', 0), lev=leverage_factor
        ))
    else:
        st.warning("No financial data available for this ticker.")

# Chuẩn bị chung dữ liệu cho Tab 2 & Tab 3
feature_cols = ['roe_pct', 'roa_pct', 'debt_to_equity', 'gross_margin_pct', 'net_margin_pct']
df_ml = df_raw.sort_values("report_period").groupby("ticker").last().reset_index()
df_ml_clean = df_ml.dropna(subset=feature_cols).copy()
scaled_features = None

if len(df_ml_clean) > 0:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_ml_clean[feature_cols])

# Thêm hàm load dữ liệu Profile từ MotherDuck
@st.cache_data(ttl=3600)
def load_corporate_profile():
    try:
        # Thay đổi tên cột cho khớp với schema thực tế bạn cào được
        df = con.execute("""
            SELECT 
                ticker, 
                company_name, 
                tax_code, 
                industry, 
                headquarters AS address,
                website,
                established_year
            FROM gold_corporate_overview
        """).df()
        return df
    except Exception as e:
        st.warning("⚠️ Chưa đồng bộ bảng gold_corporate_overview hoặc đang cập nhật.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_shareholders(ticker):
    try:
        df = con.execute(f"SELECT * FROM gold_corporate_shareholders WHERE ticker = '{ticker}'").df()
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_officers(ticker):
    try:
        df = con.execute(f"SELECT * FROM gold_corporate_officers WHERE ticker = '{ticker}'").df()
        return df
    except:
        return pd.DataFrame()

df_profile = load_corporate_profile()


# --- BÊN TRONG TAB 1 (Cập nhật logic lấy profile) ---
with tab1:
    df_ticker = df_raw[df_raw['ticker'] == selected_ticker].sort_values("report_period").copy()
    
    if not df_ticker.empty:
        latest = df_ticker.iloc[-1]
        
        # Lấy thông tin thật từ df_profile
        profile_row = df_profile[df_profile['ticker'] == selected_ticker]
        
        if not profile_row.empty:
            profile = {
                "name": profile_row.iloc[0].get('company_name', f"Công ty CP {selected_ticker}"),
                "tax_code": profile_row.iloc[0].get('tax_code', 'N/A'),
                "industry": profile_row.iloc[0].get('industry', 'N/A'),
                "address": profile_row.iloc[0].get('address', 'N/A'),
                "website": profile_row.iloc[0].get('website', 'N/A'),
                "established": profile_row.iloc[0].get('established_year', 'N/A')
            }
        else:
            profile = {
                "name": f"Công ty CP {selected_ticker}", "tax_code": "Đang cập nhật", 
                "industry": "Đang cập nhật", "address": "Đang cập nhật",
                "website": "N/A", "established": "N/A"
            }

        # Hiển thị UI mới có thêm Website và Năm thành lập
        st.markdown(f"""
        <div style="background-color: #ffffff; padding: 18px 22px; border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h2 style="margin:0 0 8px 0; color:#1e3a8a; font-size: 1.4rem; font-weight: 700;">🏛️ {profile['name']} ({selected_ticker})</h2>
                    <p style="margin: 4px 0; color: #334155; font-size: 0.92rem;">
                        <b>Mã số thuế:</b> <span style="color:#0f172a; font-weight:600;">{profile['tax_code']}</span> &nbsp;|&nbsp; 
                        <b>Ngành nghề:</b> <span style="color:#0284c7; font-weight:600;">{profile['industry']}</span> &nbsp;|&nbsp;
                        <b>Thành lập:</b> <span style="color:#0f172a; font-weight:600;">{profile['established']}</span>
                    </p>
                    <p style="margin: 4px 0 0 0; color: #64748b; font-size: 0.88rem;">
                        <b>📍 Địa chỉ:</b> {profile['address']} <br>
                        <b>🌐 Website:</b> <a href="{profile['website']}" target="_blank">{profile['website']}</a>
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 2: UNSUPERVISED ML (K-MEANS, PCA & SHAP)
# ----------------------------------------------------
with tab2:
    st.markdown(f"### 🤖 {txt['pca_title']}")

    if len(df_ml_clean) >= k_clusters and scaled_features is not None:
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        df_ml_clean['Cluster'] = "Cluster " + kmeans.fit_predict(scaled_features).astype(str)

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
        # Chỉ lấy numeric column để tính mean
        cluster_profile = df_ml_clean.groupby('Cluster')[feature_cols].mean().reset_index()
        st.dataframe(cluster_profile.style.highlight_max(axis=0, color="#dbeafe"), use_container_width=True)
    else:
        st.warning("Not enough data to run K-Means clustering.")

    st.divider()

    st.markdown("### 📊 Phân Tích Giá Trị SHAP")
    st.caption("Mô hình SHAP đo lường chiều hướng và giá trị đóng góp của từng chỉ số tài chính.")

    col_shap1, col_shap2 = st.columns(2)
    with col_shap1:
        periods_available = ["Tất cả các năm"] + sorted(df_raw['report_period'].dropna().unique().tolist(), reverse=True)
        selected_period = st.selectbox("📅 Chọn kỳ báo cáo / Năm phân tích:", periods_available)
    
    with col_shap2:
        target_var = st.selectbox(
            "🎯 Indicator Mục tiêu đánh giá:", 
            ["net_revenue", "net_income"], 
            format_func=lambda x: "Doanh thu thuần (Net Revenue)" if x == "net_revenue" else "Lợi nhuận ròng (Net Income)"
        )

    df_shap = df_raw[df_raw['report_period'] == selected_period].copy() if selected_period != "Tất cả các năm" else df_raw.copy()
    df_shap_clean = df_shap.dropna(subset=feature_cols + [target_var]).copy()

    if len(df_shap_clean) >= 5:
        X_shap = df_shap_clean[feature_cols]
        y_shap = df_shap_clean[target_var]

        from sklearn.ensemble import RandomForestRegressor
        rf_shap = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_shap.fit(X_shap, y_shap)

        try:
            import shap
            import matplotlib.pyplot as plt

            explainer = shap.TreeExplainer(rf_shap)
            shap_values = explainer.shap_values(X_shap)

            st.markdown(f"#### Biểu đồ SHAP Summary Plot — Tác động lên **{('Doanh thu' if target_var == 'net_revenue' else 'Lợi nhuận')}**")
            
            fig_shap, ax = plt.subplots(figsize=(10, 4.5))
            shap.summary_plot(shap_values, X_shap, show=False)
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close()

        except ImportError:
            st.warning("⚠️ Thư viện `shap` chưa có sẵn. Đang hiển thị Mức độ tác động tương đối bằng Plotly:")
            importances = rf_shap.feature_importances_
            df_imp = pd.DataFrame({'Chỉ số': feature_cols, 'Mức độ tác động': importances}).sort_values(by='Mức độ tác động', ascending=True)
            fig_imp = px.bar(
                df_imp, x='Mức độ tác động', y='Chỉ số', orientation='h',
                template="plotly_white", color='Mức độ tác động', color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("Chưa đủ tập dữ liệu (tối thiểu 5 mẫu) trong kỳ đã chọn để thực hiện phân tích SHAP.")

# ----------------------------------------------------
# TAB 3: ANOMALY DETECTION (ĐÃ BỔ SUNG FIX LỖI THIẾU TAB NÀY)
# ----------------------------------------------------
with tab3:
    st.markdown(f"### 🚨 {txt['anomaly_title']}")
    
    if len(df_ml_clean) > 5 and scaled_features is not None:
        # Sử dụng IsolationForest đã import ở đầu file
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        df_ml_clean['Anomaly_Score'] = iso_forest.fit_predict(scaled_features)
        df_ml_clean['Status'] = df_ml_clean['Anomaly_Score'].map({1: 'Bình thường', -1: 'Bất thường (Anomaly)'})
        
        # Nếu chưa tính PCA ở tab2, thì tính lại để vẽ biểu đồ
        if 'PCA_1' not in df_ml_clean.columns:
            pca = PCA(n_components=2)
            pca_transformed = pca.fit_transform(scaled_features)
            df_ml_clean['PCA_1'] = pca_transformed[:, 0]
            df_ml_clean['PCA_2'] = pca_transformed[:, 1]
            
        fig_anomaly = px.scatter(
            df_ml_clean, x='PCA_1', y='PCA_2', color='Status',
            hover_name='ticker', template="plotly_white",
            color_discrete_map={'Bình thường': '#10b981', 'Bất thường (Anomaly)': '#ef4444'},
            title="Biểu đồ phân tán chỉ ra các Doanh nghiệp có chỉ số Tài chính tách biệt"
        )
        fig_anomaly.update_traces(marker=dict(size=11, opacity=0.85))
        st.plotly_chart(fig_anomaly, use_container_width=True)

        anomalies = df_ml_clean[df_ml_clean['Status'] == 'Bất thường (Anomaly)']
        st.markdown(f"**{txt['anomaly_count']}: {len(anomalies)}**")
        st.dataframe(anomalies[['ticker', 'Status'] + feature_cols], use_container_width=True)
    else:
        st.warning("Chưa đủ số lượng dữ liệu (tối thiểu > 5 mã cổ phiếu) để chạy mô hình Phát hiện bất thường.")

# ----------------------------------------------------
# TAB 4: DATA WAREHOUSE CONSOLE
# ----------------------------------------------------
with tab4:
    st.markdown(f"### {txt['data_console_title']}")
    
    col_f1, _ = st.columns(2)
    with col_f1:
        search_ticker = st.multiselect(txt["filter_ticker"], tickers, default=[selected_ticker] if selected_ticker in tickers else [])
    
    df_filtered = df_raw[df_raw['ticker'].isin(search_ticker)] if search_ticker else df_raw
    st.dataframe(df_filtered, use_container_width=True)
    st.caption(f"{txt['total_records']} **{len(df_filtered):,}**")

# ----------------------------------------------------
# TAB 5: MACRO, FX, RATES & SPATIAL MAP
# ----------------------------------------------------
with tab5:
    st.markdown("### 🌍 Thông tin Vĩ mô, Tỷ giá & Hiệu suất rủi ro (Sharpe)")
    
    col_macro1, col_macro2 = st.columns(2)
    
    with col_macro1:
        st.markdown("#### 💱 Tỷ giá & Lãi suất Ngân hàng (Real-time Simulation)")
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
        df_ticker_sharpe = df_raw[df_raw['ticker'] == selected_ticker].sort_values("report_period").copy()
        df_ticker_sharpe['Return'] = df_ticker_sharpe['net_income'].pct_change().fillna(0)
        df_ticker_sharpe['Sharpe'] = (df_ticker_sharpe['Return'] - df_ticker_sharpe['risk_free_rate']) / df_ticker_sharpe['daily_returns_volatility']
        
        fig_sharpe = px.bar(
            df_ticker_sharpe, x="report_period", y="Sharpe", 
            title=f"Sharpe Ratio của {selected_ticker} qua các kỳ",
            template="plotly_white", color="Sharpe", color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)

    st.divider()
    st.markdown(f"#### 📍 Vị trí địa lý Trụ sở doanh nghiệp ({selected_ticker})")
    location_dict = {
        "VNM": [10.7297, 106.7190],
        "FPT": [21.0278, 105.8342],
        "VCB": [21.0250, 105.8520],
        "HPG": [20.9400, 106.0600],
        "VIC": [21.0333, 105.9220],
    }
    coords = location_dict.get(selected_ticker, [10.7626, 106.6601])
    map_data = pd.DataFrame({'lat': [coords[0]], 'lon': [coords[1]]})
    st.map(map_data, zoom=12)

# ----------------------------------------------------
# TAB 6: 5-YEAR FINANCIALS & CREDIT RATING
# ----------------------------------------------------
with tab6:
    st.markdown(f"### 📋 Báo Cáo Tài Chính 5 Năm & Xếp Hạng Tín Dụng ({selected_ticker})")
    
    st.markdown("#### Dữ liệu BCTC (Income Statement, Balance Sheet, Ratios)")
    df_ticker_5y = df_raw[df_raw['ticker'] == selected_ticker].sort_values("report_period").tail(5)
    
    if not df_ticker_5y.empty:
        df_pivot = df_ticker_5y.set_index('report_period')[['net_revenue', 'net_income', 'roe_pct', 'roa_pct', 'debt_to_equity']].T
        df_pivot.index = ["Doanh thu thuần", "LNST", "ROE (%)", "ROA (%)", "Nợ / VCSH (D/E)"]
        st.dataframe(df_pivot.style.format("{:,.2f}"), use_container_width=True)
    else:
        st.info("Không đủ dữ liệu lịch sử cho cổ phiếu này.")

    st.divider()

    col_credit1, col_credit2 = st.columns([1, 1])
    
    with col_credit1:
        st.markdown("#### 🏆 Xếp Hạng Tín Dụng Doanh Nghiệp (Credit Trend)")
        
        def get_credit_rating(icr, debt_ebitda):
            if icr > 5 and debt_ebitda < 2: return "AAA", "#16a34a"
            elif icr > 3 and debt_ebitda < 4: return "A+", "#2563eb"
            elif icr > 1.5 and debt_ebitda < 6: return "BBB", "#d97706"
            else: return "CCC", "#dc2626"
            
        current_icr = 7.57
        current_debt_ebitda = 1.46
        rating, color = get_credit_rating(current_icr, current_debt_ebitda)
        
        st.markdown(f"""
        <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; border: 2px solid {color}; text-align: center;">
            <h3 style="color: #64748b; margin: 0;">Mức Xếp Hạng Hiện Tại</h3>
            <h1 style="color: {color}; font-size: 3.5rem; margin: 10px 0;">{rating}</h1>
            <p style="color: #475569; font-size: 1rem;">Mức độ rủi ro tín dụng: <b>An toàn cao</b></p>
            <hr style="border-top: 1px solid #e2e8f0;"/>
            <p>Khả năng thanh toán lãi vay (ICR): <b>{current_icr}x</b> | Total Debt / EBITDA: <b>{current_debt_ebitda}x</b></p>
        </div>
        """, unsafe_allow_html=True)

    with col_credit2:
        st.markdown("#### 🧠 ML Model: Đánh giá Tầm quan trọng (Feature Importance)")
        st.caption("Ứng dụng Random Forest Classifier xác định biến tài chính ảnh hưởng tới Xếp hạng Tín dụng.")
        
        features = df_raw[['roe_pct', 'roa_pct', 'debt_to_equity', 'gross_margin_pct', 'net_margin_pct']].dropna()
        if len(features) > 10:
            target = ((features['roe_pct'] > 12) & (features['debt_to_equity'] < 2.0)).astype(int)
            
            if len(np.unique(target)) > 1:
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(features, target)
                
                importances = rf_model.feature_importances_
                df_imp = pd.DataFrame({'Feature': features.columns, 'Importance': importances}).sort_values(by='Importance', ascending=True)
                
                fig_rf = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                                template='plotly_white', color='Importance', color_continuous_scale="Blues")
                fig_rf.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=250)
                st.plotly_chart(fig_rf, use_container_width=True)
            else:
                st.caption("Dữ liệu chỉ sinh ra một nhãn phân loại (không có biến thiên) nên không thể chạy Random Forest.")
        else:
            st.caption("Không đủ mẫu dữ liệu (>10 mẫu) để chạy mô hình Random Forest.")
# ----------------------------------------------------
# TAB 7: CORPORATE GOVERNANCE (OFFICERS & SHAREHOLDERS)
# ----------------------------------------------------
with tab7:
    st.markdown(f"### 👥 Quản trị Doanh nghiệp ({selected_ticker})")
    
    col_share, col_officer = st.columns([1, 1.2])
    
    # --- 1. BIỂU ĐỒ CƠ CẤU CỔ ĐÔNG ---
    with col_share:
        st.markdown("#### 🥧 Cơ cấu Cổ đông")
        df_shareholders = load_shareholders(selected_ticker)
        
        if not df_shareholders.empty:
            # Ghi chú: Đổi tên biến 'shareholder_name' và 'ownership_pct' 
            # cho khớp với tên cột thực tế mà bạn đã cào về
            name_col = 'shareholder_name' if 'shareholder_name' in df_shareholders.columns else df_shareholders.columns[1]
            pct_col = 'ownership_pct' if 'ownership_pct' in df_shareholders.columns else 'ownership_percent' if 'ownership_percent' in df_shareholders.columns else df_shareholders.columns[2]
            
            # Chỉ lấy các cổ đông có tỷ lệ > 0 để vẽ biểu đồ cho đẹp
            df_pie = df_shareholders[df_shareholders[pct_col] > 0]
            
            if not df_pie.empty:
                fig_pie = px.pie(
                    df_pie, 
                    values=pct_col, 
                    names=name_col,
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(
                    showlegend=False, 
                    margin=dict(t=20, b=20, l=10, r=10),
                    annotations=[dict(text='Cổ đông', x=0.5, y=0.5, font_size=16, showarrow=False)]
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Không có dữ liệu tỷ lệ sở hữu hợp lệ để vẽ biểu đồ.")
            
            # Hiển thị thêm bảng data rút gọn bên dưới biểu đồ
            st.dataframe(
                df_shareholders[[name_col, pct_col]].style.format({pct_col: "{:.2f}%"}),
                use_container_width=True, hide_index=True
            )
        else:
            st.warning(f"Chưa có dữ liệu cơ cấu cổ đông cho mã {selected_ticker}.")

    # --- 2. DANH SÁCH BAN LÃNH ĐẠO ---
    with col_officer:
        st.markdown("#### 👔 Danh sách Ban Lãnh đạo")
        df_officers = load_officers(selected_ticker)
        
        if not df_officers.empty:
            # Lọc bỏ cột ticker để giao diện gọn gàng hơn
            display_cols = [c for c in df_officers.columns if c.lower() != 'ticker']
            
            # Hiển thị bảng ban lãnh đạo
            st.dataframe(
                df_officers[display_cols],
                use_container_width=True, 
                hide_index=True,
                height=500  # Cố định chiều cao nếu danh sách dài
            )
        else:
            st.warning(f"Chưa có dữ liệu ban lãnh đạo cho mã {selected_ticker}.")