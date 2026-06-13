import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="UAE Used Cars Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",Price Trends Over Time
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0a0e1a 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1623 0%, #141d2e 100%);
    border-right: 1px solid #1e2d45;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #c9a84c;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

.kpi-card {
    background: linear-gradient(135deg, #13213a 0%, #1a2d4a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    min-height: 145px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #c9a84c, #e8c96d);
}
.kpi-label {
    color: #7a9cc4;
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
    font-weight: 500;
}
.kpi-value {
    color: #e8d5a3;
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.kpi-sub {
    color: #4a7a9b;
    font-size: 0.7rem;
    margin-top: 6px;
}
.kpi-icon {
    font-size: 1.4rem;
    margin-bottom: 8px;
}

.section-header {
    color: #c9a84c;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 32px 0 16px 0;
    padding-left: 12px;
    border-left: 3px solid #c9a84c;
}

.page-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #e8d5a3;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 0;
    line-height: 1;
}
.page-subtitle {
    color: #4a7a9b;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 6px;
}

.stSelectbox label, .stMultiSelect label, .stSlider label {
    color: #7a9cc4 !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

hr { border-color: #1e2d45; margin: 24px 0; }
.stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,20,38,0.6)",
    font=dict(family="Inter", color="#8bafc7", size=11),
    title_font=dict(family="Rajdhani", color="#e8d5a3", size=16),
    margin=dict(l=16, r=16, t=48, b=16),
    colorway=["#c9a84c", "#3a7fc1", "#2ecc71", "#e74c3c", "#9b59b6",
               "#1abc9c", "#e67e22", "#34495e", "#e8c96d", "#5dade2"],
)

AXIS_STYLE = dict(
    gridcolor="#1a2a3a",
    gridwidth=1,
    zerolinecolor="#1e3a5f",
    tickfont=dict(color="#6a8fa8", size=10),
    title_font=dict(color="#7a9cc4", size=11),
)

DATA_URL = "https://raw.githubusercontent.com/Raef-h/UAE-Used-Cars-Analysis/main/UAE%20Used%20Cars%20Analysis.csv"

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.strip()
    df.dropna(subset=["Price", "Year"], inplace=True)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Mileage"] = pd.to_numeric(df.get("Mileage", pd.Series(dtype=float)), errors="coerce") if "Mileage" in df.columns else np.nan
    df.dropna(subset=["Price", "Year"], inplace=True)
    return df

with st.sidebar:
    st.markdown('<div style="text-align:center; padding: 16px 0 8px;">'
                '<span style="font-family:Rajdhani;font-size:1.4rem;color:#c9a84c;letter-spacing:3px;font-weight:700;">🚗 UAE CARS</span>'
                '<br><span style="color:#3a5a7a;font-size:0.65rem;letter-spacing:2px;">MARKET ANALYTICS</span>'
                '</div>', unsafe_allow_html=True)
    st.divider()

    with st.spinner("Loading data..."):
        df_raw = load_data()
        
    st.markdown("## Filters")
    years = sorted(df_raw["Year"].dropna().unique().astype(int))
    year_range = st.slider("📅 Year Range", min_value=min(years), max_value=max(years),
                            value=(min(years), max(years)))
    price_min, price_max = int(df_raw["Price"].min()), int(df_raw["Price"].max())
    price_range = st.slider("💰 Price Range (AED)", min_value=price_min, max_value=price_max,
                             value=(price_min, min(price_max, 500_000)), step=5_000,
                             format="%d")

    selected_city = st.selectbox("🏙️ City", ["All"] + sorted(df_raw["Location"].dropna().unique().tolist())) if "Location" in df_raw.columns else "All"
    selected_fuel = st.selectbox("⛽ Fuel Type", ["All"] + sorted(df_raw["Fuel Type"].dropna().unique().tolist())) if "Fuel Type" in df_raw.columns else "All"
    selected_trans = st.selectbox("⚙️ Transmission", ["All"] + sorted(df_raw["Transmission"].dropna().unique().tolist())) if "Transmission" in df_raw.columns else "All"
    selected_make = st.selectbox("🏎️ Car Make", ["All"] + sorted(df_raw["Make"].dropna().unique().tolist())) if "Make" in df_raw.columns else "All"

    st.divider()
    st.markdown('<p style="color:#2a4a6a;font-size:0.65rem;text-align:center;letter-spacing:1px;">DATA SOURCE: UAE USED CARS DATASET</p>',
                unsafe_allow_html=True)

df = df_raw.copy()
df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]
df = df[(df["Price"] >= price_range[0]) & (df["Price"] <= price_range[1])]
if selected_city != "All" and "Location" in df.columns: df = df[df["Location"] == selected_city]
if selected_fuel != "All" and "Fuel Type" in df.columns: df = df[df["Fuel Type"] == selected_fuel]
if selected_trans != "All" and "Transmission" in df.columns: df = df[df["Transmission"] == selected_trans]
if selected_make != "All" and "Make" in df.columns: df = df[df["Make"] == selected_make]

def fmt_price(v):
    if v >= 1_000_000: return f"{v/1_000_000:.2f}M"
    if v >= 1_000: return f"{v/1_000:.1f}K"
    return str(int(v))

def apply_theme(fig):
    fig.update_layout(**PLOT_THEME)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig

st.markdown('<div class="page-title">UAE Used Cars</div>'
            '<div class="page-subtitle">Market Intelligence Dashboard · Filtered View</div>',
            unsafe_allow_html=True)
st.markdown("---")

if len(df) == 0:
    st.warning("⚠️ No data matches your current filters.")
    st.stop()

k1, k2, k3, k4, k5, k6 = st.columns(6)
kpis = [
    (k1, "🚗", f"{len(df):,}", "Total Listings", ""),
    (k2, "💰", f"AED {fmt_price(df['Price'].mean())}", "Avg Price", ""),
    (k3, "📈", f"AED {fmt_price(df['Price'].median())}", "Median Price", ""),
    (k4, "🏆", f"AED {fmt_price(df['Price'].max())}", "Highest Price", ""),
    (k5, "📅", f"{int(df['Year'].median())}", "Median Year", ""),
    (k6, "🏎️", str(df["Make"].value_counts().index[0]) if "Make" in df.columns else "-", "Top Brand", "")
]
for col, icon, val, label, sub in kpis:
    with col:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-header">Geographic & Fuel Distribution</div>', unsafe_allow_html=True)
col1, col2 = st.columns([3, 2])
with col1:
    if "Location" in df.columns:
        city_cnt = df["Location"].value_counts().head(15).reset_index()
        city_cnt.columns = ["City", "Count"]
        fig = px.bar(city_cnt, x="Count", y="City", orientation="h", color="Count", 
             color_continuous_scale=["#1a3a5c", "#c9a84c"],
             title="Cars per City (Top 15)")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(apply_theme(fig), use_container_width=True)
with col2:
    if "Fuel Type" in df.columns:
        fuel_cnt = df["Fuel Type"].value_counts().reset_index()
        fuel_cnt.columns = ["Fuel", "Count"]
        fig = go.Figure(go.Pie(labels=fuel_cnt["Fuel"], values=fuel_cnt["Count"], hole=0.45))
        fig.update_layout(title="Fuel Type Distribution")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

st.markdown('<div class="section-header">Year Trends & Top Brands</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    year_cnt = df["Year"].value_counts().sort_index().reset_index()
    year_cnt.columns = ["Year", "Count"]
    fig = go.Figure(go.Scatter(x=year_cnt["Year"], y=year_cnt["Count"], mode="lines+markers", fill="tozeroy"))
    st.plotly_chart(apply_theme(fig), use_container_width=True)
with col4:
    if "Make" in df.columns:
        make_cnt = df["Make"].value_counts().head(10).reset_index()
        make_cnt.columns = ["Make", "Count"]
        fig = px.bar(make_cnt, x="Make", y="Count")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

st.markdown('<div class="section-header">Price Intelligence</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    if "Make" in df.columns:
        top_price = df.groupby("Make")["Price"].max().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_price, x="Make", y="Price")
        st.plotly_chart(apply_theme(fig), use_container_width=True)
with col6:
    fig = px.histogram(df[df["Price"] < df["Price"].quantile(0.98)], x="Price")
    st.plotly_chart(apply_theme(fig), use_container_width=True)

st.markdown('<div class="section-header">Models, Transmission & Color</div>', unsafe_allow_html=True)
col7, col8, col9 = st.columns(3)
with col7:
    if "Model" in df.columns:
        model_p = df.groupby("Model")["Price"].max().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(model_p, x="Price", y="Model", orientation="h")
        st.plotly_chart(apply_theme(fig), use_container_width=True)
with col8:
    if "Transmission" in df.columns:
        fig = px.pie(df, names="Transmission", hole=0.5)
        st.plotly_chart(apply_theme(fig), use_container_width=True)
with col9:
    if "Color" in df.columns:
        color_c = df["Color"].value_counts().head(9).reset_index()
        fig = px.line(color_c, x="Color", y="count")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

st.markdown('<div class="section-header">Price Trends Over Time</div>', unsafe_allow_html=True)
col10, col11 = st.columns(2)
with col10:
    avg_p = df.groupby("Year")["Price"].mean().reset_index()
    fig = go.Figure(go.Scatter(x=avg_p["Year"], y=avg_p["Price"], mode="lines+markers", fill="tozeroy"))
    st.plotly_chart(apply_theme(fig), use_container_width=True)
with col11:
if "Fuel Type" in df.columns:
    fig = px.box(df[df["Price"] < df["Price"].quantile(0.97)], 
                 x="Fuel Type", y="Price", color="Fuel Type",
                 color_discrete_map={"Gasoline": "#c9a84c", "Diesel": "#3a7fc1", 
                                     "Electric": "#2ecc71", "Hybrid": "#e74c3c"},
                 title="Price Range by Fuel Type")
    fig.update_layout(showlegend=True) 
    st.plotly_chart(apply_theme(fig), use_container_width=True)
st.markdown('<div class="section-header">Predictive Analytics (ML Forecast)</div>', unsafe_allow_html=True)
col12, col13 = st.columns(2)
with col12:
    year_cnt2 = df_raw["Year"].value_counts().sort_index()
    X = year_cnt2.index.values.reshape(-1, 1)
    y = year_cnt2.values
    model = LinearRegression().fit(X, y)
    future = np.array([2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
    preds = model.predict(future).clip(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=year_cnt2.index, y=year_cnt2.values, name="Actual"))
    fig.add_trace(go.Scatter(x=future.flatten(), y=preds, name="Predicted"))
    st.plotly_chart(apply_theme(fig), use_container_width=True)
with col13:
    total_2025 = df_raw["Price"].sum()
    fut_yrs = [2026, 2027, 2028, 2029, 2030]
    pred_prices = [total_2025 * ((1.05) ** (yr - 2025)) for yr in fut_yrs]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[2025], y=[total_2025], name="2025 Total"))
    fig.add_trace(go.Scatter(x=fut_yrs, y=pred_prices, name="Predicted (+5%/yr)"))
    st.plotly_chart(apply_theme(fig), use_container_width=True)

st.markdown('<div class="section-header">Raw Data Explorer</div>', unsafe_allow_html=True)
with st.expander(f"📋 Browse Filtered Data ({len(df):,} records)", expanded=False):
    st.dataframe(df, use_container_width=True)
