import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAE Used Cars Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark luxury theme */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0a0e1a 100%);
}

/* Sidebar */
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

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #13213a 0%, #1a2d4a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
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

/* Section headers */
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

/* Page title */
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

/* Filter labels */
.stSelectbox label, .stMultiSelect label, .stSlider label {
    color: #7a9cc4 !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* Divider */
hr { border-color: #1e2d45; margin: 24px 0; }

/* Table styling */
.stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Plotly theme ─────────────────────────────────────────────────────────────
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

# ─── Load Data ────────────────────────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/Raef-h/UAE-Used-Cars-Analysis/main/UAE%20Used%20Cars%20Analysis.csv"

@st.cache_data(show_spinner=False)
def load_data(source=None):
    if source is not None:
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(DATA_URL)
    # Clean up
    df.columns = df.columns.str.strip()
    df.dropna(subset=["Price", "Year"], inplace=True)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Mileage"] = pd.to_numeric(df.get("Mileage", pd.Series(dtype=float)), errors="coerce") if "Mileage" in df.columns else np.nan
    df.dropna(subset=["Price", "Year"], inplace=True)
    return df

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="text-align:center; padding: 16px 0 8px;">'
                '<span style="font-family:Rajdhani;font-size:1.4rem;color:#c9a84c;letter-spacing:3px;font-weight:700;">🚗 UAE CARS</span>'
                '<br><span style="color:#3a5a7a;font-size:0.65rem;letter-spacing:2px;">MARKET ANALYTICS</span>'
                '</div>', unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader("📂 Upload CSV (optional)", type=["csv"],
                                 help="Upload the UAE Used Cars CSV, or leave blank to auto-load from GitHub")

    with st.spinner("Loading data..."):
        try:
            df_raw = load_data(uploaded)
            st.success(f"✅ {len(df_raw):,} records loaded")
        except Exception as e:
            st.error(f"❌ Could not load data: {e}")
            st.info("Please upload the CSV file manually.")
            st.stop()

    st.markdown("## Filters")

    years = sorted(df_raw["Year"].dropna().unique().astype(int))
    year_range = st.slider("📅 Year Range", min_value=min(years), max_value=max(years),
                            value=(min(years), max(years)))

    price_min, price_max = int(df_raw["Price"].min()), int(df_raw["Price"].max())
    price_range = st.slider("💰 Price Range (AED)", min_value=price_min, max_value=price_max,
                             value=(price_min, min(price_max, 500_000)), step=5_000,
                             format="%d")

    if "Location" in df_raw.columns:
        cities = ["All"] + sorted(df_raw["Location"].dropna().unique().tolist())
        selected_city = st.selectbox("🏙️ City", cities)
    else:
        selected_city = "All"

    if "Fuel Type" in df_raw.columns:
        fuels = ["All"] + sorted(df_raw["Fuel Type"].dropna().unique().tolist())
        selected_fuel = st.selectbox("⛽ Fuel Type", fuels)
    else:
        selected_fuel = "All"

    if "Transmission" in df_raw.columns:
        transmissions = ["All"] + sorted(df_raw["Transmission"].dropna().unique().tolist())
        selected_trans = st.selectbox("⚙️ Transmission", transmissions)
    else:
        selected_trans = "All"

    if "Make" in df_raw.columns:
        makes = ["All"] + sorted(df_raw["Make"].dropna().unique().tolist())
        selected_make = st.selectbox("🏎️ Car Make", makes)
    else:
        selected_make = "All"

    st.divider()
    st.markdown('<p style="color:#2a4a6a;font-size:0.65rem;text-align:center;letter-spacing:1px;">DATA SOURCE: UAE USED CARS DATASET</p>',
                unsafe_allow_html=True)

# ─── Filter Data ──────────────────────────────────────────────────────────────
df = df_raw.copy()
df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]
df = df[(df["Price"] >= price_range[0]) & (df["Price"] <= price_range[1])]
if selected_city != "All" and "Location" in df.columns:
    df = df[df["Location"] == selected_city]
if selected_fuel != "All" and "Fuel Type" in df.columns:
    df = df[df["Fuel Type"] == selected_fuel]
if selected_trans != "All" and "Transmission" in df.columns:
    df = df[df["Transmission"] == selected_trans]
if selected_make != "All" and "Make" in df.columns:
    df = df[df["Make"] == selected_make]

# ─── Helper ───────────────────────────────────────────────────────────────────
def fmt_price(v):
    if v >= 1_000_000: return f"{v/1_000_000:.2f}M"
    if v >= 1_000: return f"{v/1_000:.1f}K"
    return str(int(v))

def apply_theme(fig):
    fig.update_layout(**PLOT_THEME)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">UAE Used Cars</div>'
            '<div class="page-subtitle">Market Intelligence Dashboard · Filtered View</div>',
            unsafe_allow_html=True)
st.markdown("---")

if len(df) == 0:
    st.warning("⚠️ No data matches your current filters. Please adjust the sidebar filters.")
    st.stop()

# ─── KPI Cards ────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

kpis = [
    (k1, "🚗", f"{len(df):,}", "Total Listings", ""),
    (k2, "💰", f"AED {fmt_price(df['Price'].mean())}", "Avg Price", ""),
    (k3, "📈", f"AED {fmt_price(df['Price'].median())}", "Median Price", ""),
    (k4, "🏆", f"AED {fmt_price(df['Price'].max())}", "Highest Price", ""),
    (k5, "📅", f"{int(df['Year'].median())}", "Median Year", ""),
]
if "Make" in df.columns:
    kpis.append((k6, "🏎️", str(df["Make"].value_counts().index[0]), "Top Brand", ""))
else:
    kpis.append((k6, "📊", f"{df['Year'].nunique()}", "Year Span", ""))

for col, icon, val, label, sub in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")  # spacing

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1: Location + Fuel Type
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Geographic & Fuel Distribution</div>', unsafe_allow_html=True)
col1, col2 = st.columns([3, 2])

with col1:
    if "Location" in df.columns:
        city_cnt = df["Location"].value_counts().head(15).reset_index()
        city_cnt.columns = ["City", "Count"]
        fig = px.bar(city_cnt, x="Count", y="City", orientation="h",
                     title="Cars per City (Top 15)",
                     color="Count", color_continuous_scale=["#1a3a5c", "#c9a84c"])
        fig.update_traces(texttemplate="%{x:,}", textposition="outside",
                          textfont=dict(color="#e8d5a3", size=9))
        fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False,
                          coloraxis_showscale=False)
        st.plotly_chart(apply_theme(fig), use_container_width=True)

with col2:
    if "Fuel Type" in df.columns:
        fuel_cnt = df["Fuel Type"].value_counts().reset_index()
        fuel_cnt.columns = ["Fuel", "Count"]
        FUEL_COLORS = {"Electric": "#00BFFF", "Diesel": "#E6D447",
                       "Hybrid": "#32CD32", "Gasoline": "#c9a84c"}
        colors = [FUEL_COLORS.get(f, "#888") for f in fuel_cnt["Fuel"]]
        fig = go.Figure(go.Pie(
            labels=fuel_cnt["Fuel"], values=fuel_cnt["Count"],
            marker=dict(colors=colors, line=dict(color="#0a0e1a", width=2)),
            textinfo="label+percent", hole=0.45,
            textfont=dict(color="#e8d5a3", size=11),
        ))
        fig.update_layout(title="Fuel Type Distribution",
                          showlegend=True,
                          legend=dict(font=dict(color="#8bafc7"), bgcolor="rgba(0,0,0,0)"),
                          annotations=[dict(text="Fuel", x=0.5, y=0.5, showarrow=False,
                                            font=dict(color="#c9a84c", size=13, family="Rajdhani"))])
        st.plotly_chart(apply_theme(fig), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2: Year Trend + Top Makes
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Year Trends & Top Brands</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    year_cnt = df["Year"].value_counts().sort_index().reset_index()
    year_cnt.columns = ["Year", "Count"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=year_cnt["Year"], y=year_cnt["Count"],
        mode="lines+markers",
        line=dict(color="#c9a84c", width=2.5),
        marker=dict(color="#e8c96d", size=7, line=dict(color="#0a0e1a", width=1.5)),
        fill="tozeroy", fillcolor="rgba(201,168,76,0.08)",
        name="Cars Listed",
        hovertemplate="<b>%{x}</b><br>%{y:,} cars<extra></extra>"
    ))
    fig.update_layout(title="Cars Listed by Year", showlegend=False)
    st.plotly_chart(apply_theme(fig), use_container_width=True)

with col4:
    if "Make" in df.columns:
        make_cnt = df["Make"].value_counts().head(10).reset_index()
        make_cnt.columns = ["Make", "Count"]
        fig = px.bar(make_cnt, x="Make", y="Count", title="Top 10 Car Brands by Listing Count",
                     color="Count", color_continuous_scale=["#1a3a5c", "#c9a84c"])
        fig.update_traces(texttemplate="%{y:,}", textposition="outside",
                          textfont=dict(color="#e8d5a3", size=9))
        fig.update_layout(xaxis_tickangle=-35, coloraxis_showscale=False)
        st.plotly_chart(apply_theme(fig), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3: Price Analysis
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Price Intelligence</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)

with col5:
    if "Make" in df.columns:
        top_price_make = df.groupby("Make")["Price"].max().sort_values(ascending=False).head(10).reset_index()
        top_price_make.columns = ["Make", "Price"]
        top_price_make["Label"] = top_price_make["Price"].apply(lambda x: f"AED {fmt_price(x)}")
        GOLD = ["#D4AF37", "#C0C0C0", "#C47E55"] + ["#4a6a8a"] * 7
        fig = px.bar(top_price_make, x="Make", y="Price",
                     title="Top 10 Makes by Highest Price",
                     text="Label",
                     color_discrete_sequence=GOLD)
        fig.update_traces(textposition="outside", textfont=dict(color="#e8d5a3", size=8),
                          marker_color=GOLD)
        fig.update_layout(xaxis_tickangle=-35, showlegend=False,
                          yaxis_title="Price (AED)")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

with col6:
    # Price distribution histogram
    fig = px.histogram(df[df["Price"] < df["Price"].quantile(0.98)], x="Price",
                       nbins=60, title="Price Distribution (excl. top 2%)",
                       color_discrete_sequence=["#c9a84c"])
    fig.update_traces(marker_line_color="#0a0e1a", marker_line_width=0.5,
                      opacity=0.85)
    fig.update_layout(yaxis_title="Count", xaxis_title="Price (AED)", showlegend=False)
    st.plotly_chart(apply_theme(fig), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 4: Models + Transmission + Color
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Models, Transmission & Color</div>', unsafe_allow_html=True)
col7, col8, col9 = st.columns(3)

with col7:
    if "Model" in df.columns:
        top_model_price = df.groupby("Model")["Price"].max().sort_values(ascending=False).head(10).reset_index()
        top_model_price.columns = ["Model", "Price"]
        top_model_price["Label"] = top_model_price["Price"].apply(lambda x: fmt_price(x))
        fig = px.bar(top_model_price, x="Price", y="Model", orientation="h",
                     title="Top 10 Models by Highest Price",
                     color="Price", color_continuous_scale=["#1a3a5c", "#c9a84c"],
                     text="Label")
        fig.update_traces(textposition="outside", textfont=dict(color="#e8d5a3", size=8))
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(apply_theme(fig), use_container_width=True)

with col8:
    if "Transmission" in df.columns:
        trans_cnt = df["Transmission"].value_counts().reset_index()
        trans_cnt.columns = ["Transmission", "Count"]
        fig = go.Figure(go.Pie(
            labels=trans_cnt["Transmission"], values=trans_cnt["Count"],
            marker=dict(colors=["#2A9D8F", "#D2B48C"], line=dict(color="#0a0e1a", width=2)),
            textinfo="label+percent", hole=0.5,
            pull=[0, 0.08],
            textfont=dict(color="#e8d5a3", size=12),
        ))
        fig.update_layout(title="Transmission Type",
                          showlegend=False,
                          annotations=[dict(text="⚙️", x=0.5, y=0.5, showarrow=False,
                                            font=dict(size=22))])
        st.plotly_chart(apply_theme(fig), use_container_width=True)

with col9:
    if "Color" in df.columns:
        color_cnt = df["Color"].value_counts().head(9).reset_index()
        color_cnt.columns = ["Color", "Count"]
        fig = go.Figure(go.Scatter(
            x=color_cnt["Color"], y=color_cnt["Count"],
            mode="lines+markers+text",
            line=dict(color="#c9a84c", width=2),
            marker=dict(color="#e8c96d", size=10, line=dict(color="#0a0e1a", width=2)),
            text=color_cnt["Count"].astype(str),
            textposition="top center", textfont=dict(color="#e8d5a3", size=9),
        ))
        fig.update_layout(title="Top 9 Colors", yaxis_title="Count")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 5: Price vs Year scatter + Avg Price by Year
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Price Trends Over Time</div>', unsafe_allow_html=True)
col10, col11 = st.columns(2)

with col10:
    avg_price_yr = df.groupby("Year")["Price"].mean().reset_index()
    avg_price_yr.columns = ["Year", "Avg Price"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=avg_price_yr["Year"], y=avg_price_yr["Avg Price"],
        mode="lines+markers",
        line=dict(color="#3a7fc1", width=2.5),
        marker=dict(color="#5dade2", size=8),
        fill="tozeroy", fillcolor="rgba(58,127,193,0.08)",
        hovertemplate="<b>%{x}</b><br>Avg: AED %{y:,.0f}<extra></extra>"
    ))
    fig.update_layout(title="Average Price by Year", yaxis_title="Avg Price (AED)")
    st.plotly_chart(apply_theme(fig), use_container_width=True)

with col11:
    # Box plot: price by fuel type
    if "Fuel Type" in df.columns:
        df_box = df[df["Price"] < df["Price"].quantile(0.97)]
        fig = px.box(df_box, x="Fuel Type", y="Price", color="Fuel Type",
                     title="Price Range by Fuel Type",
                     color_discrete_map={"Electric": "#00BFFF", "Diesel": "#E6D447",
                                         "Hybrid": "#32CD32", "Gasoline": "#c9a84c"})
        fig.update_layout(showlegend=False, yaxis_title="Price (AED)")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 6: Predictions
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Predictive Analytics (ML Forecast)</div>', unsafe_allow_html=True)
col12, col13 = st.columns(2)

with col12:
    year_cnt2 = df_raw["Year"].value_counts().sort_index()
    X = year_cnt2.index.values.reshape(-1, 1)
    y = year_cnt2.values
    model = LinearRegression().fit(X, y)
    future_years = np.array([2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
    preds = model.predict(future_years).clip(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=year_cnt2.index, y=year_cnt2.values,
                              mode="markers+lines", name="Actual",
                              line=dict(color="#3a7fc1", width=2),
                              marker=dict(color="#5dade2", size=7)))
    fig.add_trace(go.Scatter(x=future_years.flatten(), y=preds,
                              mode="markers+lines+text", name="Predicted",
                              line=dict(color="#2ecc71", width=2, dash="dash"),
                              marker=dict(color="#27ae60", size=9),
                              text=[str(int(p)) for p in preds],
                              textposition="top center",
                              textfont=dict(color="#2ecc71", size=9)))
    fig.update_layout(title="Car Count Forecast (Linear Regression)",
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8bafc7")))
    st.plotly_chart(apply_theme(fig), use_container_width=True)

with col13:
    total_2025 = df_raw["Price"].sum()
    growth = 0.05
    fut_yrs = [2026, 2027, 2028, 2029, 2030]
    pred_prices = [total_2025 * ((1 + growth) ** (yr - 2025)) for yr in fut_yrs]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[2025], y=[total_2025], name="2025 Total",
                          marker_color="#3a7fc1",
                          text=[f"AED {fmt_price(total_2025)}"],
                          textposition="outside", textfont=dict(color="#5dade2")))
    fig.add_trace(go.Scatter(x=fut_yrs, y=pred_prices,
                              mode="markers+lines+text", name="Predicted (+5%/yr)",
                              line=dict(color="#c9a84c", width=2, dash="dash"),
                              marker=dict(color="#e8c96d", size=9),
                              text=[fmt_price(p) for p in pred_prices],
                              textposition="top center",
                              textfont=dict(color="#c9a84c", size=9)))
    fig.update_layout(title="Total Market Value Forecast (5% YoY)",
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8bafc7")),
                      yaxis_title="Total Price (AED)")
    st.plotly_chart(apply_theme(fig), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA TABLE (Collapsible)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Raw Data Explorer</div>', unsafe_allow_html=True)
with st.expander(f"📋 Browse Filtered Data ({len(df):,} records)", expanded=False):
    sort_col = st.selectbox("Sort by", df.columns.tolist(), index=df.columns.tolist().index("Price") if "Price" in df.columns else 0)
    sort_asc = st.radio("Order", ["Descending", "Ascending"], horizontal=True) == "Ascending"
    st.dataframe(
        df.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True),
        use_container_width=True, height=400
    )
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered CSV", csv_out, "uae_cars_filtered.csv", "text/csv")

# Footer
st.markdown("""
<div style='text-align:center; color:#1e3a5f; font-size:0.65rem; letter-spacing:2px;
     text-transform:uppercase; margin-top:40px; padding: 16px 0; border-top: 1px solid #1e2d45;'>
    UAE Used Cars Dashboard · Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
