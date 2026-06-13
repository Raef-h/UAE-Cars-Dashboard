import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="UAE Used Cars Dashboard", page_icon="🚗", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0a0e1a 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f1623 0%, #141d2e 100%); border-right: 1px solid #1e2d45; }
.kpi-card { background: linear-gradient(135deg, #13213a 0%, #1a2d4a 100%); border: 1px solid #1e3a5f; border-radius: 12px; padding: 20px 24px; text-align: center; min-height: 145px; display: flex; flex-direction: column; justify-content: center; position: relative; overflow: hidden; }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #c9a84c, #e8c96d); }
.kpi-label { color: #7a9cc4; font-size: 0.72rem; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px; font-weight: 500; }
.kpi-value { color: #e8d5a3; font-family: 'Rajdhani', sans-serif; font-size: 2rem; font-weight: 700; line-height: 1; }
.kpi-icon { font-size: 1.4rem; margin-bottom: 8px; }
.section-header { color: #c9a84c; font-family: 'Rajdhani', sans-serif; font-size: 1.05rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; margin: 32px 0 16px 0; padding-left: 12px; border-left: 3px solid #c9a84c; }
.page-title { font-family: 'Rajdhani', sans-serif; font-size: 2.6rem; font-weight: 700; color: #e8d5a3; letter-spacing: 4px; text-transform: uppercase; }
.page-subtitle { color: #4a7a9b; font-size: 0.85rem; letter-spacing: 2px; text-transform: uppercase; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

PLOT_THEME = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,20,38,0.6)", font=dict(family="Inter", color="#8bafc7", size=11), title_font=dict(family="Rajdhani", color="#e8d5a3", size=16), margin=dict(l=16, r=16, t=48, b=16), colorway=["#c9a84c", "#3a7fc1", "#2ecc71", "#e74c3c", "#9b59b6"])
AXIS_STYLE = dict(gridcolor="#1a2a3a", gridwidth=1, zerolinecolor="#1e3a5f", tickfont=dict(color="#6a8fa8", size=10), title_font=dict(color="#7a9cc4", size=11))

DATA_URL = "https://raw.githubusercontent.com/Raef-h/UAE-Used-Cars-Analysis/main/UAE%20Used%20Cars%20Analysis.csv"

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.strip()
    df.dropna(subset=["Price", "Year"], inplace=True)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df

with st.sidebar:
    st.markdown('<div style="text-align:center;"><span style="font-family:Rajdhani;font-size:1.4rem;color:#c9a84c;letter-spacing:3px;font-weight:700;">🚗 UAE CARS</span></div>', unsafe_allow_html=True)
    df_raw = load_data()
    yr = st.slider("📅 Year Range", int(df_raw["Year"].min()), int(df_raw["Year"].max()), (int(df_raw["Year"].min()), int(df_raw["Year"].max())))
    pr = st.slider("💰 Price Range (AED)", int(df_raw["Price"].min()), 500000, (int(df_raw["Price"].min()), 500000))
    df = df_raw[(df_raw["Year"] >= yr[0]) & (df_raw["Year"] <= yr[1]) & (df_raw["Price"] >= pr[0]) & (df_raw["Price"] <= pr[1])]

def apply_theme(fig):
    fig.update_layout(**PLOT_THEME)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig

st.markdown('<div class="page-title">UAE Used Cars</div><div class="page-subtitle">Market Intelligence Dashboard</div>', unsafe_allow_html=True)

cols = st.columns(5)
metrics = [("Listings", len(df)), ("Avg Price", f"{df['Price'].mean()/1000:.0f}K"), ("Median", f"{df['Price'].median()/1000:.0f}K"), ("Max Price", f"{df['Price'].max()/1000:.0f}K"), ("Top Brand", df["Make"].mode()[0])]
for i, (l, v) in enumerate(metrics):
    cols[i].markdown(f'<div class="kpi-card"><div class="kpi-label">{l}</div><div class="kpi-value">{v}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">Market Analysis</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(df["Location"].value_counts(), title="Cars per City")
    st.plotly_chart(apply_theme(fig), use_container_width=True)
with c2:
    fig = px.pie(df, names="Fuel Type", title="Fuel Distribution")
    st.plotly_chart(apply_theme(fig), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    fig = px.histogram(df, x="Price", title="Price Distribution")
    st.plotly_chart(apply_theme(fig), use_container_width=True)
with c4:
    fig = px.bar(df["Make"].value_counts().head(10), title="Top 10 Brands")
    st.plotly_chart(apply_theme(fig), use_container_width=True)
