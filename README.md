# 🚗 UAE Used Cars — Streamlit Dashboard

A professional, interactive analytics dashboard for the UAE Used Cars dataset.

## Features

- **10 Interactive Plotly Charts** (bar, pie, line, scatter, box, histogram)
- **Live Sidebar Filters** — Year, Price, City, Fuel Type, Transmission, Make
- **KPI Cards** — Total listings, avg/median/max price, top brand
- **ML Forecast** — Linear regression for car count & market value
- **Data Explorer** — Sortable table with CSV download
- **Dark Luxury Theme** — Gold & navy palette

## 🚀 Deploy on Streamlit Community Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add app.py requirements.txt
git commit -m "Initial dashboard"
git remote add origin https://github.com/YOUR_USERNAME/uae-cars-dashboard.git
git push -u origin main
```

### Step 2 — Deploy
1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo, branch `main`, file `app.py`
5. Click **Deploy** 🎉

The dashboard auto-loads data from the original GitHub CSV.
You can also upload your own CSV via the sidebar.

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset Columns Used
- `Price`, `Year`, `Location`, `Fuel Type`, `Transmission`, `Make`, `Model`, `Color`
