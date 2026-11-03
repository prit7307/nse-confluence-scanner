import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import os

st.set_page_config(
    page_title="NSE Confluence Engine | Technicals & Fundamentals",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. DATA SOURCES
# Direct read from your primary working scanner repository
TECHNICAL_PARQUET_URL = "https://raw.githubusercontent.com/prit7307/nse-working-scanner/main/data/nse_rolling_60d.parquet"
SECTOR_MASTER_URL = "https://raw.githubusercontent.com/prit7307/nse-working-scanner/main/nse_sector_master.csv"
LOCAL_FUNDAMENTAL_FILE = "nse_confluence_picks_20261030.csv"


def clean_column_name(col: str) -> str:
    """Normalizes Screener.in multi-space header formats."""
    c = re.sub(r"\s+", " ", str(col)).strip().upper()
    c = c.replace(" RS.CR.", " (CR)").replace(" RS.", "")
    return c


@st.cache_data(ttl=300)
def load_and_merge_datasets():
    # --- A. Load Technical Parquet ---
    try:
        raw_tech = pd.read_parquet(TECHNICAL_PARQUET_URL)
        raw_tech["DATE"] = pd.to_datetime(raw_tech["DATE"])
        latest_date = raw_tech["DATE"].max()
        tech_df = raw_tech[raw_tech["DATE"] == latest_date].copy()

        # Clean numeric fields
        for col in ["CLOSE", "PREV_CLOSE", "HIGH", "LOW", "DELIV_QTY", "TOTTRDQTY"]:
            if col in tech_df.columns:
                tech_df[col] = pd.to_numeric(tech_df[col], errors="coerce").fillna(0.0)

        # Technical Indicators: DSI & Mansfield RS
        rolling_deliv = raw_tech.groupby("SYMBOL")["DELIV_QTY"].transform(
            lambda s: s.rolling(50, min_periods=1).mean()
        ).replace(0, np.nan)
        tech_df["DSI"] = (tech_df["DELIV_QTY"] / rolling_deliv.loc[tech_df.index]).fillna(1.0).round(2)

        benchmark = raw_tech.groupby("DATE")["CLOSE"].transform("median").replace(0, np.nan)
        raw_tech["RAW_RS"] = (raw_tech["CLOSE"] / benchmark).fillna(1.0)
        raw_tech["SMA50_RS"] = raw_tech.groupby("SYMBOL")["RAW_RS"].transform(
            lambda s: s.rolling(50, min_periods=1).mean()
        ).replace(0, np.nan)
        tech_df["MANSFIELD_RS"] = (((raw_tech.loc[tech_df.index, "RAW_RS"] / raw_tech.loc[tech_df.index, "SMA50_RS"]) - 1.0) * 100.0).fillna(0.0).round(2)

        # NR7 Contraction
        raw_tech["RANGE"] = (raw_tech["HIGH"] - raw_tech["LOW"]).round(2)
        min_7d = raw_tech.groupby("SYMBOL")["RANGE"].transform(lambda s: s.rolling(7, min_periods=1).min())
        tech_df["NR7"] = (raw_tech.loc[tech_df.index, "RANGE"] <= min_7d.loc[tech_df.index]) & (raw_tech.loc[tech_df.index, "RANGE"] > 0)

        # Price Change %
        if "PCT_CHANGE" not in tech_df.columns or tech_df["PCT_CHANGE"].abs().sum() == 0:
            tech_df["PCT_CHANGE"] = np.where(
                tech_df["PREV_CLOSE"] > 0,
                ((tech_df["CLOSE"] - tech_df["PREV_CLOSE"]) / tech_df["PREV_CLOSE"] * 100.0).round(2),
                0.0
            )

        # Accumulation Alert Classification
        tech_df["Accumulation_Alert"] = np.where(
            (tech_df["DSI"] >= 1.5) & (tech_df["PCT_CHANGE"] > 0) & (tech_df["NR7"]),
            "STRONG ACCUMULATION",
            np.where((tech_df["DSI"] >= 1.5) & (tech_df["PCT_CHANGE"] > 0), "DELIVERY SURGE",
            np.where(tech_df["NR7"], "NR7 SQUEEZE", "NEUTRAL"))
        )

        tech_df["SYMBOL"] = tech_df["SYMBOL"].astype(str).str.strip().str.upper()
    except Exception as e:
        st.error(f"Error loading technical parquet: {e}")
        return pd.DataFrame(), None

    # --- B. Load Sector Master ---
    try:
        sec_df = pd.read_csv(SECTOR_MASTER_URL)
        sec_df["SYMBOL"] = sec_df["SYMBOL"].astype(str).str.strip().str.upper()
        tech_df = tech_df.merge(sec_df[["SYMBOL", "SECTOR"]], on="SYMBOL", how="left")
        tech_df["SECTOR"] = tech_df["SECTOR"].fillna("Broad Market / Small & Mid Cap")
    except Exception:
        tech_df["SECTOR"] = "Broad Market / Small & Mid Cap"

    # --- C. Load Fundamental Screener Dataset ---
    if os.path.exists(LOCAL_FUNDAMENTAL_FILE):
        fund_df = pd.read_csv(LOCAL_FUNDAMENTAL_FILE)
        # Normalize column spacing
        fund_df.columns = [clean_column_name(c) for c in fund_df.columns]
        fund_df["SYMBOL"] = fund_df["SYMBOL"].astype(str).str.strip().str.upper()

        # Drop duplicate technical columns from CSV to favor live exchange data
        overlap_cols = [c for c in ["CLOSE", "PCT_CHANGE", "DELIV_QTY", "DSI", "MANSFIELD_RS"] if c in fund_df.columns]
        fund_df = fund_df.drop(columns=overlap_cols)

        # Merge Technicals + Fundamentals
        unified_df = tech_df.merge(fund_df, on="SYMBOL", how="inner")
    else:
        unified_df = tech_df

    return unified_df, latest_date


df, active_session = load_and_merge_datasets()

if df.empty:
    st.error("No data could be processed. Please verify internet connection and repository paths.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.title("💎 Confluence Filters")
st.sidebar.caption(f"Active Session: **{active_session.strftime('%d %b %Y')}**")

search_q = st.sidebar.text_input("🔍 Search Symbol or Company:").strip().upper()

# Score Filter
if "CONFLUENCE_SCORE" in df.columns:
    min_score = st.sidebar.slider("Minimum Confluence Score (0–5):", 0, 5, 2, 1)
    df = df[df["CONFLUENCE_SCORE"] >= min_score]

# DSI Filter
min_dsi = st.sidebar.slider("Min Delivery Surge Index (DSI):", 0.5, 6.0, 1.2, 0.1)
df = df[df["DSI"] >= min_dsi]

# ROCE Filter
roce_col = [c for c in df.columns if "ROCE" in c]
if roce_col:
    min_roce = st.sidebar.slider("Min ROCE %:", 0.0, 50.0, 15.0, 1.0)
    df = df[pd.to_numeric(df[roce_col[0]], errors="coerce").fillna(0) >= min_roce]

# Sector Filter
if "SECTOR" in df.columns:
    sectors = sorted(df["SECTOR"].unique().tolist())
    selected_sec = st.sidebar.multiselect("Filter by Sector:", options=sectors, default=sectors)
    df = df[df["SECTOR"].isin(selected_sec)]

# Search Symbol
if search_q:
    sym_match = df["SYMBOL"].str.contains(search_q, na=False)
    comp_match = df["COMPANY"].str.upper().str.contains(search_q, na=False) if "COMPANY" in df.columns else False
    df = df[sym_match | comp_match]

# --- MAIN DASHBOARD HEADER ---
st.title("💎 NSE Confluence Screener: Institutional Volume + Quality Fundamentals")
st.markdown(f"Screening **{len(df):,}** high-conviction companies meeting active confluence criteria.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Screened Universe", f"{len(df):,} Stocks")
k2.metric("Accumulation Alerts", len(df[df["Accumulation_Alert"].isin(["STRONG ACCUMULATION", "DELIVERY SURGE"])]))
k3.metric("Avg ROCE %", f"{df[roce_col[0]].mean():.1f}%" if roce_col else "N/A")
k4.metric("Avg Delivery Surge", f"{df['DSI'].mean():.2f}x")

st.divider()

dash_tabs = st.tabs([
    "📋 Actionable Confluence Watchlist",
    "📈 Confluence Distribution Matrix",
    "💾 Export Master Dataset"
])

# TAB 1: FULL UNIFIED WATCHLIST TABLE
with dash_tabs[0]:
    st.subheader("Complete Unified Technical & Fundamental Watchlist")

    # Column ordering: Identity -> Technicals -> Confluence -> Fundamentals
    preferred_order = [
        "SYMBOL", "COMPANY", "SECTOR", "CLOSE", "PCT_CHANGE", "DELIV_QTY", "DSI",
        "MANSFIELD_RS", "CONFLUENCE_SCORE", "FUNDAMENTAL_SIGNALS", "Accumulation_Alert",
        "P/E", "ROCE %", "QTR PROFIT VAR %", "QTR SALES VAR %", "MAR CAP (CR)", "DIV YLD %"
    ]
    final_cols = [c for c in preferred_order if c in df.columns]
    # Add any remaining screener columns at the end
    remaining = [c for c in df.columns if c not in final_cols and c not in ["S.NO.", "DATE", "HIGH", "LOW", "OPEN", "PREV_CLOSE", "TOTTRDQTY", "RANGE"]]
    final_cols.extend(remaining)

    format_dict = {
        "CLOSE": "₹{:,.2f}",
        "PCT_CHANGE": "{:+.2f}%",
        "DELIV_QTY": "{:,}",
        "DSI": "{:.2f}x",
        "MANSFIELD_RS": "{:+.2f}",
        "P/E": "{:.1f}",
        "ROCE %": "{:.1f}%",
        "QTR PROFIT VAR %": "{:+.1f}%",
        "QTR SALES VAR %": "{:+.1f}%",
        "MAR CAP (CR)": "₹{:,.0f} Cr"
    }
    active_formats = {k: v for k, v in format_dict.items() if k in final_cols}

    st.dataframe(
        df[final_cols].sort_values(by=["CONFLUENCE_SCORE", "DSI"], ascending=[False, False]).style.format(active_formats),
        use_container_width=True,
        height=620
    )

# TAB 2: CONFLUENCE SCATTER MATRIX
with dash_tabs[1]:
    st.subheader("Delivery Surge Index vs. ROCE % (Fundamental Quality vs. Volume Action)")
    if roce_col:
        plot_df = df.dropna(subset=["DSI", roce_col[0]]).copy()
        fig_conf = px.scatter(
            plot_df,
            x="PCT_CHANGE",
            y="DSI",
            size=roce_col[0],
            color="Accumulation_Alert" if "Accumulation_Alert" in plot_df.columns else None,
            hover_name="SYMBOL",
            hover_data=["COMPANY", "CLOSE", "DSI", roce_col[0], "P/E", "CONFLUENCE_SCORE"],
            labels={"PCT_CHANGE": "Session Price Change (%)", "DSI": "Delivery Surge Index (x Avg)"},
            template="plotly_dark",
            height=580
        )
        fig_conf.add_hline(y=1.5, line_dash="dash", line_color="gray", annotation_text="Surge Line (1.5x)")
        fig_conf.add_vline(x=0.0, line_dash="dot", line_color="gray")
        fig_conf.update_traces(marker=dict(opacity=0.8, line=dict(width=0.7, color="white")))
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.info("ROCE column not detected for matrix plotting.")

# TAB 3: CSV DOWNLOAD
with dash_tabs[2]:
    st.subheader("Export Integrated Shortlist")
    st.markdown("Download the filtered confluence dataset with all combined technical and fundamental columns.")
    csv_bytes = df[final_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Confluence Master CSV",
        data=csv_bytes,
        file_name=f"nse_confluence_master_{active_session.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )