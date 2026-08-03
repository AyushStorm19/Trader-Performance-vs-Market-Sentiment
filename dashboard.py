"""
Streamlit dashboard - Trader Performance vs Market Sentiment
Run with:  streamlit run app/dashboard.py
(run from the project root so the relative ../outputs path resolves)
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

BASE = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE, "..", "outputs")

st.set_page_config(page_title="Trader Performance vs Sentiment", layout="wide")


@st.cache_data
def load_data():
    daily = pd.read_csv(os.path.join(OUT_DIR, "daily_sentiment_merged.csv"), parse_dates=["date"])
    account_day = pd.read_csv(os.path.join(OUT_DIR, "account_day.csv"), parse_dates=["date"])
    account_summary = pd.read_csv(os.path.join(OUT_DIR, "account_summary.csv"))
    segments_path = os.path.join(OUT_DIR, "account_segments.csv")
    segments = pd.read_csv(segments_path) if os.path.exists(segments_path) else None
    clusters_path = os.path.join(OUT_DIR, "bonus_account_clusters.csv")
    clusters = pd.read_csv(clusters_path) if os.path.exists(clusters_path) else None
    return daily, account_day, account_summary, segments, clusters


daily, account_day, account_summary, segments, clusters = load_data()

st.title("Trader Performance vs. Bitcoin Market Sentiment")
st.caption("Hyperliquid trade data merged with the Fear & Greed index")

# --- Sidebar filters ---
st.sidebar.header("Filters")
sentiment_options = sorted(daily["fg_class"].dropna().unique().tolist())
selected_sentiments = st.sidebar.multiselect("Sentiment class", sentiment_options, default=sentiment_options)
date_min, date_max = daily["date"].min(), daily["date"].max()
date_range = st.sidebar.date_input("Date range", (date_min, date_max), min_value=date_min, max_value=date_max)

df = daily[daily["fg_class"].isin(selected_sentiments)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df["date"] >= start) & (df["date"] <= end)]

# --- KPI row ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Days in view", f"{len(df):,}")
c2.metric("Total net PnL", f"${df['total_net_pnl'].sum():,.0f}")
c3.metric("Avg daily win rate", f"{df['mean_win_rate'].mean():.1%}")
c4.metric("Avg long/short ratio", f"{df['long_short_ratio'].mean():.2f}")

st.divider()

# --- Time series ---
st.subheader("Daily net PnL vs. sentiment over time")
fig = px.scatter(df, x="date", y="total_net_pnl", color="fg_class",
                  color_discrete_map={"Extreme Fear": "#7b241c", "Fear": "#c0392b",
                                       "Neutral": "#7f8c8d", "Greed": "#27ae60",
                                       "Extreme Greed": "#145a32"},
                  title=None, hover_data=["fg_value", "n_active_traders"])
fig.add_hline(y=0, line_dash="dash", line_color="grey")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Fear vs Greed comparison ---
st.subheader("Fear vs. Greed comparison")
fg = df[df["sentiment_binary"].isin(["Fear", "Greed"])]
col1, col2 = st.columns(2)
with col1:
    fig1 = px.box(fg, x="sentiment_binary", y="mean_net_pnl",
                   color="sentiment_binary",
                   color_discrete_map={"Fear": "#c0392b", "Greed": "#27ae60"},
                   points=False, title="Avg per-trader daily net PnL")
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = px.box(fg, x="sentiment_binary", y="mean_win_rate",
                   color="sentiment_binary",
                   color_discrete_map={"Fear": "#c0392b", "Greed": "#27ae60"},
                   points=False, title="Avg daily win rate")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- Segments ---
if segments is not None:
    st.subheader("Trader segments")
    seg_choice = st.selectbox("Segment by", ["size_tier", "freq_tier", "consistency_tier"])
    seg_summary = segments.groupby(seg_choice)[["total_net_pnl", "overall_win_rate", "max_drawdown"]].mean().reset_index()
    fig3 = px.bar(seg_summary, x=seg_choice, y="total_net_pnl", color=seg_choice,
                   title=f"Total net PnL by {seg_choice}")
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(segments, use_container_width=True, height=250)

st.divider()

# --- Clusters ---
if clusters is not None:
    st.subheader("Behavioral archetypes (clustering)")
    fig4 = px.scatter(clusters, x="trades_per_active_day", y="mean_daily_pnl",
                       size="avg_trade_size_usd", color="archetype",
                       hover_data=["Account", "overall_win_rate", "max_drawdown"],
                       title="Traders by activity, PnL, and archetype")
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.subheader("Raw daily table")
st.dataframe(df, use_container_width=True, height=300)
