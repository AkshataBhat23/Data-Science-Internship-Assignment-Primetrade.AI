import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Trader Dashboard", layout="wide")

st.title("Trader Performance vs Market Sentiment")
st.markdown("Dashboard for analyzing Hyperliquid trader behaviors based on the Fear/Greed Index.")

@st.cache_data
def load_and_preprocess_data():
    fear_greed_df = pd.read_csv(r'D:\\Python_Learning\\DS Internship Assignment\\fear_greed_index.csv')
    historical_df = pd.read_csv(r'D:\\Python_Learning\\DS Internship Assignment\\historical_data.csv')

    fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date'], format='%d-%m-%Y', errors='coerce').fillna(
        pd.to_datetime(fear_greed_df['date'], errors='coerce')
    )
    fear_greed_df['date_only'] = fear_greed_df['date'].dt.date

    historical_df['Timestamp'] = pd.to_datetime(historical_df['Timestamp'], unit='ms')
    historical_df['date_only'] = historical_df['Timestamp'].dt.date

    merged_df = historical_df.merge(fear_greed_df, on='date_only', how='left')
    cleaned_df = merged_df.dropna(subset=['classification']).copy()

    cleaned_df['Size USD'] = pd.to_numeric(cleaned_df['Size USD'], errors='coerce')
    cleaned_df['Closed PnL'] = pd.to_numeric(cleaned_df['Closed PnL'], errors='coerce')
    cleaned_df['is_profit'] = (cleaned_df['Closed PnL'] > 0).astype(int)

    daily_metrics = cleaned_df.groupby(['Account', 'date_only']).agg(
        daily_pnl=('Closed PnL', 'sum'),
        total_trades=('Account', 'count'),
        win_rate=('is_profit', 'mean'),
        avg_trade_size=('Size USD', 'mean'),
        sentiment=('classification', 'first'),
        value=('value', 'first')
    ).reset_index()

    trader_summary = daily_metrics.groupby('Account').agg(
        total_usd_vol=('avg_trade_size', 'sum'),
        total_trades_all=('total_trades', 'sum'),
        overall_win_rate=('win_rate', 'mean')
    ).reset_index()
    
    # Clustering
    features = trader_summary[['total_usd_vol', 'total_trades_all', 'overall_win_rate']]
    scaled_features = StandardScaler().fit_transform(features)
    trader_summary['Cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(scaled_features)

    return daily_metrics, trader_summary

daily_metrics, trader_summary = load_and_preprocess_data()

st.sidebar.header("Filter Data")
sentiment_filter = st.sidebar.multiselect("Select Market Sentiment", options=daily_metrics['sentiment'].unique(), default=daily_metrics['sentiment'].unique())

filtered_metrics = daily_metrics[daily_metrics['sentiment'].isin(sentiment_filter)]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Days Recorded", filtered_metrics['date_only'].nunique())
with col2:
    st.metric("Avg Daily PnL", f"${filtered_metrics['daily_pnl'].mean():.2f}")
with col3:
    st.metric("Avg Trade Size", f"${filtered_metrics['avg_trade_size'].mean():.2f}")

st.subheader("Performance Distributions by Sentiment (SymLog Scale)")
fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(data=daily_metrics, x='sentiment', y='daily_pnl', ax=ax, showfliers=False)
ax.set_yscale('symlog', linthresh=1000)
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
st.pyplot(fig)

st.subheader("Trader Archetypes (K-Means Clustering)")
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.scatterplot(data=trader_summary, x='total_trades_all', y='overall_win_rate', hue='Cluster', palette='Set1', ax=ax2)
ax2.set_title("Total Trades vs Overall Win Rate")
st.pyplot(fig2)

st.markdown("### Actionable Output")
st.markdown("- **Strategy 1:** Reduce sizing during Extended Fear.")
st.markdown("- **Strategy 2:** Tighten algos on frequent trades to avoid deep drawdowns in Fear conditions.")
