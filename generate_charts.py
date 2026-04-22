import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

sns.set_theme(style="whitegrid")
os.makedirs("output_charts", exist_ok=True)

# 1. Load Data
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

ls_counts = cleaned_df.groupby(['Account', 'date_only', 'Side']).size().unstack(fill_value=0).reset_index()
buys = ls_counts.get('BUY', 0)
sells = ls_counts.get('SELL', 0)
ls_counts['long_short_ratio'] = np.where(sells == 0, buys, buys / sells)
daily_metrics = daily_metrics.merge(ls_counts[['Account', 'date_only', 'long_short_ratio']], on=['Account', 'date_only'], how='left')

# Chart 1: Performance vs Sentiment
sentiment_perf = daily_metrics.groupby('sentiment').agg(
    avg_daily_pnl=('daily_pnl', 'mean'),
    avg_win_rate=('win_rate', 'mean')
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(data=sentiment_perf, x='sentiment', y='avg_daily_pnl', ax=axes[0])
axes[0].set_title('Avg Daily PnL vs Sentiment')
sns.barplot(data=sentiment_perf, x='sentiment', y='avg_win_rate', ax=axes[1])
axes[1].set_title('Avg Win Rate vs Sentiment')
plt.tight_layout()
plt.savefig('output_charts/1_performance_vs_sentiment.png')
plt.close()

# Chart 2: Behavior vs Sentiment
behavior_metrics = daily_metrics.groupby('sentiment').agg(
    avg_trade_freq=('total_trades', 'mean'),
    avg_trade_size=('avg_trade_size', 'mean'),
    avg_ls_ratio=('long_short_ratio', 'mean')
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(data=behavior_metrics, x='sentiment', y='avg_trade_freq', ax=axes[0])
axes[0].set_title('Avg Trade Frequency vs Sentiment')
sns.barplot(data=behavior_metrics, x='sentiment', y='avg_trade_size', ax=axes[1])
axes[1].set_title('Avg Position Size USD vs Sentiment')
sns.barplot(data=behavior_metrics, x='sentiment', y='avg_ls_ratio', ax=axes[2])
axes[2].set_title('Long/Short Ratio vs Sentiment')
plt.tight_layout()
plt.savefig('output_charts/2_behavior_vs_sentiment.png')
plt.close()

# Summary & Clustering
trader_summary = daily_metrics.groupby('Account').agg(
    total_usd_vol=('avg_trade_size', 'sum'),
    total_trades_all=('total_trades', 'sum'),
    overall_win_rate=('win_rate', 'mean')
).reset_index()

features = trader_summary[['total_usd_vol', 'total_trades_all', 'overall_win_rate']]
scaled_features = StandardScaler().fit_transform(features)
kmeans = KMeans(n_clusters=3, random_state=42)
trader_summary['Cluster'] = kmeans.fit_predict(scaled_features)

# Chart 3: Clustering Archetypes
fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(data=trader_summary, x='total_trades_all', y='overall_win_rate', hue='Cluster', palette='Set1', ax=ax)
ax.set_title('Trader Archetypes: Frequency vs Win Rate')
plt.tight_layout()
plt.savefig('output_charts/3_trader_archetypes_clusters.png')
plt.close()

# Chart 4: Streamlit Boxplot equivalent
fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(data=daily_metrics, x='sentiment', y='daily_pnl', ax=ax, showfliers=False)
ax.set_yscale('symlog', linthresh=1000)
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
ax.set_title('PnL Distribution by Market Sentiment (SymLog Scale)')
plt.tight_layout()
plt.savefig('output_charts/4_pnl_distribution_boxplot.png')
plt.close()

# ML Feature Importance
df_ml = daily_metrics.sort_values(by=['Account', 'date_only']).copy()
df_ml['next_day_pnl'] = df_ml.groupby('Account')['daily_pnl'].shift(-1)
df_ml = df_ml.dropna(subset=['next_day_pnl'])
df_ml['target'] = (df_ml['next_day_pnl'] > 0).astype(int)
features = ['value', 'long_short_ratio', 'total_trades', 'avg_trade_size', 'win_rate']
X = df_ml[features].fillna(0)
y = df_ml['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Chart 5: Feature Importance
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind='bar', ax=ax, color='teal')
ax.set_title('Feature Importance for Next Day Profitability')
plt.tight_layout()
plt.savefig('output_charts/5_ml_feature_importance.png')
plt.close()

print("Charts successfully saved to output_charts directory!")
