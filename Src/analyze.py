"""
Part B - Analysis
Fear vs Greed performance/behavior comparison, trader segmentation,
and the charts/tables behind the insights in WRITEUP.md.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 110
COLORS = {"Fear": "#c0392b", "Greed": "#27ae60", "Neutral": "#7f8c8d"}


def load():
    daily = pd.read_csv(os.path.join(OUT_DIR, "daily_sentiment_merged.csv"), parse_dates=["date"])
    account_day = pd.read_csv(os.path.join(OUT_DIR, "account_day.csv"), parse_dates=["date"])
    account_summary = pd.read_csv(os.path.join(OUT_DIR, "account_summary.csv"))
    return daily, account_day, account_summary


def fear_greed_only(df):
    return df[df["sentiment_binary"].isin(["Fear", "Greed"])].copy()


def q1_performance_by_sentiment(daily):
    """Does performance differ between Fear vs Greed days?"""
    fg = fear_greed_only(daily)
    summary = fg.groupby("sentiment_binary").agg(
        n_days=("date", "count"),
        mean_total_net_pnl=("total_net_pnl", "mean"),
        median_total_net_pnl=("total_net_pnl", "median"),
        mean_per_trader_pnl=("mean_net_pnl", "mean"),
        mean_win_rate=("mean_win_rate", "mean"),
    ).reset_index()
    print("\n=== Q1: Performance by sentiment ===")
    print(summary.to_string(index=False))

    fear_pnl = fg.loc[fg.sentiment_binary == "Fear", "mean_net_pnl"].dropna()
    greed_pnl = fg.loc[fg.sentiment_binary == "Greed", "mean_net_pnl"].dropna()
    t, p = stats.ttest_ind(fear_pnl, greed_pnl, equal_var=False)
    print(f"\nWelch t-test (mean per-trader daily net PnL, Fear vs Greed): t={t:.3f}, p={p:.4f}")

    fear_wr = fg.loc[fg.sentiment_binary == "Fear", "mean_win_rate"].dropna()
    greed_wr = fg.loc[fg.sentiment_binary == "Greed", "mean_win_rate"].dropna()
    t2, p2 = stats.ttest_ind(fear_wr, greed_wr, equal_var=False)
    print(f"Welch t-test (mean daily win rate, Fear vs Greed): t={t2:.3f}, p={p2:.4f}")

    # chart
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    box_data = [fear_pnl, greed_pnl]
    axes[0].boxplot(box_data, labels=["Fear", "Greed"], showfliers=False,
                     patch_artist=True,
                     boxprops=dict(facecolor="#eee"))
    axes[0].axhline(0, color="grey", lw=0.8, ls="--")
    axes[0].set_title("Avg per-trader daily net PnL\nby sentiment")
    axes[0].set_ylabel("USD")

    axes[1].boxplot([fear_wr, greed_wr], labels=["Fear", "Greed"], showfliers=False,
                     patch_artist=True, boxprops=dict(facecolor="#eee"))
    axes[1].set_title("Avg daily win rate\nby sentiment")
    axes[1].set_ylabel("Win rate")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "q1_performance_by_sentiment.png"))
    plt.close()

    summary.to_csv(os.path.join(OUT_DIR, "q1_performance_by_sentiment.csv"), index=False)
    return summary, (t, p), (t2, p2)


def q2_behavior_by_sentiment(daily):
    """Do traders change behavior based on sentiment?"""
    fg = fear_greed_only(daily)
    summary = fg.groupby("sentiment_binary").agg(
        mean_trades_per_active_trader=("n_trades", lambda s: s.sum()),
        n_active_traders=("n_active_traders", "sum"),
        mean_trade_size_usd=("avg_trade_size_usd", "mean"),
        mean_long_short_ratio=("long_short_ratio", "mean"),
    ).reset_index()
    summary["trades_per_trader_per_day"] = (
        fg.groupby("sentiment_binary")["n_trades"].sum() /
        fg.groupby("sentiment_binary")["n_active_traders"].sum()
    ).values
    print("\n=== Q2: Behavior by sentiment ===")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, col, title, ylab in [
        (axes[0], "avg_trade_size_usd", "Avg trade size", "USD"),
        (axes[1], "long_short_ratio", "Long/Short ratio", "ratio (>1 = long-biased)"),
        (axes[2], "n_trades", "Total trades on day", "count"),
    ]:
        vals = [fg.loc[fg.sentiment_binary == s, col].dropna() for s in ["Fear", "Greed"]]
        ax.boxplot(vals, labels=["Fear", "Greed"], showfliers=False,
                    patch_artist=True, boxprops=dict(facecolor="#eee"))
        ax.set_title(title)
        ax.set_ylabel(ylab)
        if col == "long_short_ratio":
            ax.axhline(1, color="grey", lw=0.8, ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "q2_behavior_by_sentiment.png"))
    plt.close()

    summary.to_csv(os.path.join(OUT_DIR, "q2_behavior_by_sentiment.csv"), index=False)
    return summary


def q3_segments(account_summary, account_day, daily):
    """Segment traders and check how each segment behaves in Fear vs Greed."""
    seg = account_summary.copy()

    # Segment 1: leverage/exposure tier via avg trade size (proxy - dataset has no explicit leverage field)
    seg["size_tier"] = pd.qcut(seg["avg_trade_size_usd"], 2, labels=["Low size", "High size"])

    # Segment 2: frequency tier
    seg["freq_tier"] = pd.qcut(seg["trades_per_active_day"], 2, labels=["Infrequent", "Frequent"])

    # Segment 3: consistency (winners vs inconsistent), via std of daily PnL relative to mean
    seg["consistency_tier"] = np.where(
        (seg["mean_daily_pnl"] > 0) & (seg["std_daily_pnl"] < seg["std_daily_pnl"].median()),
        "Consistent winner",
        np.where(seg["mean_daily_pnl"] > 0, "Volatile winner", "Inconsistent/losing")
    )

    print("\n=== Q3: Segments (whole-period account summary) ===")
    for col in ["size_tier", "freq_tier", "consistency_tier"]:
        print(f"\n-- {col} --")
        print(seg.groupby(col)[["total_net_pnl", "overall_win_rate", "max_drawdown"]].mean())

    seg.to_csv(os.path.join(OUT_DIR, "account_segments.csv"), index=False)

    # How does each segment perform on Fear vs Greed days specifically?
    ad = account_day.merge(seg[["Account", "size_tier", "freq_tier", "consistency_tier"]], on="Account")
    sentiment_lookup = daily[["date", "sentiment_binary"]]
    ad = ad.merge(sentiment_lookup, on="date", how="inner")
    ad_fg = ad[ad["sentiment_binary"].isin(["Fear", "Greed"])]

    seg_sentiment = ad_fg.groupby(["size_tier", "sentiment_binary"])["net_pnl"].mean().unstack()
    print("\n-- Size tier: mean daily net PnL by sentiment --")
    print(seg_sentiment)
    seg_sentiment.to_csv(os.path.join(OUT_DIR, "q3_size_tier_by_sentiment.csv"))

    freq_sentiment = ad_fg.groupby(["freq_tier", "sentiment_binary"])["n_trades"].mean().unstack()
    print("\n-- Frequency tier: mean daily trade count by sentiment --")
    print(freq_sentiment)
    freq_sentiment.to_csv(os.path.join(OUT_DIR, "q3_freq_tier_by_sentiment.csv"))

    # chart: net pnl by size tier x sentiment
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    seg_sentiment[["Fear", "Greed"]].plot(kind="bar", ax=ax, color=[COLORS["Fear"], COLORS["Greed"]])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Mean daily net PnL by size tier and sentiment")
    ax.set_ylabel("USD")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "q3_size_tier_pnl.png"))
    plt.close()

    return seg, seg_sentiment, freq_sentiment


def extra_charts(daily):
    fg = fear_greed_only(daily).sort_values("date")
    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.plot(fg["date"], fg["total_net_pnl"], color="#2c3e50", lw=1, label="Total daily net PnL")
    ax1.axhline(0, color="grey", lw=0.7, ls="--")
    ax1.set_ylabel("Total net PnL (USD)")
    ax2 = ax1.twinx()
    ax2.scatter(fg["date"], fg["fg_value"], color=fg["sentiment_binary"].map(COLORS), s=8, alpha=0.6)
    ax2.set_ylabel("Fear & Greed index value")
    ax1.set_title("Daily net PnL vs. Fear/Greed index over time")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "timeseries_pnl_vs_sentiment.png"))
    plt.close()


def main():
    daily, account_day, account_summary = load()
    q1_performance_by_sentiment(daily)
    q2_behavior_by_sentiment(daily)
    q3_segments(account_summary, account_day, daily)
    extra_charts(daily)
    print("\nAll figures saved to outputs/figures/")


if __name__ == "__main__":
    main()
