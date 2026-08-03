"""
Part A - Data preparation
Loads the Hyperliquid trade log and the Bitcoin Fear/Greed index,
cleans them, aligns them by date, and builds the metrics used in
Part B (trader-day level, account level, and daily aggregate level).
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_raw():
    trades = pd.read_csv(os.path.join(DATA_DIR, "historical_data.csv"))
    sentiment = pd.read_csv(os.path.join(DATA_DIR, "fear_greed_index.csv"))
    return trades, sentiment


def report_shape(df, name):
    print(f"\n--- {name} ---")
    print(f"rows: {len(df):,}  cols: {df.shape[1]}")
    print(f"missing values total: {df.isna().sum().sum()}")
    print(f"duplicate rows: {df.duplicated().sum()}")


def clean_trades(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()
    df.columns = [c.strip() for c in df.columns]

    # Parse timestamp (IST) -> a plain date for daily alignment
    df["ts"] = pd.to_datetime(df["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce")
    before = len(df)
    df = df.dropna(subset=["ts"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with unparseable timestamps")

    df["date"] = df["ts"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    # Standardise a trade-level "side" (long/short) from Direction where possible,
    # falling back to Side (BUY/SELL) for rows without an explicit Open/Close tag.
    long_dirs = {"Open Long", "Close Short", "Short > Long"}
    short_dirs = {"Open Short", "Close Long", "Long > Short"}
    df["trade_bias"] = np.where(
        df["Direction"].isin(long_dirs), "Long",
        np.where(df["Direction"].isin(short_dirs), "Short",
                 np.where(df["Side"] == "BUY", "Long", "Short"))
    )

    # A trade is a "closing" trade (realises PnL) if Closed PnL != 0
    df["is_close"] = df["Closed PnL"] != 0
    df["win"] = df["Closed PnL"] > 0

    return df


def clean_sentiment(sentiment: pd.DataFrame) -> pd.DataFrame:
    df = sentiment.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"])
    # Collapse 5-class classification into a binary Fear/Greed view (Neutral kept separate)
    df["sentiment_binary"] = df["classification"].map({
        "Extreme Fear": "Fear", "Fear": "Fear",
        "Neutral": "Neutral",
        "Greed": "Greed", "Extreme Greed": "Greed",
    })
    return df[["date", "value", "classification", "sentiment_binary"]].rename(
        columns={"value": "fg_value", "classification": "fg_class"}
    )


def build_account_day(trades: pd.DataFrame) -> pd.DataFrame:
    """Account x Date level metrics."""
    g = trades.groupby(["Account", "date"])
    out = g.agg(
        n_trades=("Trade ID", "count"),
        n_closing_trades=("is_close", "sum"),
        gross_pnl=("Closed PnL", "sum"),
        wins=("win", "sum"),
        total_fees=("Fee", "sum"),
        avg_trade_size_usd=("Size USD", "mean"),
        total_volume_usd=("Size USD", "sum"),
        long_trades=("trade_bias", lambda s: (s == "Long").sum()),
        short_trades=("trade_bias", lambda s: (s == "Short").sum()),
    ).reset_index()

    out["net_pnl"] = out["gross_pnl"] - out["total_fees"]
    out["win_rate"] = np.where(out["n_closing_trades"] > 0,
                                out["wins"] / out["n_closing_trades"], np.nan)
    out["long_short_ratio"] = np.where(out["short_trades"] > 0,
                                        out["long_trades"] / out["short_trades"],
                                        np.where(out["long_trades"] > 0, np.inf, np.nan))
    return out


def build_daily(account_day: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """Market-wide daily aggregate, merged with sentiment."""
    daily = account_day.groupby("date").agg(
        n_active_traders=("Account", "nunique"),
        n_trades=("n_trades", "sum"),
        total_net_pnl=("net_pnl", "sum"),
        mean_net_pnl=("net_pnl", "mean"),
        median_net_pnl=("net_pnl", "median"),
        mean_win_rate=("win_rate", "mean"),
        total_volume_usd=("total_volume_usd", "sum"),
        avg_trade_size_usd=("avg_trade_size_usd", "mean"),
        total_long_trades=("long_trades", "sum"),
        total_short_trades=("short_trades", "sum"),
    ).reset_index()
    daily["long_short_ratio"] = daily["total_long_trades"] / daily["total_short_trades"].replace(0, np.nan)

    merged = daily.merge(sentiment, on="date", how="inner")
    return merged


def build_account_summary(account_day: pd.DataFrame) -> pd.DataFrame:
    """Whole-period per-account summary, used for trader segmentation."""
    g = account_day.groupby("Account")
    out = g.agg(
        active_days=("date", "nunique"),
        total_trades=("n_trades", "sum"),
        total_closing_trades=("n_closing_trades", "sum"),
        total_net_pnl=("net_pnl", "sum"),
        mean_daily_pnl=("net_pnl", "mean"),
        std_daily_pnl=("net_pnl", "std"),
        overall_win_rate=("wins", "sum"),
        total_volume_usd=("total_volume_usd", "sum"),
        avg_trade_size_usd=("avg_trade_size_usd", "mean"),
    ).reset_index()
    total_closes = g["n_closing_trades"].sum().values
    total_wins = g["wins"].sum().values
    out["overall_win_rate"] = np.where(total_closes > 0, total_wins / total_closes, np.nan)
    out["trades_per_active_day"] = out["total_trades"] / out["active_days"]
    # drawdown proxy: worst cumulative dip in daily net PnL over the period
    def max_drawdown(pnl_series):
        cum = pnl_series.cumsum()
        running_max = cum.cummax()
        dd = cum - running_max
        return dd.min()
    dd = account_day.sort_values("date").groupby("Account")["net_pnl"].apply(max_drawdown)
    out = out.merge(dd.rename("max_drawdown"), on="Account")
    return out


def main():
    trades_raw, sentiment_raw = load_raw()
    report_shape(trades_raw, "historical_data.csv (raw)")
    report_shape(sentiment_raw, "fear_greed_index.csv (raw)")

    trades = clean_trades(trades_raw)
    sentiment = clean_sentiment(sentiment_raw)

    account_day = build_account_day(trades)
    daily = build_daily(account_day, sentiment)
    account_summary = build_account_summary(account_day)

    print(f"\nDate range of trades: {trades['date'].min().date()} to {trades['date'].max().date()}")
    print(f"Unique accounts: {trades['Account'].nunique()}")
    print(f"Daily (merged with sentiment) rows: {len(daily)}  "
          f"(days in trade data with no sentiment match: "
          f"{account_day['date'].nunique() - len(daily)})")

    account_day.to_csv(os.path.join(OUT_DIR, "account_day.csv"), index=False)
    daily.to_csv(os.path.join(OUT_DIR, "daily_sentiment_merged.csv"), index=False)
    account_summary.to_csv(os.path.join(OUT_DIR, "account_summary.csv"), index=False)
    trades.to_parquet(os.path.join(OUT_DIR, "trades_clean.parquet"), index=False)

    print("\nSaved: account_day.csv, daily_sentiment_merged.csv, account_summary.csv, trades_clean.parquet")
    return trades, sentiment, account_day, daily, account_summary


if __name__ == "__main__":
    main()
