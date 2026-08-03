"""
Bonus - Clustering traders into behavioral archetypes using
whole-period account-summary features (frequency, size, win rate,
volatility of PnL, drawdown).
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")


def main():
    acc = pd.read_csv(os.path.join(OUT_DIR, "account_summary.csv"))

    features = ["trades_per_active_day", "avg_trade_size_usd", "overall_win_rate",
                "std_daily_pnl", "mean_daily_pnl", "max_drawdown"]
    X = acc[features].copy()
    X["std_daily_pnl"] = X["std_daily_pnl"].fillna(0)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    acc = acc.loc[X.index]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # choose k via simple inertia elbow check (k in [2..5]), then settle on 3 for interpretability
    inertias = []
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        inertias.append(km.inertia_)
    print("Inertia by k (2-5):", [round(i, 1) for i in inertias])

    k = 3
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    acc["cluster"] = km.fit_predict(Xs)

    profile = acc.groupby("cluster")[features + ["total_net_pnl"]].mean()
    counts = acc["cluster"].value_counts().sort_index()
    profile["n_traders"] = counts
    print("\nCluster profile (mean feature values):")
    print(profile)

    # Label clusters heuristically by their profile for readability
    labels = {}
    for c in profile.index:
        row = profile.loc[c]
        if profile.loc[c, "n_traders"] <= max(1, len(acc) // 20) and row["avg_trade_size_usd"] == profile["avg_trade_size_usd"].max():
            labels[c] = "High-volume outlier (whale trader)"
        elif row["mean_daily_pnl"] > 0 and row["std_daily_pnl"] < profile["std_daily_pnl"].median():
            labels[c] = "Steady performers"
        elif row["trades_per_active_day"] > profile["trades_per_active_day"].median():
            labels[c] = "High-frequency risk-takers"
        else:
            labels[c] = "Low-activity / inconsistent"
    acc["archetype"] = acc["cluster"].map(labels)
    profile["archetype"] = profile.index.map(labels)

    profile.to_csv(os.path.join(OUT_DIR, "bonus_cluster_profile.csv"))
    acc.to_csv(os.path.join(OUT_DIR, "bonus_account_clusters.csv"), index=False)

    # PCA plot for visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    palette = ["#2980b9", "#c0392b", "#27ae60", "#8e44ad", "#f39c12"]
    for c in sorted(acc["cluster"].unique()):
        mask = acc["cluster"] == c
        ax.scatter(coords[mask, 0], coords[mask, 1], label=f"{labels[c]} (n={mask.sum()})",
                   color=palette[c % len(palette)], s=70, alpha=0.8, edgecolor="white")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Trader behavioral archetypes (KMeans, k=3)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "bonus_clusters_pca.png"))
    plt.close()

    print("\nSaved cluster profile, account-cluster table, and PCA scatter chart.")


if __name__ == "__main__":
    main()
