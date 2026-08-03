"""
Bonus - Simple predictive model.
Predicts whether a trader's NEXT active day will be net-profitable
(binary), using today's sentiment + today's behavior features.
This is a realistic, honestly-scoped bonus: not a trading signal,
just a demonstration that sentiment + behavior carries some signal.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")


def build_features():
    account_day = pd.read_csv(os.path.join(OUT_DIR, "account_day.csv"), parse_dates=["date"])
    daily = pd.read_csv(os.path.join(OUT_DIR, "daily_sentiment_merged.csv"), parse_dates=["date"])
    sentiment = daily[["date", "fg_value", "sentiment_binary", "fg_class"]]

    df = account_day.merge(sentiment, on="date", how="inner")
    df = df.sort_values(["Account", "date"])

    # Target: was the trader's NEXT active day net-profitable?
    df["next_net_pnl"] = df.groupby("Account")["net_pnl"].shift(-1)
    df["target_next_profitable"] = (df["next_net_pnl"] > 0).astype(int)
    df = df.dropna(subset=["next_net_pnl"])

    feature_cols_num = ["n_trades", "win_rate", "avg_trade_size_usd", "total_volume_usd",
                         "long_short_ratio", "net_pnl", "fg_value"]
    feature_cols_cat = ["sentiment_binary"]

    df[feature_cols_num] = df[feature_cols_num].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols_num)

    X = df[feature_cols_num + feature_cols_cat]
    y = df["target_next_profitable"]
    return X, y, feature_cols_num, feature_cols_cat, df


def main():
    X, y, num_cols, cat_cols, df = build_features()
    print(f"Modeling rows: {len(X)}  positive rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ], remainder="passthrough")

    clf = Pipeline([
        ("pre", pre),
        ("model", RandomForestClassifier(n_estimators=300, max_depth=6,
                                          random_state=42, class_weight="balanced")),
    ])
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    print(f"\nTest AUC: {auc:.3f}")
    print(classification_report(y_test, pred))

    # Baseline: always predict majority class
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    print(f"Baseline (majority class) accuracy: {baseline_acc:.3f}")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
    ax.set_title(f"Next-day profitability prediction\nAUC = {auc:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "bonus_model_roc.png"))
    plt.close()

    # Feature importance
    model = clf.named_steps["model"]
    ohe = clf.named_steps["pre"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    all_feature_names = cat_feature_names + num_cols
    importances = pd.Series(model.feature_importances_, index=all_feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(6.5, 5))
    importances.plot(kind="barh", ax=ax, color="#34495e")
    ax.set_title("Feature importance - next-day profitability model")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "bonus_model_feature_importance.png"))
    plt.close()

    results = pd.DataFrame({
        "metric": ["test_auc", "baseline_majority_class_accuracy", "n_train", "n_test"],
        "value": [auc, baseline_acc, len(X_train), len(X_test)],
    })
    results.to_csv(os.path.join(OUT_DIR, "bonus_model_results.csv"), index=False)
    print("\nSaved ROC curve, feature importance chart, and results table.")


if __name__ == "__main__":
    main()
