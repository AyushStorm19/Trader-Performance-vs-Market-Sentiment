# 📈 Trader Performance vs Market Sentiment
### Primetrade.ai – Data Science Internship Round-0 Assignment

> An end-to-end data analysis project exploring how **Bitcoin Fear & Greed sentiment** influences **Hyperliquid trader behavior and profitability**.

---

## 🚀 Project Overview

This project investigates whether market sentiment affects trader performance and decision-making.

Using **211,224 Hyperliquid trades** combined with the **Bitcoin Fear & Greed Index**, the analysis explores:

- 📊 Profitability during Fear vs Greed markets
- 📈 Trading behavior changes
- 👥 Trader segmentation
- 🤖 Bonus predictive modeling & clustering
- 📉 Actionable trading strategies backed by data

---

## 📂 Repository Structure

```text
primetrade/
│
├── data/
│   ├── historical_data.csv
│   └── fear_greed_index.csv
│
├── src/
│   ├── prepare_data.py
│   ├── analyze.py
│   ├── predictive_model.py
│   └── clustering.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── app/
│   └── dashboard.py
│
├── outputs/
│   ├── charts/
│   ├── account_summary.csv
│   ├── account_segments.csv
│   └── daily_sentiment_merged.csv
│
├── WRITEUP.md
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

### 1️⃣ Bitcoin Fear & Greed Index

- Daily market sentiment
- 2018–2025
- Features:
  - Date
  - Sentiment Score
  - Classification

### 2️⃣ Hyperliquid Historical Trades

- **211,224 trades**
- **32 trader accounts**
- **246 trading pairs**
- Period:
  **May 2023 – May 2025**

Includes

- Account
- Symbol
- Execution Price
- Trade Size
- Side
- Closed PnL
- Fees
- Timestamp

---

# ⚙️ Setup

```bash
git clone <repo>

cd primetrade

python -m venv .venv

source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

# ▶️ Run the Project

### Data Preparation

```bash
python src/prepare_data.py
```

### Analysis

```bash
python src/analyze.py
```

### Bonus Model

```bash
python src/predictive_model.py
```

### Clustering

```bash
python src/clustering.py
```

### Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

### Dashboard

```bash
streamlit run app/dashboard.py
```

---

# 🧹 Data Preparation

✔ Missing Values Check

✔ Duplicate Detection

✔ Timestamp Parsing

✔ Daily Date Alignment

✔ Feature Engineering

Generated metrics include:

- Daily Net PnL
- Win Rate
- Trade Count
- Average Trade Size
- Long/Short Ratio
- Drawdown Proxy

---

# 📈 Analysis Performed

## Performance vs Sentiment

- Fear vs Greed profitability
- Win Rate comparison
- Drawdown comparison

## Behavioral Analysis

- Trading frequency
- Position size
- Long/Short bias
- Risk appetite

## Trader Segmentation

- High vs Low Trade Size
- Frequent vs Infrequent Traders
- Consistent Winners vs Losing Traders

---

# 💡 Key Findings

### 📌 Fear Days showed higher average trader profitability

Although not statistically significant, average trader PnL was approximately **32% higher** during Fear periods.

---

### 📌 Traders became more active during Fear

Instead of reducing risk, traders:

- traded more frequently
- maintained larger exposure
- increased long bias

---

### 📌 Sentiment affected trader groups differently

Large-position traders outperformed during Fear, while smaller traders achieved better results during Greed.

---

### 📌 Consistency mattered more than win rate

Winning traders were distinguished by lower drawdowns rather than exceptionally high win rates.

---

# 🎯 Strategy Recommendations

✅ Increase exposure selectively for experienced large-position traders during Fear.

✅ Encourage higher trade frequency only for traders who historically perform well in volatile markets.

✅ Monitor drawdown more closely than win rate when evaluating trader performance.

---

# 🤖 Bonus

- Random Forest profitability prediction
- K-Means trader clustering
- Streamlit dashboard

---

# 📦 Outputs

```
outputs/

├── daily_sentiment_merged.csv
├── account_summary.csv
├── account_segments.csv
├── figures/
├── bonus_model_results.csv
└── bonus_cluster_profile.csv
```

---

# 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit

---

# 📄 Documentation

For complete methodology, statistical testing, detailed insights, and strategy recommendations, see:

📘 **WRITEUP.md**

---

# 👨‍💻 Author

Prepared as part of the **Primetrade.ai Data Science Internship Assignment**.
