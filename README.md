# Trader-Performance-vs-Market-Sentiment
## 1. Methodology

**Data.** `historical_data.csv` (211,224 Hyperliquid trades, 32 accounts, 246
symbols, 2023-05-01 to 2025-05-01) was merged with `fear_greed_index.csv`
(daily Bitcoin Fear & Greed index, 2018–2025) on calendar date, at IST. Both
files arrived clean — 0 missing values, 0 duplicate rows — so cleaning was
limited to timestamp parsing and deriving a standardized long/short flag
(`trade_bias`) from the `Direction` field, falling back to `Side` where
`Direction` was a generic Buy/Sell rather than an explicit Open/Close.

**Alignment.** Trades were bucketed to calendar day and merged against the
sentiment table; 479 of 480 trading days had a sentiment match (one day fell
outside the sentiment file's range).

**Metrics built (Part A):**
- *Account × day*: trade count, net PnL (closed PnL minus fees), win rate,
  average trade size, long/short trade counts.
- *Market-wide daily*: aggregated across active traders, merged with
  sentiment class (5-class: Extreme Fear → Extreme Greed, and a
  Fear/Greed/Neutral binary collapse used for the headline comparisons).
- *Account (whole-period)*: total/mean/std of daily PnL, win rate, trades per
  active day, and a max-drawdown proxy (largest peak-to-trough dip in
  cumulative daily net PnL).

**A note on "leverage."** The dataset does not include an explicit
leverage/margin field. Rather than fabricate one, trade size in USD is used
throughout as the exposure/risk-appetite proxy (a trader putting more notional
per trade is taking on more risk per trade, which is the behaviorally
relevant thing leverage would otherwise proxy for). This is stated explicitly
wherever "size tier" is used instead of "leverage tier."

**Statistical testing.** Welch's t-test (unequal variance) was used for the
Fear vs. Greed comparisons, since the two groups have very different sample
sizes (105 Fear-days vs. 307 Greed-days).

## 2. Insights

### Insight 1 — Fear days are less common but disproportionately profitable, though the market-wide difference isn't statistically significant

| Sentiment | Days | Avg. daily total net PnL (all traders) | Avg. per-trader daily net PnL | Avg. daily win rate |
|---|---|---|---|---|
| Fear | 105 | $37,904 | $2,791 | 84.3% |
| Greed | 307 | $15,554 | $2,109 | 83.8% |

Mean per-trader PnL is ~32% higher on Fear days, but a Welch t-test on the
daily average gives **t = 0.85, p = 0.39** — not significant at conventional
thresholds. Win rate is essentially flat (**t = 0.20, p = 0.85**). Read this
as: *this trader population does not perform reliably worse in Fear regimes*
— if anything, results skew better — but the sample of Fear days (105) isn't
large enough, and the day-to-day variance in this data isn't small enough, to
call it a real edge on its own. See `outputs/figures/q1_performance_by_sentiment.png`.

### Insight 2 — Traders get *more* long-biased during Fear, not less, and trade more often

| Sentiment | Trades/trader/day | Avg. trade size | Long/Short ratio |
|---|---|---|---|
| Fear | 105.4 | $7,061 | 2.89 |
| Greed | 76.9 | $6,658 | 1.97 |

This is the counterintuitive finding: naive intuition says "Fear → traders
go short / de-risk." Here, the long/short ratio is *higher* during Fear
(2.89 longs per short) than during Greed (1.97). Trade frequency is also
~37% higher during Fear. Two consistent readings: (a) this trader cohort
leans contrarian/dip-buying during Fear regimes, or (b) Fear periods in this
window coincided with sharp, tradable BTC moves that this cohort actively
worked (more trades either direction), and their existing long bias simply
carried through. Either way, "traders quietly step back and go risk-off in
Fear" is not what the data shows. See `outputs/figures/q2_behavior_by_sentiment.png`.

### Insight 3 — The sentiment effect is not uniform: it flips by trader size tier

| Size tier | Fear avg. daily net PnL | Greed avg. daily net PnL |
|---|---|---|
| Low trade-size traders | $2,532 | **$4,555** |
| High trade-size traders | **$9,220** | $3,196 |

Large-notional traders roughly triple their average daily PnL in Fear
regimes relative to Greed; small-notional traders do the opposite — they do
*better* in Greed. This is the most actionable pattern in the dataset: a
single "traders do better/worse in Fear" headline hides two opposite
sub-stories. See `outputs/figures/q3_size_tier_pnl.png`.

### Insight 4 — Frequent traders react far more sharply to sentiment than infrequent ones

| Frequency tier | Fear: trades/day | Greed: trades/day |
|---|---|---|
| Infrequent traders | 33.8 | 38.5 (roughly flat) |
| Frequent traders | 179.9 | 132.2 (+36% in Fear) |

Infrequent traders' activity barely moves with sentiment. Frequent traders
ramp activity substantially in Fear — consistent with Insight 2 (more longs,
more trades) and suggesting this segment is actively trading the volatility
that typically accompanies Fear regimes, rather than sitting it out.

### Insight 5 (segmentation) — Consistency, not raw activity, separates winners from losers

Segmenting the 32 accounts by mean/variance of daily PnL:

| Segment | n | Total net PnL | Win rate | Max drawdown |
|---|---|---|---|---|
| Consistent winners (positive mean, low variance) | 15 | $108,749 | 88.0% | -$17,879 |
| Volatile winners (positive mean, high variance) | 13 | $669,714 | 85.2% | -$59,120 |
| Inconsistent/losing | 4 | **-$71,606** | 73.1% | **-$161,424** |

Only 4 of 32 accounts were net losers over the period, but they lost far more
consistently and drew down ~9x deeper than the consistent-winner group.
Win rate alone is a weak signal here — the losing segment still won 73% of
closed trades, but their losses were large enough (poor risk sizing on the
losing 27%) to erase the edge. See `outputs/bonus_cluster_profile.csv` for
the unsupervised version of this same story (KMeans found the same three
archetypes without being told the win/lose split).

## 3. Actionable strategy ideas

**Rule 1 — Scale size tier exposure with the sentiment regime, not against it uniformly.**
Because the sentiment effect flips by size tier (Insight 3), a single
blanket rule ("de-risk in Fear") is wrong for large-notional traders in this
cohort. Concretely: for the high-trade-size segment, maintain or increase
exposure during Fear regimes (historically their best-performing regime
here); for the low-trade-size segment, do the opposite — hold size steady
or trim slightly in Fear and lean in during Greed. This should be validated
out-of-sample before sizing real capital against it — see caveats below.

**Rule 2 — Gate trade-frequency increases in Fear behind the frequent-trader segment, not the whole book.**
Ramping activity in Fear only paid off, in this data, for the segment that
already trades frequently (Insight 4) — they're the ones actually capturing
the extra volatility. Infrequent traders showed almost no behavioral or
performance shift from sentiment, so encouraging them to trade *more* in
Fear isn't supported by the data and mainly adds fee drag and tail risk.

**Rule 3 (risk-management corollary, not a sentiment rule) — Cap drawdown by consistency segment, not by PnL alone.**
Insight 5 shows drawdown, not PnL or win rate, is what separates the
struggling segment from the rest. A hard per-trader max-drawdown circuit
breaker (e.g., pause/reduce-size after a drawdown beyond ~2x the
consistent-winner median) would have flagged the 4 losing accounts well
before their full -$161K average drawdown played out.

## 4. Bonus results

**Predictive model** (`src/predictive_model.py`) — Random Forest predicting
whether a trader's *next* active day will be net-profitable, from that
trader's current-day behavior + sentiment features (70% of trader-days in
this data are followed by a profitable day, so the target is imbalanced).
Test AUC: **0.631** (vs. 0.5 for random) — modest but real signal that
sentiment + behavior features carry some predictive value; test accuracy
(64%) actually trails the majority-class baseline (70%) because the model
trades some accuracy for better balance between classes (see
`classification_report` in `outputs/`). This should be read as a
proof-of-concept, not a deployable signal — 1,558 modeling rows across 32
accounts is a small, non-independent sample (many rows per trader), and no
walk-forward/temporal cross-validation was done.

**Clustering** (`src/clustering.py`) — KMeans (k=3) on trade frequency, trade
size, win rate, PnL volatility, mean PnL, and drawdown found three
archetypes that closely mirror Insight 5's supervised segmentation, without
using PnL sign as an input: **Steady performers** (22 traders, lower
frequency/size, high win rate, shallow drawdowns), **High-frequency
risk-takers** (9 traders, high frequency, higher variance, deeper
drawdowns), and one **high-volume outlier** (a single whale-sized trader,
kept as its own cluster rather than forced into the others). See
`outputs/figures/bonus_clusters_pca.png`.

**Streamlit dashboard** (`app/dashboard.py`) — interactive version of the
above: sentiment/date filters, the Fear vs. Greed boxplots, segment
breakdowns, and the cluster scatter, all reading from the same `outputs/`
tables the scripts produce.

## 5. Caveats

- 32 accounts is a small, non-random sample — this is one platform's
  historical trader set, not a representative market sample. Segment-level
  findings (Insights 3–5) should be treated as hypotheses, not proven rules.
- No leverage/margin field exists in the data; trade size (USD) stands in as
  the exposure proxy throughout, which is a real but imperfect substitute.
- The Fear/Greed index is a market-wide daily label, not synchronized to
  intraday trade timing — a trader's trades on a "Fear" day may have
  happened before the sentiment shifted that day.
- The predictive model and clustering are demonstrations of feasibility, not
  validated trading systems; strategy Rules 1–3 above should be backtested
  out-of-sample before any capital allocation is changed on their basis.

