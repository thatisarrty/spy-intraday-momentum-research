
# SPY Intraday Momentum Strategy - WUTIS Case Study

## Overview

This project reproduces and evaluates an intraday momentum strategy
for the SPDR S&P 500 ETF Trust (SPY), inspired by:

Zarattini, Aziz and Barbon,
**Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY).**

The project was completed as an algorithmic-trading case study.

The objectives were to:

1. Understand and reproduce the main economic and trading logic of the paper.
2. Backtest the strategy using realistic transaction-cost assumptions.
3. Evaluate parameter robustness.
4. Use a chronological train/test framework.
5. Develop and evaluate an original paper-inspired strategy variation.
6. Maintain readable and reproducible research code.

## Data

The analysis uses one-minute SPY OHLCV data.

Available sample:

- Start: 2018-01-02
- End: 2024-04-30
- Regular trading session: 09:30–15:59 US Eastern Time
- 390 one-minute bars per complete trading day

The available sample is shorter than the sample used in the original
paper. Therefore, the objective is a replication of the methodology
on an independent dataset rather than an exact reproduction of the reported performance.

## Train / Test Split

A chronological split is used.

Training / development period:

2018-01-02 through 2022-12-31

Out-of-sample test period:

2023-01-01 through 2024-04-30

The test period was kept untouched while the original strategy
variations were developed.

The original strategy specification was frozen before evaluating the
2023–2024 test sample.

## Paper Strategy

### Noise Area

For every intraday time-of-day, the strategy estimates the typical
absolute movement from the market open using the previous 14 trading
days.

The resulting upper and lower boundaries define a dynamic Noise Area.

Price inside the Noise Area is interpreted as supply-demand equilibrium.

A movement above the upper boundary indicates abnormal buying pressure.

A movement below the lower boundary indicates abnormal selling pressure.

### Gap Adjustment

The boundaries incorporate both the current session open and the
previous session close to account for overnight gaps.

### Trading Decisions

Trading decisions are evaluated every 30 minutes from 10:00 through
15:30.

### Paper Variants Implemented

1. Opposite-band stop with fixed sizing
2. Current-band + VWAP stop with fixed sizing
3. Current-band + VWAP stop with dynamic volatility sizing

The third version is referred to throughout the project as
"Paper Dynamic" and is the main benchmark.

## Dynamic Position Sizing

Position size is based on recent 14-day SPY daily volatility.

Target daily volatility: 2%

Maximum leverage: 4x

Higher realized market volatility therefore reduces exposure, while
lower volatility increases exposure subject to the leverage cap.

## Transaction Costs

The backtest applies:

- Commission: $0.0035 per share per transaction
- Slippage: $0.001 per share per transaction

Slippage is applied adversely to entry and exit execution prices.

## Look-Ahead Prevention

All rolling signals are shifted so that the current day's information
is not included in estimates used to trade that same day.

Time-of-day Noise Area calculations use only historical trading days.

The volatility-scaling estimate uses only prior daily returns.

Session VWAP used for execution decisions is lagged by one minute.

## Robustness Analysis

The Paper Dynamic strategy was evaluated across alternative:

Noise Area lookbacks:

5, 10, 14, 20, 30, 60, 90 days

Volatility multipliers:

0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00

Robustness analysis was performed using the training sample.

The paper specification of:

- Lookback = 14
- Volatility Multiplier = 1.0

was retained rather than replacing it with the ex-post best-performing
parameters.

## Original Strategy Research

### Original Strategy v1 — Adaptive Volatility Multiplier

The first extension allowed the Noise Area volatility multiplier to
change according to recent market volatility.

Economic hypothesis:

High-volatility environments may produce more false boundary breakouts,
so wider thresholds may improve signal quality.

Result:

The strategy underperformed Paper Dynamic on the training sample in
risk-adjusted return and drawdown.

Decision:

Rejected before out-of-sample testing.

### Original Strategy v2 — Confirmed Breakout

The second extension required two consecutive semi-hourly breakout
signals in the same direction before opening a new position.

Economic hypothesis:

Persistent movement outside the Noise Area may provide stronger evidence
of a genuine supply-demand imbalance than a single boundary breach.

Training result:

The confirmation rule substantially reduced the number of trades and
portfolio volatility while retaining a Sharpe ratio close to the Paper
Dynamic strategy.

The specification was therefore frozen before out-of-sample testing.

Out-of-sample result:

The strategy failed to generalize.

Although confirmation reduced turnover and volatility, it materially
reduced return, Sharpe ratio and trade-level profitability.

The evidence suggests that delaying entry until a second confirmation
often misses an economically important portion of the intraday momentum
move.

Decision:

Rejected after out-of-sample evaluation.

## Main Research Conclusion

The Paper Dynamic strategy remains the strongest specification tested
in this dataset.

The original-strategy experiments indicate an important trade-off:

greater signal confirmation reduces trading frequency and exposure,
but waiting for additional confirmation can sacrifice the timeliness
that appears important to the intraday momentum effect.

No further strategy variants were developed after observing the
out-of-sample results in order to avoid tuning on the test set and overfitting.

## Performance Metrics

Portfolio-level metrics include:

- Total Return
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Positive Day Ratio

Trade-level metrics include:

- Number of Trades
- Trade Win Rate
- Average Trade PnL
- Median Trade PnL
- Best / Worst Trade
- Profit Factor

The Sharpe ratio assumes a zero risk-free rate.

## Project Structure

```text
Algo Trading WUTIS/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── SPY_1min_clean.csv
│       └── SPY_1min_features.csv
│
├── output/
│   ├── results/
│   └── figures/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── features.py
│   ├── backtest.py
│   ├── metrics.py
│   ├── robustness.py
│   └── extensions.py
│
├── research.ipynb
├── run_pipeline.py
├── README.md
└── requirements.txt
