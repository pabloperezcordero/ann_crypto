# Cryptocurrency Directional Prediction with Artificial Neural Network

**Predicting significant next-day price movements in SOL/BTC using a feed-forward ANN**

This project implements a binary classification model to predict whether the **next day's return** in the SOL/BTC pair (Solana priced in Bitcoin) will be **strongly positive (> +0.2%)** or **strongly negative (< -0.2%)**, using daily OHLCV data from Binance.

The model is a simple **Artificial Neural Network (ANN)** built with TensorFlow/Keras, combined with extensive feature engineering, temporal lagging, technical indicators, and post-processing filters to create a basic **long/flat trading strategy**.

## Project Overview

- **Pair**: SOL/BTC (Binance daily candles)
- **Target**: Binary label → 1 if next-day return > +0.2%, 0 if < -0.2%  
  (Neutral/small movements |r| ≤ 0.2% are filtered out to reduce noise)
- **Approach**: Time-series supervised learning with feed-forward ANN  
  → No sequence models (LSTM/GRU) — explicit lag features instead
- **Objective**: Demonstrate end-to-end ML pipeline for crypto direction prediction + simple backtesting

## Features Engineered

- Price-based: body size, range, closing position
- Returns: log returns, simple returns, ROC (3/5/10 periods)
- Volatility: rolling std of log returns (5 & 10 periods)
- Volume: SMA(5), volume ratio vs SMA(5)
- Technical indicators: SMA(5/20/50), price-to-SMA ratios, SMA cross ratios
- Momentum & oscillators: RSI(14)
- Lagged features: lags 1–3 on selected columns (temporal memory for ANN)
- Position filters: conviction threshold (|position| < 0.2 → flat), volatility-based filtering

## Results Summary (Out-of-Sample Test Set)

- **Test period**: Latest chronological hold-out (~20% of data, true future)
- **Model performance (threshold 0.5)**:
  - Accuracy: ~52.6%
  - ROC AUC: ~0.506
  - Very close to random / majority-class baseline
- **Strategy performance (long/flat + filters)**:
  - Total compounded return: **+11.2%**
  - Sharpe ratio (annualized): **~0.66**
  - → Modest positive risk-adjusted edge in this backtest period

The strategy shows ability to capture some upside while avoiding large drawdowns (common benefit of trend/volatility filters in crypto).

**Important reality check**:  
Short-term directional prediction in cryptocurrency markets is extremely difficult. AUC values ~0.50–0.54 and Sharpe ratios ~0.6–1.0 are typical realistic out-of-sample results for daily models — even strong published academic/commercial attempts rarely exceed this consistently.

## Repository Structure
