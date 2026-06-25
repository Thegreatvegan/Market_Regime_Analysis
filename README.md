# Market Regime Lab

An interactive **market regime detection platform** built with Streamlit. It downloads OHLCV data via [yfinance](https://github.com/ranaroussi/yfinance), engineers volatility/momentum features, fits unsupervised regime models (K-Means, Gaussian Mixture, Hidden Markov Model), labels economic regimes, and backtests a simple regime-filtered allocation strategy.

## Quick start

```bash
cd market-regime-lab
pip install -r requirements.txt
python -m streamlit run app.py
```

> **Windows note:** If `streamlit` is not recognized, use `python -m streamlit run app.py` instead.
> Pip often installs scripts to a folder that is not on your PATH.

## Project structure

```
market-regime-lab/
├── app.py              # Streamlit UI
├── requirements.txt
├── README.md
└── src/
    ├── data.py         # yfinance download + caching
    ├── features.py     # feature engineering + scaling
    ├── models.py       # KMeans, GMM, HMM + regime labeling
    ├── backtest.py     # strategy simulation + metrics
    └── plots.py        # Plotly charts
```

## Features

| Category | Details |
|----------|---------|
| **Tickers** | SPY, QQQ, BTC-USD, NVDA, TSLA, AAPL |
| **Features** | Daily/log returns, 20d vol, 20d/50d momentum, drawdown, volume z-score |
| **Models** | K-Means, GMM, Gaussian HMM (4 components) |
| **Regimes** | Bull, Bear, Crisis, Sideways (+ Recovery when drawdown is still elevated) |
| **Backtest** | Buy & hold vs regime strategy (hold in Bull/Recovery/Sideways, cash in Crisis/Bear) |
| **Metrics** | Total return, annualized return/vol, Sharpe ratio, max drawdown |

## Methodology

### 1. Feature engineering

Given close price \(P_t\) and volume \(V_t\):

| Feature | Formula |
|---------|---------|
| Daily return | \(r_t = P_t / P_{t-1} - 1\) |
| Log return | \(\ln(P_t / P_{t-1})\) |
| 20-day volatility | \(\mathrm{std}(r_{t-19:t}) \times \sqrt{252}\) |
| 20-day momentum | \(P_t / P_{t-20} - 1\) |
| 50-day momentum | \(P_t / P_{t-50} - 1\) |
| Drawdown | \(P_t / \max_{s \le t} P_s - 1\) |
| Volume z-score | \((V_t - \mu_{20}) / \sigma_{20}\) |

Features are standardized with `StandardScaler` (zero mean, unit variance) before clustering.

### 2. Regime models

All three models partition the standardized feature space into **4 latent states**:

- **K-Means** — partitions observations by Euclidean distance to centroids. Fast and interpretable; assumes spherical clusters.
- **Gaussian Mixture Model (GMM)** — soft clustering with full covariance matrices; captures elliptical clusters and overlapping regimes.
- **Gaussian HMM** — models regimes as a Markov chain: each day’s state depends on the previous state. Produces a **transition matrix** \(A_{ij} = P(S_{t+1}=j \mid S_t=i)\).

### 3. Automatic regime labeling

Cluster IDs are mapped to economic labels using centroid statistics:

1. **Crisis** — highest mean 20-day volatility (stress / crash-like)
2. **Bear** — lowest mean daily return among remaining clusters
3. **Bull** — highest mean daily return among remaining clusters
4. **Sideways** — remaining cluster (muted return and volatility)

If the Bull cluster still shows a deep average drawdown (\(< -8\%\)), it is relabeled **Recovery** (positive drift but not yet healed).

### 4. Backtesting

**Buy & hold:** fully invested every day.

**Regime strategy:**

- **Long (100% exposure):** Bull, Recovery, Sideways
- **Cash (0% exposure):** Crisis, Bear

Positions are **lagged by one day** so today’s regime determines tomorrow’s allocation (no look-ahead bias).

Daily strategy return: \(r^{\mathrm{strat}}_t = w_{t-1} \cdot r_t\) where \(w_t \in \{0, 1\}\).

Equity curve: \(E_t = \prod_{s \le t} (1 + r^{\mathrm{strat}}_s)\), normalized to start at \$1.

### 5. Performance metrics

| Metric | Definition |
|--------|------------|
| Total return | \(E_T - 1\) |
| Annualized return | \((1 + \text{total})^{252/n} - 1\) |
| Annualized volatility | \(\mathrm{std}(r) \times \sqrt{252}\) |
| Sharpe ratio | Ann. return / Ann. volatility (risk-free ≈ 0) |
| Max drawdown | \(\min_t (E_t / \max_{s \le t} E_s - 1)\) |

## UI tabs

1. **Overview** — price chart shaded by regime, timeline, occupancy pie
2. **Regime Model** — 2D feature scatter, cluster stats, HMM transition heatmap
3. **Backtest** — equity curves, drawdowns, metrics table
4. **Research Stats** — feature correlations, per-regime statistics, raw data preview

## Caching

- `download_ohlcv` is cached via `@st.cache_data` (1-hour TTL) in `src/data.py`
- The full analysis pipeline in `app.py` is cached on ticker, date range, and model choice

## Disclaimer

This project is for **research and education only**. Past regime labels and backtests do not guarantee future performance. Not financial advice.

## License

MIT
