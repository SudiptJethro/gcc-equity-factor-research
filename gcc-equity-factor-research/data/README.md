# Data

This folder holds only **derived research outputs**: factor portfolio returns, Fama-MacBeth estimates, IC series, quintile assignments and summary tables. The dashboard and tests run from these files.

**Raw market data is not included.** Prices, daily bars, returns and fundamentals came from Yahoo Finance (via `yfinance`) and TradingView. Both providers' terms restrict redistribution. To rebuild the raw data and every layer yourself:

```bash
pip install -r requirements-pipeline.txt
# run every step from the repository root
python Layer_1/main.py                      # prices, filters, monthly returns
python Layer_1/fetch_tv_data.py             # optional: fill ADX / DFM gaps from TradingView
python Layer_2/factor_construction.py       # factor scores, quintile portfolios
python Layer_3_analysis/fama_macbeth_3a.py  # Fama-MacBeth regressions
python Layer_3_analysis/backtest.py         # attribution, IC decay, sub-periods
python Layer_3_analysis/portfolio.py        # two-sleeve portfolio
python Layer_3_analysis/layer3c_vol_surface.py  # simulated options overlay
```

The fundamentals are a single snapshot taken in December 2025, not point-in-time history. A fresh download will give different value and quality scores. See the limitations section of the main README.

| File | What it is |
|---|---|
| `portfolio_returns.csv` | Monthly quintile and long-short returns for each factor (full GCC universe) |
| `layer3b_returns.parquet` | Monthly returns of the two-sleeve multi-factor portfolio: gross, net of costs, vol-scaled, with circuit breaker |
| `quintile_assignments.parquet` | Monthly quintile membership by stock and factor |
| `factor_snapshot.csv` | Latest factor z-scores per stock |
| `universe.csv`, `ticker_names.csv` | Stocks that passed the filters, with names |
| `factors/fm_summary.csv`, `factors/fm_coefficients.csv` | Fama-MacBeth premia and Newey-West t-stats |
| `factors/backtest_summary.csv` | Quintile spreads, Sharpe ratios, IC statistics by exchange and factor |
| `factors/ic_monthly.csv` | Monthly rank IC per factor |
| `factors/sub_period_analysis.csv` | Results split into pre-COVID, COVID, post-COVID and recent periods |
| `factors/quintile_returns_*.csv` | Quintile return series by exchange and factor |
| `factors/layer3b_*.csv`, `factors/layer3c_*.csv` | Portfolio tear sheets and the simulated options-overlay results |
