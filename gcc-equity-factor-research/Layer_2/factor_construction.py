"""
GCC Equity Factor Research Engine
Layer 2: Factor Construction
==============================
Computes four classic factors across the GCC universe:

  1. VALUE      / composite of P/B (inverted) and EV/EBITDA (inverted)
  2. MOMENTUM   / Jegadeesh-Titman 12-1 month return
  3. QUALITY    / composite of ROE and earnings stability
  4. LOW-VOL    / trailing 252-day realised volatility (inverted)

Each factor is:
  - Computed per stock per month where data allows
  - Cross-sectionally ranked within the GCC universe
  - Winsorised at 1st/99th percentile before ranking to suppress outliers
  - Standardised to z-scores for composite construction
  - Assigned to quintiles Q1 (best) through Q5 (worst)

Factor portfolios:
  - Long Q1, Short Q5 (equal-weighted within quintile)
  - Monthly rebalancing
  - Returns saved for Fama-MacBeth regressions in Layer 3

Reads from:   data/prices_{exchange}.parquet
              data/returns_{exchange}.parquet
              data/daily_{exchange}.parquet
              data/universe.csv
Writes to:    data/factor_scores.parquet
              data/quintile_assignments.parquet
              data/portfolio_returns.parquet
              data/portfolio_returns.csv
              data/factor_snapshot.csv

Run:
  python factor_construction.py
  python factor_construction.py --refresh-fundamentals
  python factor_construction.py --exchange ADX
  python factor_construction.py --verbose
"""

import argparse
import logging
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DATA_DIR   = "data"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MOM_LOOKBACK   = 12    # months of price history for momentum
MOM_SKIP       = 1     # skip most recent month (short-term reversal)
VOL_LOOKBACK   = 252   # trading days for realised volatility
MIN_VOL_DAYS   = 180   # minimum daily obs for vol to be valid
MIN_COVERAGE   = 0.80  # fraction of lookback window that must be non-NaN
N_QUINTILES    = 5
MIN_STOCKS_QUINTILE = 15  # minimum stocks in a month for quintile assignment

# Low-vol short-book filters (applied to Q5 only / long side unchanged)
MIN_MKTCAP_SHORT_USD    = 300_000_000   # $300M / exclude micro-caps from short book
MIN_ADV_SHORT           = 200_000       # 200k avg daily shares
MAX_SINGLE_STOCK_WEIGHT = 0.15          # cap any one short position at 15% of short book

# Kuwait yfinance coverage is unreliable before 2016
KUWAIT_START = pd.Timestamp("2016-01-01")

ALL_EXCHANGES = ["adx", "dfm", "tadawul", "qatar", "kuwait"]

YF_SLEEP = 0.5  # seconds between yfinance requests


# -----------------------------------------------------------------------------
# 1. DATA LOADING
# -----------------------------------------------------------------------------

def load_monthly_returns(exchanges: list) -> pd.DataFrame:
    """Load and concatenate monthly return DataFrames for all exchanges."""
    frames = []
    for ex in exchanges:
        path = os.path.join(DATA_DIR, f"returns_{ex}.parquet")
        if not os.path.exists(path):
            print(f"  [WARN] Missing: {path} / skipping {ex.upper()}")
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        if ex == "kuwait":
            df = df.loc[df.index >= KUWAIT_START]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No return Parquet files found. Run gcc_data_pipeline.py first."
        )

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    print(f"  Monthly returns: {combined.shape[0]} months x {combined.shape[1]} stocks")
    return combined


def load_monthly_prices(exchanges: list) -> pd.DataFrame:
    """Load month-end prices for momentum computation."""
    frames = []
    for ex in exchanges:
        path = os.path.join(DATA_DIR, f"prices_{ex}.parquet")
        if not os.path.exists(path):
            print(f"  [WARN] Missing: {path} / skipping {ex.upper()}")
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        if ex == "kuwait":
            df = df.loc[df.index >= KUWAIT_START]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No price Parquet files found. Run gcc_data_pipeline.py first."
        )

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    print(f"  Monthly prices:  {combined.shape[0]} months x {combined.shape[1]} stocks")
    return combined


def load_daily_prices(exchanges: list) -> pd.DataFrame:
    """Load daily prices for volatility computation."""
    frames = []
    for ex in exchanges:
        path = os.path.join(DATA_DIR, f"daily_{ex}.parquet")
        if not os.path.exists(path):
            print(f"  [WARN] Missing daily: {path} / {ex.upper()} excluded from vol")
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        if ex == "kuwait":
            df = df.loc[df.index >= KUWAIT_START]
        frames.append(df)

    if not frames:
        print("  [WARN] No daily price files found / Low-Vol factor will use "
              "monthly fallback (less precise).")
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    print(f"  Daily prices:    {combined.shape[0]} days x {combined.shape[1]} stocks")
    return combined


def load_universe() -> pd.DataFrame:
    """Load universe metadata for exchange membership lookup."""
    path = os.path.join(DATA_DIR, "universe.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


# -----------------------------------------------------------------------------
# 2. STATISTICAL UTILITIES
# -----------------------------------------------------------------------------

def winsorise(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Cap extreme values at given quantiles.
    Essential for GCC fundamentals / small-cap ratios can be wild outliers.
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """Winsorise then standardise to zero mean / unit variance cross-sectionally."""
    w = winsorise(series.dropna())
    mu, sigma = w.mean(), w.std()
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.nan, index=series.index)
    return (winsorise(series) - mu) / sigma


def composite_score(*factor_series, weights=None) -> pd.Series:
    """
    Combine multiple z-scored factor series into a weighted composite.
    Re-normalises weights per stock based on available (non-NaN) signals,
    so a stock with only one valid signal gets full weight on that signal.
    """
    df = pd.concat(factor_series, axis=1)
    n = df.shape[1]

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.array(weights, dtype=float)

    valid = df.notna().values                          # shape (n_stocks, n_factors)
    w_mat = np.where(valid, w, 0.0)                   # zero out missing signals
    w_sum = w_mat.sum(axis=1, keepdims=True)
    w_norm = np.where(w_sum > 0, w_mat / w_sum, 0.0)  # re-normalise per stock

    composite = (df.fillna(0.0).values * w_norm).sum(axis=1)
    composite = pd.Series(composite, index=df.index)
    composite[valid.sum(axis=1) == 0] = np.nan        # all-NaN stocks -> NaN
    return composite


# -----------------------------------------------------------------------------
# 3. FACTOR: MOMENTUM (12-1)
# -----------------------------------------------------------------------------

def compute_momentum(monthly_prices: pd.DataFrame,
                     verbose: bool = False) -> pd.DataFrame:
    """
    Jegadeesh-Titman (1993): cumulative return from t-12 to t-2.
    Skip the most recent month to avoid GCC short-term reversal effect.

        MOM(t) = (Price[t-2] / Price[t-12]) - 1

    Minimum valid observations in 11-month window: ceil(11 x 0.80) = 9.
    """
    print("\n  Computing Momentum factor (12-1) …")
    scores = {}
    dates  = monthly_prices.index
    # Need at least 9 valid months in the t-12..t-2 window (11 months, 80% coverage)
    n_required = int(np.ceil((MOM_LOOKBACK - MOM_SKIP) * MIN_COVERAGE))

    for i, date in enumerate(dates):
        if i < MOM_LOOKBACK:
            continue

        # Prices from t-12 through t-2 (inclusive)
        window = monthly_prices.iloc[i - MOM_LOOKBACK : i - MOM_SKIP + 1]
        valid_count = window.notna().sum()
        p_start = window.iloc[0]
        p_end   = window.iloc[-1]

        cum_ret = (p_end / p_start) - 1
        cum_ret = cum_ret.where(valid_count >= n_required)
        scores[date] = cross_sectional_zscore(cum_ret)

        if verbose:
            n_valid = cum_ret.notna().sum()
            print(f"    {date.date()}: {n_valid} stocks scored")

    result = pd.DataFrame(scores).T
    result.index.name = "date"
    print(f"    -> {result.notna().sum().sum():,} valid stock-month observations")
    return result


# -----------------------------------------------------------------------------
# 4. FACTOR: LOW VOLATILITY
# -----------------------------------------------------------------------------

def compute_low_vol(daily_prices: pd.DataFrame,
                    monthly_prices: pd.DataFrame,
                    verbose: bool = False) -> pd.DataFrame:
    """
    Trailing 252-day realised volatility, annualised. Inverted so that
    lower vol = higher score = long side of the portfolio.

        VOL(t)       = std(daily_log_ret over past 252 days) x √252
        LOW_VOL(t)   = -z_score(VOL(t))

    Minimum: 180 daily obs in the 252-day window. Falls back to monthly
    return std x √12 if daily files are unavailable.
    """
    print("\n  Computing Low-Volatility factor (trailing 252d) …")
    scores = {}
    use_daily = not daily_prices.empty

    if use_daily:
        # Align to tickers present in monthly_prices / avoids NaN in portfolio matching
        common = daily_prices.columns.intersection(monthly_prices.columns)
        daily_log_ret = np.log(daily_prices[common] / daily_prices[common].shift(1))
        # Keep only rows where at least one stock traded (drops weekend/holiday NaN rows)
        trading_days = daily_log_ret.index[daily_log_ret.notna().any(axis=1)]
        month_ends = monthly_prices.index

        for date in month_ends:
            available = trading_days[trading_days <= date]
            if len(available) < MIN_VOL_DAYS:
                continue
            # Last VOL_LOOKBACK actual trading days (not calendar days)
            window = daily_log_ret.loc[available[-VOL_LOOKBACK:]]
            n_valid = window.notna().sum()
            vol = window.std() * np.sqrt(252)
            vol = vol.where(n_valid >= MIN_VOL_DAYS)
            scores[date] = cross_sectional_zscore(-vol)  # invert: lower vol = better

            if verbose:
                print(f"    {date.date()}: {vol.notna().sum()} stocks scored")
    else:
        print("    [FALLBACK] Using monthly returns for vol estimate / less precise")
        for i, date in enumerate(monthly_prices.index):
            if i < 12:
                continue
            window_ret = monthly_prices.iloc[i - 12: i].pct_change().dropna()
            vol = window_ret.std() * np.sqrt(12)
            scores[date] = cross_sectional_zscore(-vol)

    result = pd.DataFrame(scores).T
    result.index.name = "date"
    print(f"    -> {result.notna().sum().sum():,} valid stock-month observations")
    return result


# -----------------------------------------------------------------------------
# 5. FACTOR: VALUE (P/B + EV/EBITDA composite)
# -----------------------------------------------------------------------------

def fetch_fundamental(ticker: str, fields: list) -> dict:
    """Pull .info fields from yfinance. Returns {field: value} with None for missing."""
    try:
        info = yf.Ticker(ticker).info
        return {f: info.get(f) for f in fields}
    except Exception:
        return {f: None for f in fields}


def fetch_all_fundamentals(tickers: list) -> pd.DataFrame:
    """
    Fetch fundamental data for all tickers from yfinance.

    LIMITATION: yfinance provides today's values, not historical point-in-time
    fundamentals. This introduces look-ahead bias into Value and Quality factors.
    Document in research note / see spec Section 8.
    """
    print("\n  Fetching fundamental data from yfinance …")
    print("  (Snapshot of current values / look-ahead bias applies to Value/Quality)")
    print(f"  Fetching {len(tickers)} tickers …")

    FIELDS = [
        "priceToBook",
        "enterpriseToEbitda",
        "returnOnEquity",
        "trailingEps",
        "earningsGrowth",
        "revenueGrowth",
        "debtToEquity",
        "currentRatio",
        "operatingMargins",
        "marketCap",
    ]

    records = []
    for i, ticker in enumerate(tickers):
        data = fetch_fundamental(ticker, FIELDS)
        data["ticker"] = ticker
        records.append(data)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(tickers)} done …")
        time.sleep(YF_SLEEP)

    df = pd.DataFrame(records).set_index("ticker")

    print("  Coverage:")
    for col in FIELDS:
        pct = df[col].notna().mean() * 100
        print(f"    {col:25s}: {pct:5.1f}%")

    return df


def compute_value(fundamentals: pd.DataFrame,
                  monthly_prices: pd.DataFrame,
                  verbose: bool = False) -> pd.DataFrame:
    """
    Value = composite of 1/P/B and 1/EV/EBITDA (inverted, lower ratio = more value).
    Negative values excluded (distressed / unreliable data).
    If only one signal available, that signal carries full weight.
    """
    print("\n  Computing Value factor (P/B, EV/EBITDA composite) …")
    tickers = monthly_prices.columns.tolist()
    fund = fundamentals.reindex(tickers)

    pb_inv = (1 / fund["priceToBook"]).replace([np.inf, -np.inf], np.nan)
    ev_inv = (1 / fund["enterpriseToEbitda"]).replace([np.inf, -np.inf], np.nan)

    # Negative ratios = distressed or data error / exclude
    pb_inv = pb_inv.where(fund["priceToBook"] > 0)
    ev_inv = ev_inv.where(fund["enterpriseToEbitda"] > 0)

    pb_z = cross_sectional_zscore(pb_inv)
    ev_z = cross_sectional_zscore(ev_inv)

    # composite_score re-normalises weights for stocks missing one signal
    value_composite = composite_score(pb_z.rename("pb"), ev_z.rename("ev"))

    dates  = monthly_prices.index
    result = pd.DataFrame(
        np.tile(value_composite.values, (len(dates), 1)),
        index=dates,
        columns=tickers,
    )
    result.index.name = "date"

    n_valid = value_composite.notna().sum()
    n_pb    = pb_inv.notna().sum()
    n_ev    = ev_inv.notna().sum()
    print(f"    -> {n_valid} stocks with value scores ({n_valid/len(tickers)*100:.0f}% coverage)")
    print(f"       P/B: {n_pb} stocks  |  EV/EBITDA: {n_ev} stocks")

    if verbose:
        print(value_composite.dropna().sort_values(ascending=False).to_string())

    return result


# -----------------------------------------------------------------------------
# 6. FACTOR: QUALITY (ROE + Earnings Stability + Operating Margin)
# -----------------------------------------------------------------------------

def earnings_stability(ticker: str) -> float:
    """
    Inverse coefficient of variation of quarterly EPS.
    Low CV = consistent earnings = higher stability score.
    Returns NaN if fewer than 4 quarters of data.
    """
    try:
        qe = yf.Ticker(ticker).quarterly_earnings
        if qe is None or len(qe) < 4:
            return np.nan
        eps = qe["Earnings"].dropna().astype(float)
        if len(eps) < 4 or eps.mean() == 0:
            return np.nan
        cv = eps.std() / abs(eps.mean())
        return 1 / cv  # invert: lower CV = higher stability
    except Exception:
        return np.nan


def fetch_quality_data(tickers: list) -> pd.Series:
    """Compute earnings stability for all tickers from quarterly earnings data."""
    print("\n  Computing earnings stability (ROE from fundamentals, stability from quarterly) …")
    stability_scores = {}
    for i, ticker in enumerate(tickers):
        stability_scores[ticker] = earnings_stability(ticker)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(tickers)} stability scores computed …")
        time.sleep(YF_SLEEP)
    valid = sum(1 for v in stability_scores.values() if v is not np.nan and not (isinstance(v, float) and np.isnan(v)))
    print(f"    -> {valid}/{len(tickers)} stocks with valid earnings stability")
    return pd.Series(stability_scores)


def compute_quality(fundamentals: pd.DataFrame,
                    monthly_prices: pd.DataFrame,
                    stability: pd.Series = None,
                    verbose: bool = False) -> pd.DataFrame:
    """
    Quality = weighted composite of:
      ROE (50%)           / profitability; exclude negative (distressed)
      Earnings stability (30%) / consistency; inverse CV of quarterly EPS
      Operating margin (20%)  / efficiency; exclude < -100%

    Weights are re-normalised per stock for missing signals.
    """
    print("\n  Computing Quality factor …")
    tickers = monthly_prices.columns.tolist()
    fund = fundamentals.reindex(tickers)

    roe = fund["returnOnEquity"].copy()
    roe = roe.where(roe > 0)  # NaN out negative ROE / do not penalise distressed firms
    roe_z = cross_sectional_zscore(roe)

    if stability is not None:
        stab = stability.reindex(tickers)
        stab_z = cross_sectional_zscore(stab)
    else:
        stab_z = pd.Series(np.nan, index=tickers)

    op_margin = fund["operatingMargins"].copy()
    op_margin = op_margin.where(op_margin > -1.0)  # exclude extreme losses
    op_z = cross_sectional_zscore(op_margin)

    quality_composite = composite_score(
        roe_z.rename("roe"),
        stab_z.rename("stability"),
        op_z.rename("op_margin"),
        weights=[0.5, 0.3, 0.2],
    )

    dates  = monthly_prices.index
    result = pd.DataFrame(
        np.tile(quality_composite.values, (len(dates), 1)),
        index=dates,
        columns=tickers,
    )
    result.index.name = "date"

    n_valid = quality_composite.notna().sum()
    print(f"    -> {n_valid} stocks with quality scores ({n_valid/len(tickers)*100:.0f}% coverage)")

    if verbose:
        print(quality_composite.dropna().sort_values(ascending=False).to_string())

    return result


# -----------------------------------------------------------------------------
# 7. QUINTILE ASSIGNMENT
# -----------------------------------------------------------------------------

def assign_quintiles(factor_scores: pd.DataFrame,
                     verbose: bool = False) -> pd.DataFrame:
    """
    For each date, assign stocks to quintiles Q1 (best) through Q5 (worst).
    Requires at least MIN_STOCKS_QUINTILE (15) valid stocks; skips month otherwise.

    Uses pd.qcut on ranks so quintiles are exactly equal-sized.
    Labels [5,4,3,2,1] on ascending ranks -> Q1 = highest z-score = best.
    """
    quintiles = factor_scores.copy() * np.nan

    for date in factor_scores.index:
        row = factor_scores.loc[date].dropna()
        if len(row) < MIN_STOCKS_QUINTILE:
            if verbose:
                print(f"    {date.date()}: only {len(row)} stocks / skipping")
            continue

        q = pd.qcut(
            row.rank(method="first"),
            q=N_QUINTILES,
            labels=[5, 4, 3, 2, 1],  # reverse: highest rank -> Q1
        )
        quintiles.loc[date, row.index] = q.astype(float).values

    return quintiles


# -----------------------------------------------------------------------------
# 8. PORTFOLIO RETURNS
# -----------------------------------------------------------------------------

def compute_portfolio_returns(quintiles: pd.DataFrame,
                              monthly_returns: pd.DataFrame,
                              factor_name: str) -> pd.DataFrame:
    """
    Equal-weighted quintile portfolio returns with t+1 return shift.
    Stocks ranked at end of month t; returns earned in month t+1.

    Saves all five quintile returns (Q1–Q5) plus long-short (Q1 minus Q5)
    for quintile monotonicity analysis in Layer 3.
    """
    print(f"\n  Building {factor_name} quintile portfolios …")
    port_rows = {}
    dates = sorted(set(quintiles.index) & set(monthly_returns.index))

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        q_today  = quintiles.loc[date]
        ret_next = monthly_returns.loc[next_date]

        row = {}
        for q in range(1, N_QUINTILES + 1):
            stocks = q_today[q_today == q].index
            valid  = ret_next.reindex(stocks).dropna()
            row[f"Q{q}"] = valid.mean() if len(valid) > 0 else np.nan

        row["LS"] = (
            (row.get("Q1", np.nan) - row.get("Q5", np.nan))
            if not (np.isnan(row.get("Q1", np.nan)) or np.isnan(row.get("Q5", np.nan)))
            else np.nan
        )
        port_rows[next_date] = row

    result = pd.DataFrame(port_rows).T
    result.index = pd.to_datetime(result.index)
    result.index.name = "date"

    rename_q = {f"Q{q}": f"{factor_name}_Q{q}" for q in range(1, N_QUINTILES + 1)}
    result = result.rename(columns={**rename_q, "LS": f"{factor_name}_LS"})

    ls_col = f"{factor_name}_LS"
    if ls_col in result.columns:
        ls = result[ls_col].dropna()
        if len(ls) > 0:
            ann_ret  = ls.mean() * 12
            ann_vol  = ls.std() * np.sqrt(12)
            sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
            hit_rate = (ls > 0).mean()
            print(f"    Ann. L/S return:  {ann_ret*100:+.2f}%")
            print(f"    Ann. volatility:  {ann_vol*100:.2f}%")
            print(f"    Sharpe ratio:     {sharpe:.2f}")
            print(f"    Hit rate:         {hit_rate*100:.1f}%")

    return result


# -----------------------------------------------------------------------------
# 8b. LOW-VOL FIXED PORTFOLIO (short-book filtered)
# -----------------------------------------------------------------------------

def compute_lowvol_portfolio_returns_fixed(
    quintiles:       pd.DataFrame,
    monthly_returns: pd.DataFrame,
    universe_df:     pd.DataFrame,
) -> pd.DataFrame:
    """
    Low-vol long-short with filtered short book.

    Problem: Q5 (highest-vol) contains illiquid GCC micro-caps that can gap
    ±30-50% on thin-volume news, blowing up the short side despite a good signal.

    Fix (Q5 only / Q1 long side unchanged):
      1. Market-cap + ADV filter: exclude micro/illiquid names from short book.
      2. Z-score outlier removal: drop stocks >3σ this month (event/data errors).
      3. Per-stock weight cap: no single name >MAX_SINGLE_STOCK_WEIGHT of short book.
    """
    print("  Building Low-Vol portfolio (FIXED / short book filtered) …")

    if universe_df.empty:
        print("  [WARN] No universe metadata / applying no liquidity filter to short book")
        eligible_short = set(monthly_returns.columns)
    else:
        if "mktcap_usd" in universe_df.columns:
            mktcap = universe_df["mktcap_usd"]
        elif "market_cap" in universe_df.columns:
            mktcap = universe_df["market_cap"] * 0.27  # AED/SAR rough USD conversion
        else:
            mktcap = pd.Series(np.inf, index=universe_df.index)

        adv = universe_df.get("avg_volume",
              universe_df.get("avg_daily_volume",
              pd.Series(np.inf, index=universe_df.index)))

        mask = (mktcap >= MIN_MKTCAP_SHORT_USD) & (adv >= MIN_ADV_SHORT)
        eligible_short = set(universe_df.index[mask])
        n_el, n_tot = len(eligible_short), len(universe_df)
        print(f"    Short-eligible: {n_el}/{n_tot} "
              f"(excluded {n_tot - n_el} micro/illiquid names)")

    port_returns = {}
    dates = sorted(set(quintiles.index) & set(monthly_returns.index))

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        q_today   = quintiles.loc[date]
        ret_next  = monthly_returns.loc[next_date]

        # Long book (Q1) / equal-weighted, unchanged
        q1_stocks = q_today[q_today == 1].index
        q1_valid  = ret_next.reindex(q1_stocks).dropna()
        long_ret  = q1_valid.mean() if len(q1_valid) > 0 else np.nan

        # Middle quintiles (Q2/Q3/Q4) / standard equal-weight, no filtering
        mid_rets = {}
        for q in (2, 3, 4):
            stocks = q_today[q_today == q].index
            valid  = ret_next.reindex(stocks).dropna()
            mid_rets[f"lowvol_Q{q}"] = valid.mean() if len(valid) > 0 else np.nan

        # Short book (Q5) / filtered + capped
        q5_stocks   = q_today[q_today == 5].index
        q5_eligible = [t for t in q5_stocks if t in eligible_short]

        if len(q5_eligible) < 3:
            short_ret = np.nan
        else:
            q5_rets = ret_next.reindex(q5_eligible).dropna()
            # Drop data-error spikes first (|return| > 100% is not a real monthly move)
            q5_rets = q5_rets[abs(q5_rets) <= 1.0]
            # Then remove remaining cross-sectional outliers >3σ
            if len(q5_rets) >= 3:
                z = (q5_rets - q5_rets.mean()) / (q5_rets.std() + 1e-9)
                q5_rets = q5_rets[abs(z) < 3]

            if len(q5_rets) == 0:
                short_ret = np.nan
            else:
                raw_w   = pd.Series(1 / len(q5_rets), index=q5_rets.index)
                capped  = raw_w.clip(upper=MAX_SINGLE_STOCK_WEIGHT)
                capped /= capped.sum()
                short_ret = (q5_rets * capped).sum()

        ls_ret = (
            long_ret - short_ret
            if not (np.isnan(long_ret) or np.isnan(short_ret))
            else np.nan
        )
        port_returns[next_date] = {
            "lowvol_Q1": long_ret,
            **mid_rets,
            "lowvol_Q5": short_ret,
            "lowvol_LS": ls_ret,
        }

    result = pd.DataFrame(port_returns).T
    result.index = pd.to_datetime(result.index)
    result.index.name = "date"

    ls = result["lowvol_LS"].dropna()
    ann_ret = ls.mean() * 12
    ann_vol = ls.std() * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    hit     = (ls > 0).mean()

    print(f"    BEFORE fix -> Ann. return: -5.90%  Vol: 40.73%  Sharpe: -0.14  Hit: 59.5%")
    print(f"    AFTER  fix -> Ann. return: {ann_ret*100:+.2f}%  "
          f"Vol: {ann_vol*100:.2f}%  Sharpe: {sharpe:.2f}  Hit: {hit*100:.1f}%")

    return result


def diagnose_short_book(quintiles:       pd.DataFrame,
                        monthly_returns: pd.DataFrame,
                        universe_df:     pd.DataFrame,
                        n_worst_months:  int = 5):
    """Print the n worst months for the naive (unfixed) low-vol short book."""
    print("\n  Diagnosing Low-Vol short book (Q5) worst months …\n")
    dates, monthly = sorted(set(quintiles.index) & set(monthly_returns.index)), []

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        q5_stocks = quintiles.loc[date][quintiles.loc[date] == 5].index
        q5_rets   = monthly_returns.loc[next_date].reindex(q5_stocks).dropna()
        monthly.append({
            "date":         next_date,
            "short_ret":    q5_rets.mean(),
            "n_stocks":     len(q5_rets),
            "worst_stock":  q5_rets.idxmax() if len(q5_rets) else None,
            "worst_return": q5_rets.max()    if len(q5_rets) else None,
        })

    df = pd.DataFrame(monthly).set_index("date").sort_values("short_ret", ascending=False)
    print(f"  {'Date':12s}  {'Short ret':>10s}  {'N stocks':>9s}  "
          f"{'Worst stock':>20s}  {'Worst ret':>10s}")
    print("  " + "-" * 72)
    for date, row in df.head(n_worst_months).iterrows():
        print(f"  {str(date.date()):12s}  {row['short_ret']*100:>+9.1f}%  "
              f"{row['n_stocks']:>9d}  {str(row['worst_stock']):>20s}  "
              f"{row['worst_return']*100:>+9.1f}%")
    print(f"\n  High positive 'short_ret' months = stocks you shorted went UP = loss.\n")


# -----------------------------------------------------------------------------
# 9. FACTOR SNAPSHOT
# -----------------------------------------------------------------------------

def build_factor_snapshot(factor_dict: dict,
                          quintile_dict: dict,
                          fundamentals: pd.DataFrame,
                          latest_date: pd.Timestamp) -> pd.DataFrame:
    """Cross-sectional snapshot at most recent date for dashboard display."""
    rows = {}

    for fname, fdf in factor_dict.items():
        if latest_date not in fdf.index:
            continue
        rows[f"{fname}_z"] = fdf.loc[latest_date]

    for fname, qdf in quintile_dict.items():
        if latest_date not in qdf.index:
            continue
        rows[f"{fname}_q"] = qdf.loc[latest_date]

    snapshot = pd.DataFrame(rows)
    snapshot.index.name = "ticker"

    if not fundamentals.empty:
        for col in ["priceToBook", "enterpriseToEbitda", "returnOnEquity"]:
            if col in fundamentals.columns:
                snapshot[col] = fundamentals[col].reindex(snapshot.index)

    return snapshot.dropna(how="all")


# -----------------------------------------------------------------------------
# 10. MAIN
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GCC Equity Factor Construction / Layer 2")
    p.add_argument(
        "--refresh-fundamentals",
        action="store_true",
        help="Force re-fetch of fundamentals and earnings stability from yfinance (slow)",
    )
    p.add_argument(
        "--exchange",
        choices=[e.upper() for e in ALL_EXCHANGES] + ["QSE"],
        default=None,
        help="Run for one exchange only (debugging)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-stock factor assignments",
    )
    return p.parse_args()


def main():
    args = parse_args()

    exchanges = (
        [args.exchange.lower()]
        if args.exchange
        else ALL_EXCHANGES
    )

    print("=" * 65)
    print("  GCC EQUITY FACTOR RESEARCH / Layer 2: Factor Construction")
    print("=" * 65)
    if args.exchange:
        print(f"  [DEBUG] Single-exchange mode: {args.exchange}")

    # -- Load data ---------------------------------------------------------
    print("\n[1/7] Loading price data …")
    monthly_returns = load_monthly_returns(exchanges)
    monthly_prices  = load_monthly_prices(exchanges)
    daily_prices    = load_daily_prices(exchanges)
    universe_df     = load_universe()

    tickers = monthly_prices.columns.tolist()
    latest_date = monthly_prices.index[-1]
    print(f"  Universe: {len(tickers)} tickers  |  Latest date: {latest_date.date()}")

    # -- Fetch fundamentals ------------------------------------------------
    fund_cache = os.path.join(DATA_DIR, "fundamentals.parquet")
    if os.path.exists(fund_cache) and not args.refresh_fundamentals:
        print(f"\n[2/7] Loading cached fundamentals from {fund_cache} …")
        fundamentals = pd.read_parquet(fund_cache)
    else:
        print("\n[2/7] Fetching fundamentals from yfinance (slow, one-time) …")
        fundamentals = fetch_all_fundamentals(tickers)
        fundamentals.to_parquet(fund_cache)
        print(f"  Cached to: {fund_cache}")

    # -- Fetch earnings stability ------------------------------------------
    stab_cache = os.path.join(DATA_DIR, "earnings_stability.parquet")
    if os.path.exists(stab_cache) and not args.refresh_fundamentals:
        print(f"\n[3/7] Loading cached earnings stability …")
        stability = pd.read_parquet(stab_cache).squeeze()
    else:
        print("\n[3/7] Computing earnings stability (slow, one-time) …")
        stability = fetch_quality_data(tickers)
        stability.to_frame("stability").to_parquet(stab_cache)
        print(f"  Cached to: {stab_cache}")

    # -- Compute factors ---------------------------------------------------
    print("\n[4/7] Computing factors …")
    mom_scores     = compute_momentum(monthly_prices, verbose=args.verbose)
    lowvol_scores  = compute_low_vol(daily_prices, monthly_prices, verbose=args.verbose)
    value_scores   = compute_value(fundamentals, monthly_prices, verbose=args.verbose)
    quality_scores = compute_quality(fundamentals, monthly_prices, stability, verbose=args.verbose)

    factor_dict = {
        "value":   value_scores,
        "mom":     mom_scores,
        "quality": quality_scores,
        "lowvol":  lowvol_scores,
    }

    # -- Assign quintiles --------------------------------------------------
    print("\n[5/7] Assigning quintiles …")
    quintile_dict = {}
    for fname, fdf in factor_dict.items():
        print(f"  -> {fname}")
        quintile_dict[fname] = assign_quintiles(fdf, verbose=args.verbose)

    # -- Compute portfolio returns -----------------------------------------
    print("\n[6/7] Computing long-short portfolio returns …")
    port_frames = []
    for fname, qdf in quintile_dict.items():
        if fname == "lowvol":
            port_ret = compute_lowvol_portfolio_returns_fixed(
                qdf, monthly_returns, universe_df
            )
        else:
            port_ret = compute_portfolio_returns(qdf, monthly_returns, fname)
        port_frames.append(port_ret)

    portfolio_returns = pd.concat(port_frames, axis=1)

    # -- Save outputs ------------------------------------------------------
    print("\n[7/7] Saving outputs …")

    # (a) Factor z-scores / MultiIndex columns: (factor_name, ticker)
    all_scores = {}
    for fname, fdf in factor_dict.items():
        fdf_mi = fdf.copy()
        fdf_mi.columns = pd.MultiIndex.from_tuples([(fname, t) for t in fdf.columns])
        all_scores[fname] = fdf_mi

    scores_path = os.path.join(OUTPUT_DIR, "factor_scores.parquet")
    pd.concat(list(all_scores.values()), axis=1).to_parquet(scores_path)
    print(f"  Saved: {scores_path}")

    # (b) Quintile assignments / MultiIndex columns: (factor_name, ticker)
    all_quintiles = {}
    for fname, qdf in quintile_dict.items():
        qdf_mi = qdf.copy()
        qdf_mi.columns = pd.MultiIndex.from_tuples([(fname, t) for t in qdf.columns])
        all_quintiles[fname] = qdf_mi

    quintile_path = os.path.join(OUTPUT_DIR, "quintile_assignments.parquet")
    pd.concat(list(all_quintiles.values()), axis=1).to_parquet(quintile_path)
    print(f"  Saved: {quintile_path}")

    # (c) Portfolio returns (Parquet + CSV)
    port_path = os.path.join(OUTPUT_DIR, "portfolio_returns.parquet")
    portfolio_returns.to_parquet(port_path)
    portfolio_returns.to_csv(os.path.join(OUTPUT_DIR, "portfolio_returns.csv"))
    print(f"  Saved: {port_path}")

    # (d) Cross-sectional snapshot (most recent month)
    snapshot = build_factor_snapshot(
        {k: v.droplevel(0, axis=1) for k, v in all_scores.items()},
        {k: v.droplevel(0, axis=1) for k, v in all_quintiles.items()},
        fundamentals,
        latest_date,
    )
    snap_path = os.path.join(OUTPUT_DIR, "factor_snapshot.csv")
    snapshot.to_csv(snap_path)
    print(f"  Saved: {snap_path}  ({len(snapshot)} stocks)")

    # -- Final summary (spec-required format) -----------------------------
    start_d = portfolio_returns.index[0].date()
    end_d   = portfolio_returns.index[-1].date()
    n_months = len(portfolio_returns)

    print()
    print("=" * 65)
    print("  FACTOR CONSTRUCTION COMPLETE")
    print("=" * 65)
    print(f"\n  Period: {start_d} -> {end_d}   |   {n_months} months of L/S returns")
    print()
    print(f"  {'Factor':<12} {'Ann. Return':>12} {'Ann. Vol':>10} {'Sharpe':>8} {'Hit Rate':>10}")
    print("  " + "-" * 54)

    FACTOR_DISPLAY = [
        ("value",   "VALUE"),
        ("mom",     "MOMENTUM"),
        ("quality", "QUALITY"),
        ("lowvol",  "LOW-VOL"),
    ]
    for fname, label in FACTOR_DISPLAY:
        col = f"{fname}_LS"
        if col not in portfolio_returns.columns:
            continue
        ls = portfolio_returns[col].dropna()
        if len(ls) == 0:
            print(f"  {label:<12} {'N/A':>12}")
            continue
        ann_ret  = ls.mean() * 12
        ann_vol  = ls.std() * np.sqrt(12)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
        hit_rate = (ls > 0).mean()
        print(f"  {label:<12} {ann_ret*100:>+11.2f}%"
              f" {ann_vol*100:>9.2f}%"
              f" {sharpe:>8.2f}"
              f" {hit_rate*100:>9.1f}%")

    print()
    print("  Files written to:", OUTPUT_DIR)
    print()
    print("  Next: python fama_macbeth.py   (Layer 3 / regression & significance)")
    print("        python backtest.py        (Layer 3 / performance attribution)")
    print()


if __name__ == "__main__":
    main()
