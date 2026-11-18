"""Data loading for the GCC Equity Factor Research dashboard.

Reads real output files from /data and /data/factors.
Falls back to synthetic data only when a file is missing.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
FACTORS_DIR = DATA_DIR / "factors"

FACTORS = ["MOMENTUM", "QUALITY", "VALUE", "LOW-VOL"]
EXCHANGES = ["GCC", "ADX", "DFM", "Tadawul", "QSE"]

_cache: dict = {}

TICKER_NAMES_CACHE = DATA_DIR / "ticker_names.csv"


# ---------------------------------------------------------------------------
# Ticker name helpers
# ---------------------------------------------------------------------------

def load_ticker_names() -> dict:
    """Return {ticker: short_name} from cached CSV. Empty dict if not built yet."""
    if "ticker_names" in _cache:
        return _cache["ticker_names"]
    if not TICKER_NAMES_CACHE.exists():
        _cache["ticker_names"] = {}
        return {}
    try:
        df = pd.read_csv(TICKER_NAMES_CACHE, index_col=0)
        result = df["name"].dropna().to_dict()
        _cache["ticker_names"] = result
        return result
    except Exception:
        _cache["ticker_names"] = {}
        return {}


def build_ticker_name_cache(verbose: bool = True) -> dict:
    """Fetch shortName from yfinance for all tickers (universe + snapshot) and save to CSV.

    One-time operation (~3-5 min). Subsequent calls read from cache instantly.
    Returns {ticker: name}.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed -- pip install yfinance")
        return {}

    ticker_set: set = set()
    uni_path = DATA_DIR / "universe.csv"
    if uni_path.exists():
        ticker_set.update(pd.read_csv(uni_path)["ticker"].dropna().unique().tolist())
    snap_path = DATA_DIR / "factor_snapshot.csv"
    if snap_path.exists():
        ticker_set.update(pd.read_csv(snap_path, index_col=0).index.tolist())

    if not ticker_set:
        print("No universe or snapshot file found.")
        return {}

    tickers = sorted(ticker_set)
    if verbose:
        print(f"Fetching names for {len(tickers)} tickers ...")

    existing: dict = {}
    if TICKER_NAMES_CACHE.exists():
        try:
            existing = pd.read_csv(TICKER_NAMES_CACHE, index_col=0)["name"].dropna().to_dict()
        except Exception:
            pass

    names: dict = dict(existing)
    # Re-fetch tickers whose name still equals the ticker code (failed last time)
    missing = [
        t for t in tickers
        if t not in names or names[t] == t.split(".")[0]
    ]

    if verbose and missing:
        print(f"  {len(missing)} tickers to fetch ...")

    for i, tk in enumerate(missing):
        try:
            info = yf.Ticker(tk).info
            name = info.get("shortName") or info.get("longName") or ""
            names[tk] = name if name else tk.split(".")[0]
        except Exception:
            names[tk] = tk.split(".")[0]
        if verbose and (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(missing)} done ...")

    pd.DataFrame({"name": names}).to_csv(TICKER_NAMES_CACHE)
    _cache["ticker_names"] = names
    if verbose:
        print(f"Saved {len(names)} ticker names -> {TICKER_NAMES_CACHE}")
    return names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv(rel: str) -> pd.DataFrame | None:
    p = Path(rel) if os.path.isabs(rel) else FACTORS_DIR / rel
    if not p.exists():
        p = DATA_DIR / rel
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _parquet(rel: str) -> pd.DataFrame | None:
    p = DATA_DIR / rel
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


def _sig_stars(t: float) -> str:
    if pd.isna(t):
        return ""
    a = abs(t)
    if a >= 2.576:
        return "***"
    if a >= 1.960:
        return "**"
    if a >= 1.645:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Page 1 / Factor Analysis
# ---------------------------------------------------------------------------

def load_fm_summary() -> pd.DataFrame:
    if "fm_summary" in _cache:
        return _cache["fm_summary"]
    df = _csv("fm_summary.csv")
    if df is None:
        df = pd.DataFrame({
            "factor": FACTORS,
            "lambda_joint_bps": [13.9, 21.1, -9.3, -76.3],
            "t_joint_nw3": [0.65, 1.35, -0.56, -0.66],
            "lambda_uni_bps": [25.5, 42.6, -34.2, -70.2],
            "t_uni_nw3": [1.97, 2.86, -2.40, -0.68],
            "sig_uni": ["**", "***", "**", ""],
            "IC_mean": [0.056, 0.075, -0.025, 0.078],
            "IC_std": [0.188, 0.182, 0.145, 0.172],
            "t_IC": [3.32, 4.77, -2.01, 5.13],
            "ICIR_ann": [1.03, 1.42, -0.60, 1.58],
            "port_ann_ret_pct": [10.3, 13.4, -7.4, -27.8],
            "port_ann_vol_pct": [15.9, 14.3, 14.3, 99.7],
            "port_sharpe": [0.65, 0.94, -0.52, -0.28],
            "T_months": [124, 124, 124, 124],
        })
    df["factor"] = df["factor"].str.upper()
    df["sig_stars"] = df["t_uni_nw3"].apply(_sig_stars)
    _cache["fm_summary"] = df
    return df


def load_quintile_spreads() -> pd.DataFrame:
    if "quintile_spreads" in _cache:
        return _cache["quintile_spreads"]
    df = _csv("quintile_spreads.csv")
    if df is None:
        df = pd.DataFrame({
            "factor": FACTORS,
            "Q1": [11.6, 6.6, -3.5, 0.5],
            "Q2": [10.1, 6.4, 3.1, 4.4],
            "Q3": [4.0, 1.0, 2.3, 1.1],
            "Q4": [1.8, -1.3, 1.8, -0.8],
            "Q5": [1.3, -6.8, 4.0, 28.4],
            "Q1-Q5": [10.3, 13.4, -7.4, -27.8],
        })
    df["factor"] = df["factor"].str.upper()
    _cache["quintile_spreads"] = df
    return df


def load_ic_monthly() -> pd.DataFrame:
    if "ic_monthly" in _cache:
        return _cache["ic_monthly"]
    df = _csv("ic_monthly.csv")
    if df is None:
        np.random.seed(42)
        dates = pd.date_range("2015-03-31", "2025-12-31", freq="ME")
        df = pd.DataFrame({
            "date": dates.astype(str),
            "VALUE": np.random.normal(-0.025, 0.145, len(dates)),
            "MOMENTUM": np.random.normal(0.056, 0.188, len(dates)),
            "QUALITY": np.random.normal(0.075, 0.182, len(dates)),
            "LOW-VOL": np.random.normal(0.078, 0.172, len(dates)),
        })
    date_col = df.columns[0]
    df.index = pd.to_datetime(df[date_col])
    df = df.drop(columns=[date_col])
    _cache["ic_monthly"] = df
    return df


def load_backtest_summary() -> pd.DataFrame:
    if "backtest_summary" in _cache:
        return _cache["backtest_summary"]
    df = _csv("backtest_summary.csv")
    if df is None:
        rows = []
        for exch in EXCHANGES:
            for fac in FACTORS:
                rows.append({
                    "exchange": exch, "factor": fac,
                    "LS_ann_pct": np.random.normal(8, 5),
                    "LS_vol_pct": np.random.uniform(12, 25),
                    "LS_sharpe": np.random.normal(0.55, 0.3),
                    "LS_maxdd_pct": np.random.uniform(-60, -20),
                    "LS_hit_pct": np.random.uniform(48, 65),
                    "IC_mean": np.random.normal(0.05, 0.03),
                    "IC_t": np.random.normal(2.5, 1.2),
                    "T_months": 124,
                })
        df = pd.DataFrame(rows)
    df["factor"] = df["factor"].str.upper()
    _cache["backtest_summary"] = df
    return df


def compute_fm_interactions() -> pd.DataFrame:
    """Exchange × factor interaction terms: LS premium differential vs GCC baseline."""
    if "fm_interactions" in _cache:
        return _cache["fm_interactions"]

    bs = load_backtest_summary()
    records = []
    for fac in ["MOMENTUM", "QUALITY"]:
        fdf = bs[bs["factor"] == fac]
        gcc_row = fdf[fdf["exchange"].str.upper() == "GCC"]
        gcc_ls = gcc_row["LS_ann_pct"].values[0] if len(gcc_row) else fdf["LS_ann_pct"].mean()
        gcc_vol = gcc_row["LS_vol_pct"].values[0] if len(gcc_row) else fdf["LS_vol_pct"].mean()
        t_months = fdf["T_months"].median() if "T_months" in fdf.columns else 124

        for _, row in fdf.iterrows():
            exch = row["exchange"]
            coef = row["LS_ann_pct"] - gcc_ls
            exch_vol = row.get("LS_vol_pct", gcc_vol)
            se = exch_vol / np.sqrt(max(t_months, 1)) * np.sqrt(12) if exch_vol > 0 else 1.0
            tstat = coef / se if se > 0 else 0.0
            records.append({
                "Factor": fac.title(),
                "Exchange": exch,
                "Coef (ann. %)": round(coef, 2),
                "t-stat": round(tstat, 2),
                "Sig.": _sig_stars(tstat),
            })

    df = pd.DataFrame(records)
    _cache["fm_interactions"] = df
    return df


# ---------------------------------------------------------------------------
# Page 2 / Base Portfolio
# ---------------------------------------------------------------------------

def load_portfolio_returns() -> pd.DataFrame:
    """Returns monthly return series for the base CB portfolio."""
    if "portfolio_returns" in _cache:
        return _cache["portfolio_returns"]

    l3b = _parquet("layer3b_returns.parquet")
    if l3b is not None and "combined_cb" in l3b.columns:
        df = l3b[["combined_cb"]].copy()
        df.columns = ["base_return"]
    else:
        np.random.seed(123)
        dates = pd.date_range("2016-01-31", "2024-12-31", freq="ME")
        n = len(dates)
        df = pd.DataFrame({
            "base_return": np.random.normal(0.078 / 12, 0.108 / np.sqrt(12), n),
        }, index=dates)

    df = df.dropna(subset=["base_return"])
    df = df[df["base_return"] != 0]
    _cache["portfolio_returns"] = df
    return df


def compute_portfolio_stats(df: pd.DataFrame, col: str) -> dict:
    s = df[col].dropna()
    ann_ret = s.mean() * 12
    ann_vol = s.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + s).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    hit = (s > 0).mean()
    return {
        "Ann. Return": f"{ann_ret:+.2%}",
        "Ann. Vol": f"{ann_vol:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "Max DD": f"{max_dd:.2%}",
        "Calmar": f"{calmar:.2f}",
        "Hit Rate": f"{hit:.1%}",
    }


def compute_monthly_heatmap(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col].dropna()
    s.index = pd.to_datetime(s.index)
    monthly = s.copy()
    if monthly.index.freq not in (None,) and hasattr(monthly.index, 'to_period'):
        monthly = monthly.groupby([monthly.index.year, monthly.index.month]).last()
    pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    })
    return pivot.pivot(index="year", columns="month", values="ret") * 100


# ---------------------------------------------------------------------------
# Page 3 / Summary
# ---------------------------------------------------------------------------

def load_tearsheet() -> pd.DataFrame:
    if "tearsheet" in _cache:
        return _cache["tearsheet"]
    df_port = load_portfolio_returns()
    stats = compute_portfolio_stats(df_port, "base_return")
    df = pd.DataFrame({
        "Portfolio": ["Base CB Portfolio"],
        "Ann. Ret": [stats["Ann. Return"]],
        "Ann. Vol": [stats["Ann. Vol"]],
        "Sharpe": [float(stats["Sharpe"])],
        "Max DD": [stats["Max DD"]],
        "Calmar": [float(stats["Calmar"])],
    })
    _cache["tearsheet"] = df
    return df


# ---------------------------------------------------------------------------
# Factor returns per exchange, oil macro control, crowding
# ---------------------------------------------------------------------------

FACTOR_FILE_MAP = {
    "VALUE": "value",
    "MOMENTUM": "mom",
    "QUALITY": "quality",
    "LOW-VOL": "lowvol",
}

EXCHANGE_FILE_MAP = {
    "GCC": "gcc",
    "ADX": "adx",
    "DFM": "dfm",
    "Tadawul": "tadawul",
    "QSE": "qse",
}


def load_factor_returns_by_exchange() -> dict:
    """Load L/S monthly return series per exchange and factor from quintile CSVs.

    Returns {exchange: {factor: pd.Series of monthly LS returns}}.
    Missing exchange-factor combos are omitted (no synthetic fallback).
    """
    if "factor_returns_by_exchange" in _cache:
        return _cache["factor_returns_by_exchange"]

    result: dict = {}
    for exch, exch_file in EXCHANGE_FILE_MAP.items():
        result[exch] = {}
        for factor, factor_file in FACTOR_FILE_MAP.items():
            fname = f"quintile_returns_{exch_file}_{factor_file}.csv"
            df = _csv(fname)
            if df is not None and "LS" in df.columns:
                date_col = df.columns[0]
                s = df.set_index(pd.to_datetime(df[date_col]))["LS"]
                result[exch][factor] = s

    _cache["factor_returns_by_exchange"] = result
    return result


def load_oil_returns() -> tuple:
    """Fetch Brent crude monthly returns via yfinance (fallback: WTI CL=F).

    Returns (pd.Series | None, status_message_str).
    """
    if "oil_returns" in _cache:
        return _cache["oil_returns"]

    try:
        import yfinance as yf
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout

        def _fetch(ticker):
            return yf.download(
                ticker, start="2015-01-01", end="2025-12-31",
                interval="1mo", progress=False, auto_adjust=True,
            )

        for ticker in ["BZ=F", "CL=F"]:
            try:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(_fetch, ticker)
                    try:
                        raw = future.result(timeout=15)
                    except _Timeout:
                        continue
                if raw is None or len(raw) < 10:
                    continue
                close = raw["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                ret = close.pct_change().dropna()
                ret.index = ret.index.to_period("M").to_timestamp("M")
                result = (ret, f"{ticker}: {len(ret)} monthly obs loaded")
                _cache["oil_returns"] = result
                return result
            except Exception:
                continue
        result = (None, "Oil data unavailable - macro control disabled")
        _cache["oil_returns"] = result
        return result
    except Exception as exc:
        result = (None, f"Oil fetch error: {str(exc)[:80]}")
        _cache["oil_returns"] = result
        return result


def _nw_regression(y: pd.Series, x: pd.Series, nlags: int = 4) -> dict:
    """OLS y ~ const + x with Newey-West HAC standard errors.

    Returns dict: alpha (annualized monthly intercept), beta, t_alpha, t_beta, r2, n.
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    combined = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(combined) < 10:
        return {"alpha": np.nan, "beta": np.nan,
                "t_alpha": np.nan, "t_beta": np.nan,
                "r2": np.nan, "n": 0}

    X = add_constant(combined["x"])
    model = OLS(combined["y"], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": nlags}
    )
    return {
        "alpha": float(model.params["const"]) * 12,
        "beta": float(model.params["x"]),
        "t_alpha": float(model.tvalues["const"]),
        "t_beta": float(model.tvalues["x"]),
        "r2": float(model.rsquared),
        "n": int(model.nobs),
    }


def compute_oil_regression() -> tuple:
    """Regress each GCC factor's LS returns on Brent crude monthly returns (NW-4).

    Returns (DataFrame[Factor, Oil Beta, t-stat, R²], status_msg).
    """
    if "oil_regression" in _cache:
        return _cache["oil_regression"]

    oil, status = load_oil_returns()
    gcc_rets = load_factor_returns_by_exchange().get("GCC", {})

    rows = []
    for factor in ["MOMENTUM", "QUALITY", "VALUE", "LOW-VOL"]:
        f_series = gcc_rets.get(factor)
        if f_series is None or oil is None:
            rows.append({"Factor": factor.title(),
                         "Oil Beta": np.nan, "t-stat": np.nan, "R²": np.nan})
            continue
        st = _nw_regression(f_series, oil, nlags=4)
        rows.append({
            "Factor": factor.title(),
            "Oil Beta": round(st["beta"], 3),
            "t-stat": round(st["t_beta"], 2),
            "R²": round(st["r2"], 3),
        })

    df = pd.DataFrame(rows)
    result = (df, status)
    _cache["oil_regression"] = result
    return result


def compute_oil_adjusted_quintile_spreads() -> tuple:
    """Residualize each GCC quintile return series against oil (NW-4 alpha as adjusted return).

    Returns (DataFrame same format as load_quintile_spreads, status_msg).
    """
    if "oil_adj_quintile" in _cache:
        return _cache["oil_adj_quintile"]

    oil, status = load_oil_returns()
    if oil is None:
        return (load_quintile_spreads(), f"Unadjusted - {status}")

    rows = []
    for factor, factor_file in FACTOR_FILE_MAP.items():
        raw = _csv(f"quintile_returns_gcc_{factor_file}.csv")
        if raw is None:
            fb = load_quintile_spreads()
            fb_row = fb[fb["factor"] == factor]
            rows.append(fb_row.iloc[0].to_dict() if len(fb_row) else {"factor": factor})
            continue

        date_col = raw.columns[0]
        raw = raw.set_index(pd.to_datetime(raw[date_col])).drop(columns=[date_col])

        q_vals: dict = {}
        for qcol in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            if qcol not in raw.columns:
                q_vals[qcol] = np.nan
                continue
            st = _nw_regression(raw[qcol], oil, nlags=4)
            # alpha = monthly intercept * 12 * 100 → annualized %
            q_vals[qcol] = (
                round(st["alpha"] * 100, 2) if not pd.isna(st["alpha"]) else np.nan
            )

        q1, q5 = q_vals.get("Q1", np.nan), q_vals.get("Q5", np.nan)
        ls_val = round(q1 - q5, 2) if not any(pd.isna(v) for v in [q1, q5]) else np.nan
        rows.append({
            "factor": factor,
            "Q1": q1, "Q2": q_vals.get("Q2"), "Q3": q_vals.get("Q3"),
            "Q4": q_vals.get("Q4"), "Q5": q5, "Q1-Q5": ls_val,
        })

    result = (pd.DataFrame(rows), f"Oil-adjusted ({status})")
    _cache["oil_adj_quintile"] = result
    return result


def compute_intra_exchange_corr() -> dict:
    """Within-exchange factor-factor Pearson correlation matrices.

    Returns {exchange: pd.DataFrame NxN | None if <2 factors available}.
    """
    if "intra_exchange_corr" in _cache:
        return _cache["intra_exchange_corr"]

    factor_rets = load_factor_returns_by_exchange()
    result: dict = {}
    for exch in ["ADX", "DFM", "Tadawul", "QSE", "Kuwait"]:
        if exch == "Kuwait" or exch not in factor_rets:
            result[exch] = None
            continue
        exch_data = factor_rets[exch]
        if len(exch_data) < 2:
            result[exch] = None
            continue
        df = pd.DataFrame({f.title(): s for f, s in exch_data.items()}).dropna(how="all")
        result[exch] = df.corr()

    _cache["intra_exchange_corr"] = result
    return result


def compute_cross_exchange_corr() -> pd.DataFrame:
    """Pairwise cross-exchange correlations for each factor's L/S return series.

    Returns DataFrame: rows = exchange pairs, columns = Exchange Pair + factor names.
    Cells > 0.6 are crowding signals.
    """
    if "cross_exchange_corr" in _cache:
        return _cache["cross_exchange_corr"]

    factor_rets = load_factor_returns_by_exchange()
    exchanges = ["ADX", "DFM", "Tadawul", "QSE"]
    factors = ["MOMENTUM", "QUALITY", "VALUE", "LOW-VOL"]

    rows = []
    for i, e1 in enumerate(exchanges):
        for e2 in exchanges[i + 1:]:
            row: dict = {"Exchange Pair": f"{e1} / {e2}"}
            for factor in factors:
                s1 = factor_rets.get(e1, {}).get(factor)
                s2 = factor_rets.get(e2, {}).get(factor)
                if s1 is not None and s2 is not None:
                    aligned = pd.DataFrame({"s1": s1, "s2": s2}).dropna()
                    if len(aligned) >= 10:
                        row[factor.title()] = round(float(aligned.corr().iloc[0, 1]), 3)
                    else:
                        row[factor.title()] = np.nan
                else:
                    row[factor.title()] = np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    _cache["cross_exchange_corr"] = df
    return df


# ---------------------------------------------------------------------------
# Layer 3 / Backtesting
# ---------------------------------------------------------------------------

def load_sub_period_analysis() -> pd.DataFrame:
    if "sub_period" in _cache:
        return _cache["sub_period"]
    df = _csv("sub_period_analysis.csv")
    if df is None:
        df = pd.DataFrame(columns=["period", "factor", "lambda_bps",
                                    "t_stat", "IC_mean", "IC_std", "sharpe", "T"])
    df["factor"] = df["factor"].str.upper()
    _cache["sub_period"] = df
    return df


def load_quintile_stock_membership() -> dict:
    """Return latest-date stock membership per factor × quintile from snapshot or parquet.

    Returns:
        {factor_display: {"Q1": [tickers], ..., "Q5": [tickers]}}
        factor_display ∈ ["Momentum", "Quality", "Value", "Low-Vol"]
    """
    if "quintile_membership" in _cache:
        return _cache["quintile_membership"]

    FACTOR_COL_MAP = {
        "Momentum": "mom_q",
        "Quality":  "quality_q",
        "Value":    "value_q",
        "Low-Vol":  "lowvol_q",
    }

    snap = None
    snap_path = DATA_DIR / "factor_snapshot.csv"
    if snap_path.exists():
        try:
            snap = pd.read_csv(snap_path, index_col=0)
        except Exception:
            snap = None

    if snap is None:
        qdf = _parquet("quintile_assignments.parquet")
        if qdf is not None:
            latest = qdf.index.max()
            rows = []
            for fname, display in [("mom", "Momentum"), ("quality", "Quality"),
                                    ("value", "Value"), ("lowvol", "Low-Vol")]:
                level0 = [c[0] for c in qdf.columns] if isinstance(qdf.columns, pd.MultiIndex) else []
                if fname in level0:
                    sub = qdf[fname].loc[latest].dropna()
                    rows_snap = {t: {f"mom_q": np.nan, f"quality_q": np.nan,
                                     f"value_q": np.nan, f"lowvol_q": np.nan} for t in sub.index}
                    for t, v in sub.items():
                        col = {"mom": "mom_q", "quality": "quality_q",
                               "value": "value_q", "lowvol": "lowvol_q"}[fname]
                        if t not in rows_snap:
                            rows_snap[t] = {}
                        rows_snap[t][col] = v
            snap = pd.DataFrame(rows_snap).T if rows else pd.DataFrame()

    if snap is None or snap.empty:
        _cache["quintile_membership"] = {}
        return {}

    result: dict = {}
    for display, col in FACTOR_COL_MAP.items():
        if col not in snap.columns:
            continue
        sub = snap[col].dropna()
        quintile_map: dict = {}
        for q in range(1, 6):
            members = sub[sub == q].index.tolist()
            quintile_map[f"Q{q}"] = sorted(members)
        result[display] = quintile_map

    _cache["quintile_membership"] = result
    return result


def load_quintile_timeseries(exchange: str, factor: str) -> pd.DataFrame | None:
    """Load quintile return time-series for one exchange × factor pair."""
    ex_slug  = EXCHANGE_FILE_MAP.get(exchange, exchange.lower())
    fac_slug = FACTOR_FILE_MAP.get(factor.upper(), factor.lower())
    key = f"qt_{ex_slug}_{fac_slug}"
    if key in _cache:
        return _cache[key]
    p = FACTORS_DIR / f"quintile_returns_{ex_slug}_{fac_slug}.csv"
    if not p.exists():
        _cache[key] = None
        return None
    try:
        df = pd.read_csv(p, parse_dates=["date"], index_col="date")
        _cache[key] = df
        return df
    except Exception:
        _cache[key] = None
        return None


def compute_thesis_scorecard() -> tuple:
    """Compute Thesis Scorecard Panel A (factor significance) and Panel B (heterogeneity).

    Returns (panel_a_df, panel_b_df).
    """
    if "thesis_scorecard" in _cache:
        return _cache["thesis_scorecard"]

    fm = load_fm_summary()
    bs = load_backtest_summary()
    idf = compute_fm_interactions()
    factor_order = ["MOMENTUM", "QUALITY", "VALUE", "LOW-VOL"]

    # Panel A / Claim 1: Factor Premia Significance
    panel_a_rows = []
    for factor in factor_order:
        row = fm[fm["factor"] == factor]
        r = row.iloc[0] if len(row) else {}
        t = float(r.get("t_uni_nw3", np.nan)) if len(row) else np.nan
        lam = float(r.get("lambda_uni_bps", np.nan)) if len(row) else np.nan
        sig = (not pd.isna(t)) and abs(t) >= 2
        direction = "Positive" if (not pd.isna(lam)) and lam > 0 else "Negative"
        fbs = bs[(bs["factor"] == factor) & (bs["exchange"] != "GCC")]
        best_exch = (
            fbs.loc[fbs["LS_sharpe"].idxmax(), "exchange"]
            if len(fbs) else "N/A"
        )
        panel_a_rows.append({
            "Factor": factor.title(),
            "Significant (|t|>2)": "Yes" if sig else "No",
            "t-stat (NW-4)": f"{t:+.2f}" if not pd.isna(t) else "N/A",
            "Direction": direction,
            "Strongest Exchange": best_exch,
        })
    panel_a = pd.DataFrame(panel_a_rows)

    # Panel B / Claim 2: Cross-Exchange Heterogeneity
    panel_b_rows = []
    for factor in factor_order:
        fac_title = factor.title()
        fac_idf = idf[idf["Factor"] == fac_title]
        if len(fac_idf):
            idx = fac_idf["t-stat"].abs().idxmax()
            max_t = float(fac_idf.loc[idx, "t-stat"])
            outlier_exch = fac_idf.loc[idx, "Exchange"]
        else:
            max_t, outlier_exch = np.nan, "N/A"
        het_sig = (not pd.isna(max_t)) and abs(max_t) >= 2
        panel_b_rows.append({
            "Factor": fac_title,
            "Max Interaction |t|": f"{max_t:+.2f}" if not pd.isna(max_t) else "N/A",
            "Cross-Exch. Significant (|t|>2)": "Yes" if het_sig else "No",
            "Outlier Exchange": outlier_exch,
        })
    panel_b = pd.DataFrame(panel_b_rows)

    result = (panel_a, panel_b)
    _cache["thesis_scorecard"] = result
    return result
