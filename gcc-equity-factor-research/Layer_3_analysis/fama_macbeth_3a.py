"""
GCC Equity Factor Research Engine
Layer 3a: Fama-MacBeth Regressions & Statistical Significance
==============================================================

For each month t, fits a cross-sectional OLS:

    r_{i,t+1} = alpha_t + lambda_{value,t}*z_value + lambda_{mom,t}*z_mom
              + lambda_{quality,t}*z_quality + lambda_{lowvol,t}*z_lowvol + eps_{i,t}

Then takes the time-series mean of each lambda_t and computes Newey-West
t-statistics (4 lags, Bartlett kernel) to assess statistical significance.

Also computes:
  - Univariate FM (each factor standalone)
  - Spearman IC + annualised ICIR
  - Quintile monotonicity (Q1 -> Q5 spread)
  - Sub-period analysis (pre-COVID / COVID / post-COVID / recent)

Reads from:   data/factor_scores.parquet
              data/portfolio_returns.parquet
              data/returns_{exchange}.parquet

Writes to:    data/factors/fm_coefficients.csv
              data/factors/fm_summary.csv
              data/factors/ic_monthly.csv
              data/factors/quintile_spreads.csv
              data/factors/sub_period_analysis.csv

Run:
  python fama_macbeth.py
  python fama_macbeth.py --univariate    (print univariate tables too)
  python fama_macbeth.py --verbose
"""

import argparse
import os
import warnings
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.dirname(_HERE)
DATA_DIR   = os.path.join(_ROOT, "data")
OUTPUT_DIR = os.path.join(_ROOT, "data", "factors")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FACTORS = ["value", "mom", "quality", "lowvol"]
FACTOR_LABELS = {
    "value":   "VALUE",
    "mom":     "MOMENTUM",
    "quality": "QUALITY",
    "lowvol":  "LOW-VOL",
}

ALL_EXCHANGES = ["adx", "dfm", "tadawul", "qatar", "kuwait"]
KUWAIT_START  = pd.Timestamp("2016-01-01")

NW_LAGS       = 4    # Newey-West lags (Fama-MacBeth standard for monthly)
MIN_STOCKS_FM = 20   # minimum cross-sectional observations per month

SUB_PERIODS = {
    "Full sample":      (None,         None        ),
    "Pre-COVID 15-19":  ("2015-01-01", "2019-12-31"),
    "COVID 2020":       ("2020-01-01", "2020-12-31"),
    "Post-COVID 21-23": ("2021-01-01", "2023-12-31"),
    "Recent 24-26":     ("2024-01-01", "2026-12-31"),
}


# -----------------------------------------------------------------------------
# 1. DATA LOADING
# -----------------------------------------------------------------------------

def load_factor_scores() -> dict:
    """
    Load Layer 2 factor z-scores.
    Returns {factor_name: DataFrame(date x ticker)}.
    """
    path = os.path.join(DATA_DIR, "factor_scores.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run factor_construction.py first."
        )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)

    result = {}
    top_level = df.columns.get_level_values(0).unique()
    for fname in FACTORS:
        if fname in top_level:
            result[fname] = df[fname].copy()
    return result


def load_monthly_returns() -> pd.DataFrame:
    frames = []
    for ex in ALL_EXCHANGES:
        path = os.path.join(DATA_DIR, f"returns_{ex}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        if ex == "kuwait":
            df = df.loc[df.index >= KUWAIT_START]
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No return parquet files found.")
    combined = pd.concat(frames, axis=1)
    return combined.loc[:, ~combined.columns.duplicated()]


def load_portfolio_returns() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "portfolio_returns.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}.")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


# -----------------------------------------------------------------------------
# 2. NEWEY-WEST t-STATISTIC
# -----------------------------------------------------------------------------

def nw_tstat(series: pd.Series, n_lags: int = NW_LAGS):
    """
    Newey-West (1987) t-statistic for H0: mean = 0.
    Uses Bartlett kernel with n_lags.
    Returns (mean, t_stat). Both NaN if series too short.
    """
    x = np.asarray(series.dropna(), dtype=float)
    T = len(x)
    if T < max(10, n_lags + 2):
        return np.nan, np.nan

    mu       = x.mean()
    demeaned = x - mu

    # HAC variance: Gamma_0 + 2 * sum_{l=1}^{L} (1 - l/(L+1)) * Gamma_l
    var = np.dot(demeaned, demeaned) / T
    for lag in range(1, n_lags + 1):
        w     = 1.0 - lag / (n_lags + 1)
        gamma = np.dot(demeaned[lag:], demeaned[:-lag]) / T
        var  += 2.0 * w * gamma

    var = max(var, 1e-14)
    se  = np.sqrt(var / T)
    return mu, mu / se


# -----------------------------------------------------------------------------
# 3. CROSS-SECTIONAL REGRESSION (single month)
# -----------------------------------------------------------------------------

def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Least-squares via normal equations. Returns coefficient vector."""
    try:
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return coefs
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan)


def cross_sectional_ols(scores_t: pd.DataFrame,
                        ret_next: pd.Series,
                        factors_to_use: list) -> dict:
    """
    Single-month OLS: r_{i,t+1} = alpha + Sum lambda_k * z_{k,i,t}
    Returns {factor: coef, 'alpha': intercept, 'n': n_obs, 'r2': R2}.
    """
    X = scores_t[factors_to_use].copy()
    y = ret_next.reindex(X.index)

    valid = X.notna().all(axis=1) & y.notna()
    X, y  = X[valid], y[valid]
    n     = len(y)

    if n < MIN_STOCKS_FM:
        return {"n": n, "r2": np.nan}

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    X_mat = np.column_stack([np.ones(n), X_arr])
    coefs = _ols(y_arr, X_mat)

    y_hat  = X_mat @ coefs
    ss_res = np.sum((y_arr - y_hat) ** 2)
    ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    result = {"alpha": coefs[0], "n": n, "r2": r2}
    for i, fname in enumerate(factors_to_use):
        result[fname] = coefs[i + 1]
    return result


# -----------------------------------------------------------------------------
# 4. FAMA-MACBETH LOOP
# -----------------------------------------------------------------------------

def run_fama_macbeth(factor_scores: dict,
                     monthly_returns: pd.DataFrame,
                     factors_to_use: list,
                     label: str = "Joint") -> pd.DataFrame:
    """
    Full-sample Fama-MacBeth loop.
    Returns DataFrame of monthly lambda_t  (T x factors + alpha + n + r2).
    """
    active = [f for f in factors_to_use if f in factor_scores]
    if not active:
        return pd.DataFrame()

    print(f"  [{label}]  factors: {', '.join(active)} ...")

    # Collect all dates that appear in at least one factor score
    all_score_dates: set = set()
    for f in active:
        all_score_dates |= set(factor_scores[f].index)
    dates = sorted(all_score_dates & set(monthly_returns.index))

    monthly_betas = []

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        if next_date not in monthly_returns.index:
            continue

        ret_next = monthly_returns.loc[next_date]

        # Build (ticker x factor) cross-section for this month
        factor_cols = {}
        for fname in active:
            fdf = factor_scores[fname]
            if date in fdf.index:
                factor_cols[fname] = fdf.loc[date]
        if not factor_cols:
            continue

        scores_t = pd.DataFrame(factor_cols)
        row = cross_sectional_ols(scores_t, ret_next, active)
        row["date"] = next_date
        monthly_betas.append(row)

    if not monthly_betas:
        return pd.DataFrame()

    result = pd.DataFrame(monthly_betas).set_index("date")
    T_valid  = result[active[0]].notna().sum()
    avg_n    = result["n"].mean()
    avg_r2   = result["r2"].dropna().mean()
    print(f"    T = {T_valid} months  |  avg N = {avg_n:.0f} stocks/month  "
          f"|  avg R2 = {avg_r2*100:.1f}%")
    return result


# -----------------------------------------------------------------------------
# 5. INFORMATION COEFFICIENT
# -----------------------------------------------------------------------------

def compute_ic(factor_scores: dict,
               monthly_returns: pd.DataFrame) -> dict:
    """
    Monthly Spearman IC for each factor.
    IC_t = Spearman(z_{i,t}, r_{i,t+1})  cross-sectionally.

    Returns {factor_name: pd.Series(date -> IC)}.
    """
    dates = sorted(monthly_returns.index)
    ic_raw = {fname: {} for fname in factor_scores}

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        if next_date not in monthly_returns.index:
            continue
        ret_next = monthly_returns.loc[next_date]

        for fname, fdf in factor_scores.items():
            if date not in fdf.index:
                continue
            scores = fdf.loc[date].dropna()
            y      = ret_next.reindex(scores.index).dropna()
            x      = scores.reindex(y.index)
            if len(y) < 10:
                continue
            rho, _ = stats.spearmanr(x, y)
            ic_raw[fname][next_date] = rho

    result = {}
    for fname, d in ic_raw.items():
        s = pd.Series(d)
        s.index = pd.to_datetime(s.index)
        result[fname] = s
    return result


# -----------------------------------------------------------------------------
# 6. QUINTILE SPREAD
# -----------------------------------------------------------------------------

def quintile_spread(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised mean return (%) for each quintile. Checks Q1>Q2>Q3>Q4>Q5."""
    rows = []
    for fname in FACTORS:
        q_cols = {q: f"{fname}_Q{q}" for q in range(1, 6)}
        available = {q: c for q, c in q_cols.items()
                     if c in portfolio_returns.columns}
        if len(available) < 2:
            continue
        row: dict[str, Any] = {"factor": FACTOR_LABELS.get(fname, fname)}
        for q, col in available.items():
            row[f"Q{q}"] = float(portfolio_returns[col].dropna().mean()) * 12 * 100

        ls_col = f"{fname}_LS"
        if ls_col in portfolio_returns.columns:
            row["Q1-Q5"] = float(portfolio_returns[ls_col].dropna().mean()) * 12 * 100

        # Monotonicity check: fraction of adjacent pairs going in right direction
        vals: list[float] = [float(row[f"Q{q}"]) for q in range(1, 6)
                             if f"Q{q}" in row]
        if len(vals) >= 2:
            mono = sum(1 for i in range(len(vals)-1) if vals[i] > vals[i+1])
            row["monotone"] = f"{mono}/{len(vals)-1}"
        rows.append(row)

    return pd.DataFrame(rows).set_index("factor")


# -----------------------------------------------------------------------------
# 7. SUB-PERIOD ANALYSIS
# -----------------------------------------------------------------------------

def sub_period_analysis(joint_betas:      pd.DataFrame,
                        ic_series:        dict,
                        portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    """lambda_mean, NW t-stat, IC, and L/S Sharpe for each factor x sub-period."""
    active = [f for f in FACTORS if f in joint_betas.columns]
    rows   = []

    for period_label, (start, end) in SUB_PERIODS.items():
        for fname in active:
            # FM betas
            betas = joint_betas[fname].copy()
            if start: betas = betas[betas.index >= start]
            if end:   betas = betas[betas.index <= end]
            mu, t = nw_tstat(betas)

            # IC
            ic = ic_series.get(fname, pd.Series(dtype=float)).copy()
            if start: ic = ic[ic.index >= start]
            if end:   ic = ic[ic.index <= end]

            # L/S portfolio Sharpe
            ls_col = f"{fname}_LS"
            ls = portfolio_returns[ls_col].copy() \
                 if ls_col in portfolio_returns.columns \
                 else pd.Series(dtype=float)
            if start: ls = ls[ls.index >= start]
            if end:   ls = ls[ls.index <= end]
            ls = ls.dropna()

            ann_ret = ls.mean() * 12
            ann_vol = ls.std() * np.sqrt(12)
            sharpe  = ann_ret / ann_vol if ann_vol > 0 and len(ls) > 2 else np.nan

            rows.append({
                "period":       period_label,
                "factor":       FACTOR_LABELS.get(fname, fname),
                "lambda_bps":   mu * 10_000 if not np.isnan(mu) else np.nan,
                "t_stat":       t,
                "IC_mean":      ic.mean(),
                "IC_std":       ic.std(),
                "sharpe":       sharpe,
                "T":            int(betas.notna().sum()),
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 8. PRINT HELPERS
# -----------------------------------------------------------------------------

def _sig(t: float) -> str:
    if np.isnan(t): return " "
    if abs(t) >= 2.576: return "***"
    if abs(t) >= 1.960: return "** "
    if abs(t) >= 1.645: return "*  "
    return "   "


def print_fm_table(monthly_betas: pd.DataFrame, label: str = "JOINT"):
    w = 68
    print(f"\n  {'-'*w}")
    print(f"  {label} FAMA-MACBETH  (Newey-West {NW_LAGS}-lag t-statistics)")
    print(f"  Stars: ***p<1%  **p<5%  *p<10%")
    print(f"  {'-'*w}")
    print(f"  {'Factor':<12} {'lambda (bps/mo)':>12} {'t-stat':>10}  {'sig':>4}  "
          f"{'mean N':>7}  {'T':>5}")
    print(f"  {'-'*w}")
    for fname in FACTORS:
        if fname not in monthly_betas.columns:
            continue
        mu, t = nw_tstat(monthly_betas[fname])
        T     = int(monthly_betas[fname].notna().sum())
        avg_n = monthly_betas["n"].mean() if "n" in monthly_betas.columns else np.nan
        lam   = mu * 10_000 if not np.isnan(mu) else np.nan
        lab   = FACTOR_LABELS.get(fname, fname)
        print(f"  {lab:<12} {lam:>+11.1f}  {t:>+9.2f}  {_sig(t):>4}  "
              f"{avg_n:>7.0f}  {T:>5}")
    print(f"  {'-'*w}")


def print_ic_table(ic_series: dict):
    w = 70
    print(f"\n  {'-'*w}")
    print(f"  INFORMATION COEFFICIENT  (Spearman rank-IC, monthly)")
    print(f"  ICIR = IC_mean / IC_std x sqrt12  (annualised)")
    print(f"  {'-'*w}")
    print(f"  {'Factor':<12} {'IC mean':>9} {'IC std':>8} {'t(IC!=0)':>9} "
          f"{'ICIR ann':>10} {'Hit%':>7} {'T':>5}")
    print(f"  {'-'*w}")
    for fname in FACTORS:
        if fname not in ic_series:
            continue
        ic   = ic_series[fname].dropna()
        T    = len(ic)
        if T < 5:
            continue
        mu   = ic.mean()
        sd   = ic.std()
        t_ic = mu / (sd / np.sqrt(T)) if sd > 0 else np.nan
        icir = mu / sd * np.sqrt(12)  if sd > 0 else np.nan
        hit  = (ic > 0).mean() * 100
        lab  = FACTOR_LABELS.get(fname, fname)
        print(f"  {lab:<12} {mu:>+8.4f}  {sd:>7.4f}  {t_ic:>+8.2f}  "
              f"{icir:>+9.2f}  {hit:>6.1f}%  {T:>4}")
    print(f"  {'-'*w}")


def print_quintile_table(q_df: pd.DataFrame):
    cols = [c for c in ["Q1","Q2","Q3","Q4","Q5","Q1-Q5","monotone"]
            if c in q_df.columns]
    num_cols  = [c for c in cols if c != "monotone"]
    has_mono  = "monotone" in cols
    w = 12 + len(num_cols) * 9 + (8 if has_mono else 0) + 4

    print(f"\n  {'-'*w}")
    print(f"  QUINTILE MONOTONICITY  (annualised mean return %)")
    print(f"  Q1 = best (long), Q5 = worst (short)")
    print(f"  {'-'*w}")
    header = f"  {'Factor':<12}" + "".join(f"{c:>9}" for c in num_cols)
    if has_mono:
        header += f"  {'mono':>6}"
    print(header)
    print(f"  {'-'*w}")
    for factor, row in q_df.iterrows():
        line = f"  {factor:<12}"
        for c in num_cols:
            v = row.get(c, np.nan)
            line += f" {v:>+8.1f}%" if not np.isnan(v) else f"{'N/A':>9}"
        if has_mono:
            line += f"  {row.get('monotone', ''):>6}"
        print(line)
    print(f"  {'-'*w}")


def print_sub_period_table(sub_df: pd.DataFrame):
    print(f"\n  SUB-PERIOD ANALYSIS")

    for period in SUB_PERIODS:
        block = sub_df[sub_df["period"] == period]
        if block.empty:
            continue
        print(f"\n  -- {period}")
        print(f"  {'Factor':<12} {'lambda (bps)':>10} {'t-stat':>10} "
              f"{'IC':>8} {'Sharpe':>8} {'T':>5}")
        print(f"  {'-'*55}")
        for _, row in block.iterrows():
            t   = row["t_stat"]
            sig = _sig(t).strip()
            sig_str = f"{t:>+7.2f}{sig:<3}" if not np.isnan(t) else "   N/A   "
            lam = row["lambda_bps"]
            ic  = row["IC_mean"]
            sh  = row["sharpe"]
            T   = row["T"]
            print(f"  {row['factor']:<12} {lam:>+9.1f}  {sig_str}  "
                  f"{ic:>+7.3f}  {sh:>+7.2f}  {T:>4}")


# -----------------------------------------------------------------------------
# 9. MAIN
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="GCC Factor Research -- Layer 3a: Fama-MacBeth"
    )
    p.add_argument("--univariate", action="store_true",
                   help="Print univariate FM tables (each factor standalone)")
    p.add_argument("--verbose", action="store_true",
                   help="Print monthly coefficient series")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("  GCC EQUITY FACTOR RESEARCH -- Layer 3a: Fama-MacBeth")
    print("=" * 65)

    # -- Load --------------------------------------------------------------
    print("\n[1/5] Loading data ...")
    factor_scores     = load_factor_scores()
    monthly_returns   = load_monthly_returns()
    portfolio_returns = load_portfolio_returns()

    avail = [f for f in FACTORS if f in factor_scores]
    print(f"  Factors available: {', '.join(avail)}")
    start = portfolio_returns.index[0].date()
    end   = portfolio_returns.index[-1].date()
    T_total = len(portfolio_returns)
    print(f"  Period: {start} -> {end}  |  {T_total} months of portfolio returns")

    # -- FM regressions ----------------------------------------------------
    print("\n[2/5] Running Fama-MacBeth regressions ...")
    joint_betas = run_fama_macbeth(
        factor_scores, monthly_returns, avail, label="Joint"
    )

    univariate_betas = {}
    for fname in avail:
        univariate_betas[fname] = run_fama_macbeth(
            factor_scores, monthly_returns, [fname],
            label=f"Uni-{FACTOR_LABELS.get(fname, fname)}"
        )

    # -- IC ----------------------------------------------------------------
    print("\n[3/5] Computing Spearman IC ...")
    ic_series = compute_ic(factor_scores, monthly_returns)
    for fname in avail:
        ic = ic_series.get(fname, pd.Series(dtype=float)).dropna()
        print(f"  {FACTOR_LABELS.get(fname, fname):<12} IC={ic.mean():+.4f}  "
              f"T={len(ic)}")

    # -- Quintile spread ---------------------------------------------------
    print("\n[4/5] Quintile monotonicity ...")
    q_spread = quintile_spread(portfolio_returns)

    # -- Sub-period --------------------------------------------------------
    sub_df = sub_period_analysis(joint_betas, ic_series, portfolio_returns)

    # -- Print results -----------------------------------------------------
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)

    print_fm_table(joint_betas, "JOINT")

    if args.univariate:
        for fname in avail:
            ub = univariate_betas[fname]
            if not ub.empty:
                print_fm_table(ub, f"UNIVARIATE -- {FACTOR_LABELS.get(fname, fname)}")

    print_ic_table(ic_series)
    print_quintile_table(q_spread)
    print_sub_period_table(sub_df)

    if args.verbose:
        print("\n  Monthly FM coefficients (joint):")
        print(joint_betas[[f for f in avail if f in joint_betas.columns]].to_string())

    # -- Save outputs ------------------------------------------------------
    print(f"\n[5/5] Saving ...")

    joint_betas.to_csv(os.path.join(OUTPUT_DIR, "fm_coefficients.csv"))
    print(f"  data/fm_coefficients.csv")

    # fm_summary: one row per factor, all key stats
    summary_rows = []
    for fname in avail:
        mu_j, t_j = nw_tstat(joint_betas[fname]) \
                    if fname in joint_betas.columns else (np.nan, np.nan)
        ub = univariate_betas.get(fname, pd.DataFrame())
        mu_u, t_u = nw_tstat(ub[fname]) \
                    if (not ub.empty and fname in ub.columns) else (np.nan, np.nan)
        ic   = ic_series.get(fname, pd.Series(dtype=float))
        ic_m = ic.mean();  ic_s = ic.std();  T_ic = len(ic.dropna())
        t_ic = ic_m / (ic_s / np.sqrt(T_ic)) if ic_s > 0 and T_ic > 0 else np.nan
        icir = ic_m / ic_s * np.sqrt(12) if ic_s > 0 else np.nan
        ls_col = f"{fname}_LS"
        ls     = portfolio_returns[ls_col].dropna() \
                 if ls_col in portfolio_returns.columns else pd.Series(dtype=float)
        ann_r  = ls.mean() * 12
        ann_v  = ls.std()  * np.sqrt(12)
        sharpe = ann_r / ann_v if ann_v > 0 else np.nan
        summary_rows.append({
            "factor":           FACTOR_LABELS.get(fname, fname),
            "lambda_joint_bps": mu_j * 10_000 if not np.isnan(mu_j) else np.nan,
            "t_joint_nw3":      t_j,
            "sig_joint":        _sig(t_j).strip(),
            "lambda_uni_bps":   mu_u * 10_000 if not np.isnan(mu_u) else np.nan,
            "t_uni_nw3":        t_u,
            "sig_uni":          _sig(t_u).strip(),
            "IC_mean":          ic_m,
            "IC_std":           ic_s,
            "t_IC":             t_ic,
            "ICIR_ann":         icir,
            "port_ann_ret_pct": ann_r * 100,
            "port_ann_vol_pct": ann_v * 100,
            "port_sharpe":      sharpe,
            "T_months":         int(joint_betas[fname].notna().sum())
                                if fname in joint_betas.columns else 0,
        })

    summary_df = pd.DataFrame(summary_rows).set_index("factor")
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "fm_summary.csv"))
    print(f"  data/fm_summary.csv")

    ic_df = pd.DataFrame(
        {FACTOR_LABELS.get(f, f): s for f, s in ic_series.items()}
    )
    ic_df.to_csv(os.path.join(OUTPUT_DIR, "ic_monthly.csv"))
    print(f"  data/ic_monthly.csv")

    q_spread.to_csv(os.path.join(OUTPUT_DIR, "quintile_spreads.csv"))
    print(f"  data/quintile_spreads.csv")

    sub_df.to_csv(os.path.join(OUTPUT_DIR, "sub_period_analysis.csv"), index=False)
    print(f"  data/sub_period_analysis.csv")

    # -- Final summary -----------------------------------------------------
    print()
    print("=" * 65)
    print("  LAYER 3a COMPLETE -- Statistical Significance Summary")
    print("=" * 65)
    print()
    print(f"  {'Factor':<12} {'lambda jt (bps)':>11} {'t-stat':>9}  "
          f"{'sig':>5}  {'ICIR ann':>10}  {'L/S Sharpe':>11}")
    print("  " + "-" * 62)

    for fname in avail:
        mu, t = nw_tstat(joint_betas[fname]) \
                if fname in joint_betas.columns else (np.nan, np.nan)
        ic   = ic_series.get(fname, pd.Series(dtype=float))
        ic_m = ic.mean();  ic_s = ic.std()
        icir = ic_m / ic_s * np.sqrt(12) if ic_s > 0 else np.nan
        ls_col = f"{fname}_LS"
        ls     = portfolio_returns[ls_col].dropna() \
                 if ls_col in portfolio_returns.columns else pd.Series(dtype=float)
        sharpe = (ls.mean() * 12) / (ls.std() * np.sqrt(12)) \
                 if len(ls) > 2 else np.nan
        lam  = mu * 10_000 if not np.isnan(mu) else float("nan")
        lab  = FACTOR_LABELS.get(fname, fname)
        print(f"  {lab:<12} {lam:>+10.1f}  {t:>+8.2f}  "
              f"{_sig(t):>5}  {icir:>+9.2f}  {sharpe:>+10.2f}")

    print()
    print("  Significance: ***p<1%  **p<5%  *p<10%  (NW 3-lag t-stat)")
    print()
    print("  Files written to:", OUTPUT_DIR)
    print()
    print("  Next: python backtest.py   (Layer 3b -- drawdown, turnover, IC decay)")
    print()


if __name__ == "__main__":
    main()
