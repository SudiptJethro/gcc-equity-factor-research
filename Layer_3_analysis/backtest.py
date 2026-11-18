"""
GCC Equity Factor Research Engine
Layer 3b: Standard Factor Attribution Backtest
===============================================
Academic-standard Q1-Q5 factor backtest across all five GCC exchanges.
Separate from portfolio.py (which implements the sleeve strategy on top of these results).

For each factor (value, momentum, quality, lowvol) and each exchange
(pooled GCC, ADX, DFM, Tadawul, QSE, Kuwait):
  - Assign stocks to quintiles Q1 (best score) through Q5 (worst) each month
  - Compute equal-weighted Q1-Q5 monthly returns
  - Report: annualised return per quintile, Q1-Q5 spread Sharpe, max drawdown,
            monotonicity check, IC, 50bps round-trip transaction cost impact
  - Value factor: shown in full with sign-flip callout (GCC finding, not a bug)
  - Kuwait: split pre-MSCI (<June 2020) and post-MSCI (>=June 2020)

Reads from:   data/factor_scores.parquet
              data/returns_{exchange}.parquet
              data/portfolio_returns.parquet

Writes to:    data/factors/quintile_returns_{exchange}_{factor}.csv
              data/factors/backtest_summary.csv
              data/factors/kuwait_msci_split.csv

Run:
  python Layer_3_analysis/backtest.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.dirname(_HERE)
DATA_DIR   = os.path.join(_ROOT, "data")
OUT_DIR    = os.path.join(_ROOT, "data", "factors")
os.makedirs(OUT_DIR, exist_ok=True)

FACTORS = ["value", "mom", "quality", "lowvol"]
FACTOR_LABELS = {
    "value":   "VALUE",
    "mom":     "MOMENTUM",
    "quality": "QUALITY",
    "lowvol":  "LOW-VOL",
}

# Exchange identified by ticker suffix
EXCHANGE_SUFFIX = {
    "ADX":     ".AD",
    "DFM":     ".DU",
    "Tadawul": ".SR",
    "QSE":     ".QA",
    "Kuwait":  ".KW",
}
ALL_EXCHANGES_DATA = ["adx", "dfm", "tadawul", "qatar", "kuwait"]
KUWAIT_START       = pd.Timestamp("2016-01-01")
KUWAIT_MSCI_DATE   = pd.Timestamp("2020-06-01")

TC_BPS    = 50    # round-trip transaction cost (bps) / standard academic assumption
N_QUINTILES = 5
MIN_STOCKS  = 10  # minimum stocks per quintile assignment


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def load_factor_scores() -> dict:
    """Load factor z-scores; return {factor: DataFrame(date x ticker)}."""
    path = os.path.join(DATA_DIR, "factor_scores.parquet")
    df   = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    out = {}
    for f in FACTORS:
        if f in df.columns.get_level_values(0):
            out[f] = df[f].copy()
    return out


def load_monthly_returns() -> pd.DataFrame:
    """Concatenate returns across all exchanges; Kuwait filtered from 2016."""
    frames = []
    for ex in ALL_EXCHANGES_DATA:
        p = os.path.join(DATA_DIR, f"returns_{ex}.parquet")
        if not os.path.exists(p):
            continue
        r = pd.read_parquet(p)
        r.index = pd.to_datetime(r.index)
        if ex == "kuwait":
            r = r.loc[r.index >= KUWAIT_START]
        frames.append(r)
    combined = pd.concat(frames, axis=1)
    return combined.loc[:, ~combined.columns.duplicated()]


def exchange_tickers(all_tickers: list) -> dict:
    """Map exchange name to list of tickers using suffix convention."""
    mapping = {"GCC": list(all_tickers)}
    for name, suffix in EXCHANGE_SUFFIX.items():
        mapping[name] = [t for t in all_tickers if t.endswith(suffix)]
    return mapping


# ---------------------------------------------------------------------------
# 2. CORE METRICS
# ---------------------------------------------------------------------------

def annualise(series: pd.Series) -> float:
    return series.mean() * 12


def ann_vol(series: pd.Series) -> float:
    return series.std() * np.sqrt(12)


def sharpe(series: pd.Series) -> float:
    av = ann_vol(series)
    return annualise(series) / av if av > 0 else np.nan


def max_drawdown(series: pd.Series) -> float:
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def hit_rate(series: pd.Series) -> float:
    return (series > 0).mean()


def monthly_ic(factor_row: pd.Series, ret_row: pd.Series) -> float:
    """Spearman rank-IC between factor scores and next-month returns."""
    common = factor_row.dropna().index.intersection(ret_row.dropna().index)
    if len(common) < 10:
        return np.nan
    rho, _ = spearmanr(factor_row[common], ret_row[common])
    return rho


def monotonicity(q_returns: dict) -> tuple:
    """
    Check Q1 > Q2 > Q3 > Q4 > Q5.
    Returns (n_correct_pairs/4, spearman_rank_corr of quintile_number vs return).
    """
    vals = [q_returns.get(q, np.nan) for q in [1, 2, 3, 4, 5]]
    if any(np.isnan(v) for v in vals):
        return np.nan, np.nan
    pairs_correct = sum(1 for i in range(4) if vals[i] > vals[i + 1])
    rho, _ = spearmanr([1, 2, 3, 4, 5], vals)
    return pairs_correct, rho   # pairs_correct out of 4; rho should be negative for working factor


# ---------------------------------------------------------------------------
# 3. QUINTILE ASSIGNMENT + PORTFOLIO RETURNS
# ---------------------------------------------------------------------------

def quintile_assign(scores_t: pd.Series, tickers: list) -> pd.Series:
    """Assign quintiles within the given ticker subset for one month."""
    sub = scores_t.reindex(tickers).dropna()
    if len(sub) < MIN_STOCKS:
        return pd.Series(dtype=float)
    q = pd.qcut(sub.rank(method="first"), q=N_QUINTILES,
                 labels=[5, 4, 3, 2, 1])   # reverse: highest score -> Q1
    return q.astype(float)


def build_quintile_returns(factor_scores: pd.DataFrame,
                            monthly_returns: pd.DataFrame,
                            tickers: list,
                            tc_bps: float = TC_BPS) -> pd.DataFrame:
    """
    Build monthly Q1-Q5 portfolio returns for one factor × one exchange subset.
    Returns DataFrame with columns Q1..Q5, LS (Q1-Q5 gross), LS_net (after 50bps TC).
    """
    all_dates = sorted(set(factor_scores.index) & set(monthly_returns.index))
    rows = {}

    prev_long  = set()
    prev_short = set()

    for i, date in enumerate(all_dates[:-1]):
        next_date = all_dates[i + 1]
        q_assign  = quintile_assign(factor_scores.loc[date], tickers)
        if q_assign.empty:
            continue

        ret_next = monthly_returns.loc[next_date]
        row      = {}
        q1_set   = set(q_assign[q_assign == 1].index)
        q5_set   = set(q_assign[q_assign == 5].index)

        for q in range(1, N_QUINTILES + 1):
            stocks = q_assign[q_assign == q].index
            valid  = ret_next.reindex(stocks).dropna()
            row[f"Q{q}"] = valid.mean() if len(valid) > 0 else np.nan

        ls = (row.get("Q1", np.nan) - row.get("Q5", np.nan))
        row["LS"] = ls

        # Transaction cost: 50bps on fraction of portfolio that changes
        q1_turnover = len(q1_set.symmetric_difference(prev_long))   / max(len(q1_set), 1)
        q5_turnover = len(q5_set.symmetric_difference(prev_short))  / max(len(q5_set), 1)
        avg_turnover = (q1_turnover + q5_turnover) / 2
        tc_drag = avg_turnover * (tc_bps / 10_000)
        row["LS_net"] = ls - tc_drag if not np.isnan(ls) else np.nan

        rows[next_date] = row
        prev_long  = q1_set
        prev_short = q5_set

    result = pd.DataFrame(rows).T
    result.index = pd.to_datetime(result.index)
    result.index.name = "date"
    return result


# ---------------------------------------------------------------------------
# 4. SUMMARY STATS FOR ONE BACKTEST SLICE
# ---------------------------------------------------------------------------

def summarise_backtest(qt_df: pd.DataFrame, factor: str,
                       exchange: str) -> dict:
    """Compute all metrics for one factor × exchange combination."""
    rec = {"exchange": exchange, "factor": FACTOR_LABELS.get(factor, factor)}

    q_ann = {}
    for q in range(1, N_QUINTILES + 1):
        col = f"Q{q}"
        if col in qt_df.columns:
            q_ann[q] = annualise(qt_df[col].dropna())
            rec[f"Q{q}_ann_pct"] = round(q_ann[q] * 100, 2)
        else:
            q_ann[q] = np.nan
            rec[f"Q{q}_ann_pct"] = np.nan

    if "LS" in qt_df.columns:
        ls = qt_df["LS"].dropna()
        rec["LS_ann_pct"]    = round(annualise(ls) * 100, 2)
        rec["LS_vol_pct"]    = round(ann_vol(ls) * 100, 2)
        rec["LS_sharpe"]     = round(sharpe(ls), 3)
        rec["LS_maxdd_pct"]  = round(max_drawdown(ls) * 100, 2)
        rec["LS_hit_pct"]    = round(hit_rate(ls) * 100, 1)

    if "LS_net" in qt_df.columns:
        ls_net = qt_df["LS_net"].dropna()
        rec["LS_net_ann_pct"] = round(annualise(ls_net) * 100, 2)
        rec["LS_net_sharpe"]  = round(sharpe(ls_net), 3)

    pairs_ok, mono_rho = monotonicity(q_ann)
    rec["monotone_pairs"] = f"{int(pairs_ok)}/4" if not np.isnan(pairs_ok) else "N/A"
    rec["monotone_rho"]   = round(mono_rho, 3) if not np.isnan(mono_rho) else np.nan

    # Value sign-flip flag
    if factor == "value" and "LS_ann_pct" in rec:
        if rec["LS_ann_pct"] < 0:
            rec["value_note"] = (
                f"Value excluded from portfolio / negative sign in GCC "
                f"(annualised spread: {rec['LS_ann_pct']:.1f}%). "
                "Documented as a regional finding, not a data error."
            )

    return rec


# ---------------------------------------------------------------------------
# 5. PER-EXCHANGE IC SERIES
# ---------------------------------------------------------------------------

def compute_ic_series(factor_scores: pd.DataFrame,
                      monthly_returns: pd.DataFrame,
                      tickers: list) -> pd.Series:
    """Monthly Spearman IC for one factor × exchange subset."""
    all_dates = sorted(set(factor_scores.index) & set(monthly_returns.index))
    ic_vals   = {}
    for i, date in enumerate(all_dates[:-1]):
        next_date = all_dates[i + 1]
        scores_t  = factor_scores.loc[date].reindex(tickers).dropna()
        ret_t     = monthly_returns.loc[next_date].reindex(scores_t.index).dropna()
        valid_idx = ret_t.index
        if len(valid_idx) < 10:
            continue
        rho, _ = spearmanr(scores_t[valid_idx], ret_t)
        ic_vals[next_date] = rho
    return pd.Series(ic_vals)


# ---------------------------------------------------------------------------
# 6. MAIN BACKTEST LOOP
# ---------------------------------------------------------------------------

def run_all_backtests(factor_scores: dict,
                      monthly_returns: pd.DataFrame) -> tuple:
    """
    Run backtest for every factor × exchange combination.
    Returns (summary_rows, qt_tables) where qt_tables is dict keyed by (exchange, factor).
    """
    all_tickers    = monthly_returns.columns.tolist()
    exchange_map   = exchange_tickers(all_tickers)
    summary_rows   = []
    qt_tables      = {}

    exchanges_ordered = ["GCC", "ADX", "DFM", "Tadawul", "QSE", "Kuwait"]

    for exch in exchanges_ordered:
        tickers = exchange_map.get(exch, [])
        if not tickers:
            print(f"  [WARN] No tickers for {exch} / skipping")
            continue

        for fname in FACTORS:
            if fname not in factor_scores:
                continue
            fdf  = factor_scores[fname]
            label = FACTOR_LABELS.get(fname, fname)
            print(f"  {exch:10s} | {label:<10s} | {len(tickers)} tickers")

            qt = build_quintile_returns(fdf, monthly_returns, tickers)
            if qt.empty:
                continue

            qt_tables[(exch, fname)] = qt

            # IC
            ic = compute_ic_series(fdf, monthly_returns, tickers)
            rec = summarise_backtest(qt, fname, exch)
            rec["IC_mean"]  = round(ic.mean(), 4)  if len(ic) > 0 else np.nan
            rec["IC_std"]   = round(ic.std(),  4)  if len(ic) > 0 else np.nan
            rec["IC_t"]     = round(
                ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else np.nan, 2
            )
            rec["T_months"] = len(qt)
            summary_rows.append(rec)

    return summary_rows, qt_tables


# ---------------------------------------------------------------------------
# 7. KUWAIT MSCI SPLIT
# ---------------------------------------------------------------------------

def run_kuwait_msci_split(factor_scores: dict,
                           monthly_returns: pd.DataFrame) -> list:
    """
    Run full factor attribution for Kuwait, split pre/post MSCI upgrade (June 2020).
    Shows whether factor premia compressed post-upgrade.
    """
    kw_tickers = [t for t in monthly_returns.columns if t.endswith(".KW")]
    if not kw_tickers:
        print("  [WARN] No Kuwait tickers found / skipping MSCI split")
        return []

    kw_ret = monthly_returns[kw_tickers]
    rows   = []

    for period_label, date_filter in [
        ("Pre-MSCI  (<Jun-2020)",   lambda idx: idx <  KUWAIT_MSCI_DATE),
        ("Post-MSCI (>=Jun-2020)",  lambda idx: idx >= KUWAIT_MSCI_DATE),
    ]:
        kw_ret_period = kw_ret.loc[date_filter(kw_ret.index)]

        for fname in FACTORS:
            if fname not in factor_scores:
                continue
            fdf_period = factor_scores[fname].loc[date_filter(factor_scores[fname].index)]
            qt = build_quintile_returns(fdf_period, kw_ret_period, kw_tickers)
            if qt.empty:
                continue

            ic = compute_ic_series(fdf_period, kw_ret_period, kw_tickers)
            rec = summarise_backtest(qt, fname, f"Kuwait {period_label}")
            rec["period"]   = period_label
            rec["IC_mean"]  = round(ic.mean(), 4) if len(ic) > 0 else np.nan
            rec["T_months"] = len(qt)
            rows.append(rec)

    return rows


# ---------------------------------------------------------------------------
# 8. PRINT HELPERS
# ---------------------------------------------------------------------------

def print_exchange_table(summary_rows: list, exchange: str) -> None:
    rows = [r for r in summary_rows if r["exchange"] == exchange]
    if not rows:
        return
    sep = "-" * 72
    print(f"\n  {sep}")
    print(f"  {exchange.upper()} -- Quintile Returns (annualised %)")
    print(f"  {sep}")
    print(f"  {'Factor':<12} {'Q1':>7} {'Q2':>7} {'Q3':>7} {'Q4':>7} {'Q5':>7} "
          f"{'LS':>8} {'Sharpe':>8} {'MaxDD':>8} {'Mono':>6}")
    print(f"  {sep}")
    for r in rows:
        q_vals = "  ".join(
            f"{r.get(f'Q{q}_ann_pct', np.nan):>+6.1f}%" for q in range(1, 6)
        )
        ls     = r.get("LS_ann_pct",   np.nan)
        sh     = r.get("LS_sharpe",    np.nan)
        dd     = r.get("LS_maxdd_pct", np.nan)
        mono   = r.get("monotone_pairs", "N/A")
        print(f"  {r['factor']:<12} {q_vals}  {ls:>+7.1f}%  {sh:>7.2f}  "
              f"{dd:>+7.1f}%  {mono:>5}")
        if "value_note" in r:
            print(f"    ** {r['value_note']}")
    print(f"  {sep}")


# ---------------------------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  GCC EQUITY FACTOR RESEARCH - Layer 3b: Factor Attribution Backtest")
    print("  Transaction cost: 50bps round-trip  |  Equal-weighted quintiles")
    print("=" * 72)

    # -- Load ---------------------------------------------------------------
    print("\n[1/4] Loading data …")
    factor_scores   = load_factor_scores()
    monthly_returns = load_monthly_returns()
    print(f"  Factors: {list(factor_scores.keys())}")
    print(f"  Universe: {monthly_returns.shape[1]} stocks  |  "
          f"{monthly_returns.shape[0]} months")

    # -- Run backtests ------------------------------------------------------
    print("\n[2/4] Running exchange × factor backtests …")
    summary_rows, qt_tables = run_all_backtests(factor_scores, monthly_returns)

    # -- Kuwait MSCI split --------------------------------------------------
    print("\n[3/4] Kuwait MSCI split (pre/post June 2020) …")
    kw_rows = run_kuwait_msci_split(factor_scores, monthly_returns)

    # -- Print results ------------------------------------------------------
    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)
    for exch in ["GCC", "ADX", "DFM", "Tadawul", "QSE", "Kuwait"]:
        print_exchange_table(summary_rows, exch)

    if kw_rows:
        print(f"\n  {'-'*60}")
        print(f"  KUWAIT -- MSCI Upgrade Effect")
        print(f"  {'-'*60}")
        for period in ["Pre-MSCI  (<Jun-2020)", "Post-MSCI (>=Jun-2020)"]:
            prows = [r for r in kw_rows if r.get("period") == period]
            if not prows:
                continue
            print(f"  {period}")
            for r in prows:
                ls  = r.get("LS_ann_pct", np.nan)
                sh  = r.get("LS_sharpe",  np.nan)
                ic  = r.get("IC_mean",    np.nan)
                T   = r.get("T_months",   0)
                print(f"    {r['factor']:<12} LS={ls:>+6.1f}%  Sharpe={sh:>5.2f}  "
                      f"IC={ic:>+.3f}  T={T}")

    # -- Save outputs -------------------------------------------------------
    print("\n[4/4] Saving outputs …")

    # Individual quintile CSVs
    for (exch, fname), qt in qt_tables.items():
        exch_slug = exch.lower().replace(" ", "_")
        fpath = os.path.join(OUT_DIR, f"quintile_returns_{exch_slug}_{fname}.csv")
        qt.to_csv(fpath)

    print(f"  Saved {len(qt_tables)} quintile_returns_*.csv files to data/factors/")

    # Master summary table
    all_rows = summary_rows + kw_rows
    summary_df = pd.DataFrame(all_rows)
    summary_path = os.path.join(OUT_DIR, "backtest_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: data/factors/backtest_summary.csv  ({len(summary_df)} rows)")

    # Kuwait MSCI split table
    if kw_rows:
        kw_df = pd.DataFrame(kw_rows)
        kw_path = os.path.join(OUT_DIR, "kuwait_msci_split.csv")
        kw_df.to_csv(kw_path, index=False)
        print(f"  Saved: data/factors/kuwait_msci_split.csv")

    # Final summary
    print()
    print("=" * 72)
    print("  BACKTEST COMPLETE / Key findings (GCC pooled)")
    print("=" * 72)
    gcc_rows = [r for r in summary_rows if r["exchange"] == "GCC"]
    print(f"  {'Factor':<12} {'LS ann%':>9} {'Sharpe':>8} {'MaxDD%':>9} "
          f"{'IC mean':>9} {'Mono':>6} {'Note'}")
    print("  " + "-" * 68)
    for r in gcc_rows:
        note = "** SIGN FLIP **" if "value_note" in r else ""
        print(f"  {r['factor']:<12} "
              f"{r.get('LS_ann_pct', np.nan):>+8.1f}%  "
              f"{r.get('LS_sharpe',  np.nan):>7.2f}  "
              f"{r.get('LS_maxdd_pct', np.nan):>+8.1f}%  "
              f"{r.get('IC_mean', np.nan):>+8.4f}  "
              f"{r.get('monotone_pairs', 'N/A'):>5}  {note}")
    print()
    print("  Next: streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
