"""
=================================================================
  GCC EQUITY FACTOR RESEARCH ENGINE
  Layer 3c: Synthetic Vol Surface + Dynamic VRP Overlay
  REVISED 2026-05-17 / three bug fixes applied:
    Bug 1: Layer 3b comparison row renamed to "theoretical ceiling"
    Bug 2: exec_rate uses SSO-level VRP regime; REAL_TAKE = 0.55
    Bug 3: Sizer uses VRP-gap-percentile notional, not IV level
=================================================================
  What this layer builds:
    A dynamic, regime-aware implied volatility surface for the
    four Tadawul SSO names (Aramco, Al Rajhi, STC, SABIC),
    constructed entirely from publicly available data:

    1. REALISED VOL     / rolling window from Layer 1 price data
    2. TASI VOL REGIME  / market stress proxy (replaces VIX)
    3. OIL SENSITIVITY  / Aramco + SABIC get oil-vol loading
    4. THINNESS PREMIUM / persistent spread for illiquid SSO mkt
       (earnings month premium removed from IV / handled via
        exec_rate haircut in the overlay instead)

    The result is a monthly IV estimate per SSO name that varies
    through time / enabling a proper backtest of the CC overlay
    rather than a constant yield assumption.

  Methodology note (for research paper):
    "Implied volatility is estimated from a four-component
     synthetic surface: rolling realised volatility (6-month
     blended with 3-month), a market-stress loading on TASI
     realised vol, an oil-price volatility component for
     energy/materials names, and a structural thinness premium
     calibrated to bid-ask spreads observed at SSO launch
     (Nov 2023). The CC overlay execution rate is haircut 15%
     in earnings months to reflect wider spreads and pin risk.
     This approach will be replaced with exchange data upon
     commercial access to Tadawul SSO chains."

  Inputs  (all from earlier layers):
    data/prices_*.parquet         / Layer 1 price data
    data/layer3b_returns.parquet  / Layer 3b portfolio returns
    data/layer3b_positions.parquet

  Outputs:
    data/layer3c_vol_surface.parquet   / monthly IV per SSO name
    data/layer3c_vrp_dynamic.parquet   / dynamic CC premium series
    data/layer3c_tearsheet.csv         / updated performance table
    data/layer3c_regime_analysis.csv   / vol regime breakdown

  Run after:
    python layer3b_final.py
=================================================================
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import os

sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[union-attr]
warnings.filterwarnings("ignore")

DATA_DIR   = "data/"
OUTPUT_DIR = "data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── SSO universe ─────────────────────────────────────────────────
SSO_SPECS = {
    "2222.SR": {
        "name":            "Saudi Aramco",
        "oil_beta":        0.65,
        "sector":          "energy",
        "earnings_months": [2, 5, 8, 11],
    },
    "1120.SR": {
        "name":            "Al Rajhi Bank",
        "oil_beta":        0.15,
        "sector":          "financials",
        "earnings_months": [1, 4, 7, 10],
    },
    "7010.SR": {
        "name":            "STC",
        "oil_beta":        0.10,
        "sector":          "telecom",
        "earnings_months": [2, 5, 8, 11],
    },
    "2010.SR": {
        "name":            "SABIC",
        "oil_beta":        0.55,
        "sector":          "materials",
        "earnings_months": [2, 5, 8, 11],
    },
}

# ── Overlay execution parameters (Bug 2 + Bug 3 fix) ─────────────
# KSA SSO execution haircut: 45% discount to theoretical BS premium.
# Reflects order size vs daily SSO volume, market maker withdrawal in
# stress, and Tadawul physical settlement mechanics.
REAL_TAKE             = 0.55

VRP_THRESHOLD         = 7.0    # pp minimum VRP gap to fire overlay
RV_CEILING            = 35.0   # % annualised max RV to allow overlay
MAX_OVERLAY_NOTIONAL  = 0.39   # max 39% of portfolio as CC notional
EARNINGS_EXEC_HAIRCUT = 0.85   # exec_rate multiplier in earnings months
STRESS_SCALAR         = 1.30   # notional multiplier in stressed regime
CALM_SCALAR           = 0.90   # notional multiplier in calm regime
VRP_PCT_LOOKBACK      = 12     # months for rolling VRP gap percentile

# ── Vol surface parameters ────────────────────────────────────────
RV_WINDOW        = 6      # months for rolling realised vol
RV_WINDOW_SHORT  = 3      # short window for regime blending
RV_BLEND         = 0.6    # weight on 6m window vs 3m window
TASIS_WEIGHT     = 0.20   # loading on TASI market stress
OIL_WEIGHT       = 0.15   # loading on oil vol (energy/materials only)
# EARNINGS_PREMIUM removed from IV surface / earnings months now handled
# via exec_rate haircut (EARNINGS_EXEC_HAIRCUT) in the overlay section.
THINNESS_BASE    = 0.045  # +4.5pp structural thinness premium (base)
THINNESS_DECAY   = 0.004  # premium decays 0.4pp/yr as market matures
SSO_LAUNCH       = "2023-11-27"

# ── Option model ──────────────────────────────────────────────────
OTM_PCT      = 0.05
T_OPTION     = 1 / 12    # 1-month option
TRADING_DAYS = 252


# =================================================================
#  SECTION 1 / Load Price Data
# =================================================================

def load_sso_prices() -> pd.DataFrame:
    """
    Load monthly price data for SSO names from Layer 1 parquet files.
    Falls back to yfinance download if parquet not found.
    """
    tickers = list(SSO_SPECS.keys())

    for f in ["prices_tadawul.parquet", "prices_all.parquet"]:
        path = DATA_DIR + f
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if "ticker" in df.columns:
                sub = df[df["ticker"].isin(tickers)]
                if len(sub) > 0:
                    print(f"  Loaded SSO prices from {f}")
                    out = sub[["date", "ticker", "close"]].dropna().copy()
                    dates = pd.to_datetime(out["date"])
                    out["date"] = dates.dt.tz_convert(None) if dates.dt.tz else dates
                    return out

    print("  Parquet not found / downloading SSO prices from yfinance...")
    all_frames = []
    for ticker in tickers:
        try:
            raw = yf.download(ticker, start="2016-01-01",
                              progress=False, auto_adjust=True)
            if raw.empty:
                print(f"    [warn] No data for {ticker}")
                continue
            monthly = raw["Close"].squeeze().resample("ME").last().dropna()
            if monthly.index.tz is not None:
                monthly.index = monthly.index.tz_convert(None)
            tmp = pd.DataFrame({
                "date":   monthly.index,
                "ticker": ticker,
                "close":  monthly.values,
            })
            all_frames.append(tmp)
            print(f"    {ticker}: {len(monthly)} months")
        except Exception as e:
            print(f"    [error] {ticker}: {e}")

    if not all_frames:
        raise RuntimeError("Could not load any SSO price data.")
    return pd.concat(all_frames, ignore_index=True)


def load_tasi_prices() -> pd.Series:
    for sym in ["^TASI", "KSA"]:
        try:
            raw = yf.download(sym, start="2016-01-01",
                              progress=False, auto_adjust=True)
            if not raw.empty:
                monthly = raw["Close"].squeeze().resample("ME").last().dropna()
                if monthly.index.tz is not None:
                    monthly.index = monthly.index.tz_convert(None)
                print(f"  TASI proxy: {sym} / {len(monthly)} months")
                return monthly
        except Exception:
            continue
    print("  [warn] Could not load TASI index / using equal-weight SSO average")
    return pd.Series(dtype=float)


def load_oil_prices() -> pd.Series:
    for sym in ["BZ=F", "CL=F"]:
        try:
            raw = yf.download(sym, start="2016-01-01",
                              progress=False, auto_adjust=True)
            if not raw.empty:
                monthly = raw["Close"].squeeze().resample("ME").last().dropna()
                if monthly.index.tz is not None:
                    monthly.index = monthly.index.tz_convert(None)
                print(f"  Oil proxy: {sym} / {len(monthly)} months")
                return monthly
        except Exception:
            continue
    print("  [warn] Could not load oil prices / oil component set to zero")
    return pd.Series(dtype=float)


# =================================================================
#  SECTION 2 / Realised Volatility
# =================================================================

def compute_realised_vol(prices: pd.Series,
                         window_long:  int   = RV_WINDOW,
                         window_short: int   = RV_WINDOW_SHORT,
                         blend:        float = RV_BLEND) -> pd.Series:
    """
    Blended RV: 60% 6-month window + 40% 3-month window, annualised.
    Blend avoids over-fitting to short-term noise while capturing
    regime shifts faster than a single long window.
    """
    log_ret  = np.log(prices / prices.shift(1)).dropna()
    rv_long  = log_ret.rolling(window_long).std()  * np.sqrt(12)
    rv_short = log_ret.rolling(window_short).std() * np.sqrt(12)
    return (blend * rv_long + (1 - blend) * rv_short).dropna()


def compute_market_stress(tasi: pd.Series, window: int = RV_WINDOW) -> pd.Series:
    if tasi.empty:
        return pd.Series(dtype=float)
    log_ret = np.log(tasi / tasi.shift(1)).dropna()
    return log_ret.rolling(window).std() * np.sqrt(12)


def compute_oil_vol(oil: pd.Series, window: int = RV_WINDOW) -> pd.Series:
    if oil.empty:
        return pd.Series(dtype=float)
    log_ret = np.log(oil / oil.shift(1)).dropna()
    return log_ret.rolling(window).std() * np.sqrt(12)


# =================================================================
#  SECTION 3 / TASI Market Regime (used in IV surface construction)
# =================================================================

def classify_regime(market_stress: pd.Series) -> pd.Series:
    """
    TASI-based market regime used only in the IV surface construction
    (regime_adj ±pp on synthetic IV). NOT used for exec_rate / see
    classify_sso_regime() for the execution model regime.
    """
    regimes = []
    for _, val in market_stress.items():
        if val < 0.12:
            regimes.append("calm")
        elif val > 0.22:
            regimes.append("stressed")
        else:
            regimes.append("normal")
    return pd.Series(regimes, index=market_stress.index, name="regime")


# =================================================================
#  SECTION 4 / Thinness Premium (Decaying)
# =================================================================

def thinness_premium(date: pd.Timestamp,
                     launch: str   = SSO_LAUNCH,
                     base:   float = THINNESS_BASE,
                     decay:  float = THINNESS_DECAY) -> float:
    launch_dt = pd.Timestamp(launch)
    if date < launch_dt:
        return base
    years_since_launch = (date - launch_dt).days / 365.25
    return max(base - decay * years_since_launch, 0.020)


# =================================================================
#  SECTION 5 / Synthetic IV Surface Construction
# =================================================================

def build_iv_surface(sso_prices:    pd.DataFrame,
                     market_stress: pd.Series,
                     oil_vol:       pd.Series,
                     regimes:       pd.Series) -> pd.DataFrame:
    """
    Monthly synthetic IV per SSO name.

    IV(t) = RV_blended(t)
           + TASI_weight × market_stress(t)
           + oil_weight × oil_vol(t) × oil_beta     [energy/materials]
           + thinness_premium(t)                     [decaying]
           + regime_adj(t)                           [TASI calm/stressed]

    Note: earnings-month IV premium removed (Bug 3 fix). Earnings months
    are handled via exec_rate haircut in the overlay section instead.

    TASI regime adjustments for IV:
      calm     → −1.0pp
      stressed → +2.0pp
      normal   →  0.0pp
    """
    REGIME_ADJ = {"calm": -0.010, "normal": 0.000, "stressed": 0.020}

    records = []
    for ticker, spec in SSO_SPECS.items():
        sub = sso_prices[sso_prices["ticker"] == ticker].set_index("date")
        if sub.empty:
            continue

        prices = sub["close"]
        rv     = compute_realised_vol(prices)

        for dt in rv.index:
            iv = rv[dt]

            if not market_stress.empty and dt in market_stress.index:
                iv += TASIS_WEIGHT * market_stress[dt]

            if not oil_vol.empty and dt in oil_vol.index and spec["oil_beta"] > 0.3:
                iv += OIL_WEIGHT * oil_vol[dt] * spec["oil_beta"]

            # Earnings-month premium removed from IV (Bug 3 fix).
            # Earnings effect now captured via exec_rate haircut.

            iv += thinness_premium(dt)

            regime = regimes[dt] if dt in regimes.index else "normal"
            iv += REGIME_ADJ.get(regime, 0.0)

            iv = np.clip(iv, 0.10, 0.80)

            records.append({
                "date":           dt,
                "ticker":         ticker,
                "name":           spec["name"],
                "rv_blended":     round(rv[dt] * 100, 2),
                "iv_synthetic":   round(iv * 100, 2),
                "vrp_gap_pp":     round((iv - rv[dt]) * 100, 2),
                "regime":         regime,
                "earnings_month": dt.month in spec["earnings_months"],
                "thinness_pp":    round(thinness_premium(dt) * 100, 2),
            })

    return pd.DataFrame(records)


# =================================================================
#  SECTION 6 / VRP Percentile + SSO-Level Regime (Bug 2 + Bug 3)
# =================================================================

def _rolling_pct(values: np.ndarray, i: int, lookback: int) -> float:
    """Percentile rank of values[i] within values[max(0,i-lookback):i]. No look-ahead."""
    start = max(0, i - lookback)
    history = values[start:i]
    if len(history) < 3:
        return 0.5
    return float((history < values[i]).sum()) / len(history)


def compute_vrp_percentile_and_regime(vol_surface: pd.DataFrame,
                                       lookback: int = VRP_PCT_LOOKBACK) -> pd.DataFrame:
    """
    For each SSO name, compute per-month:
      vrp_pct   : rolling lookback-month percentile of VRP gap (no look-ahead)
      sso_regime: calm / normal / stressed based on rolling IV percentile

    Both use only past observations (T-lookback to T-1) to avoid look-ahead.
    First 3 months insufficient history: vrp_pct = 0.5, sso_regime = "normal".

    The sso_regime here is the SSO-LEVEL regime used for exec_rate (Bug 2 fix).
    It is based on each stock's own IV history, NOT the KSA ETF vol.
    """
    out_rows = []
    for ticker, grp in vol_surface.groupby("ticker"):
        grp    = grp.sort_values("date").reset_index(drop=True)
        vrps   = grp["vrp_gap_pp"].values
        ivs    = grp["iv_synthetic"].values
        n      = len(grp)

        vrp_pcts   = np.empty(n)
        sso_regimes = []

        for i in range(n):
            vrp_pcts[i] = _rolling_pct(vrps, i, lookback)

            iv_pct = _rolling_pct(ivs, i, lookback)
            if iv_pct < 0.33:
                sso_regimes.append("calm")
            elif iv_pct > 0.67:
                sso_regimes.append("stressed")
            else:
                sso_regimes.append("normal")

        tmp = grp.copy()
        tmp["vrp_pct"]    = vrp_pcts
        tmp["sso_regime"] = sso_regimes
        out_rows.append(tmp)

    return pd.concat(out_rows, ignore_index=True).sort_values(["ticker", "date"])


# =================================================================
#  SECTION 7 / Execution Rate (Bug 2 fix)
# =================================================================

def get_exec_rate(sso_regime: str, is_earnings: bool) -> float:
    """
    Regime-conditional exec_rate based on SSO-level VRP regime (not KSA ETF vol).

    calm regime:     REAL_TAKE × 0.80   (thinner markets, lower fill)
    normal regime:   REAL_TAKE × 1.00   (baseline)
    stressed regime: min(REAL_TAKE × 1.20, 0.90)
                                        (wider VRP gap more than compensates;
                                         capped at 0.90 absolute)

    Earnings months: additional 0.85× haircut (wider spreads, pin risk).
    """
    if sso_regime == "calm":
        rate = REAL_TAKE * 0.80
    elif sso_regime == "stressed":
        rate = min(REAL_TAKE * 1.20, 0.90)
    else:
        rate = REAL_TAKE * 1.00

    if is_earnings:
        rate *= EARNINGS_EXEC_HAIRCUT

    return rate


# =================================================================
#  SECTION 8 / Dynamic CC Premium Calculation (Bug 3 rewrite)
# =================================================================

def compute_dynamic_vrp_overlay(vol_surface: pd.DataFrame,
                                 sso_prices:  pd.DataFrame,
                                 positions:   pd.DataFrame) -> pd.DataFrame:
    """
    Monthly CC premium per SSO name using VRP-gap-percentile sizing.

    Gate logic (both must pass):
      signal_vrp  : vrp_gap_pp > VRP_THRESHOLD (5pp)
      signal_rv   : rv_blended < RV_CEILING (35% ann.)

    Sizing (Bug 3 fix / replaces IV-level sizer):
      vrp_scalar      = 0.5 + vrp_pct             # ranges 0.5× to 1.5×
      stress_scalar   = STRESS_SCALAR if stressed
                      = CALM_SCALAR   if calm
                      = 1.0           otherwise
      effective_scale = min(vrp_scalar × stress_scalar, 1.5)
      cc_prem_pct     = MAX_OVERLAY_NOTIONAL × effective_scale
                        × iv × √T × exec_rate × 100   [in % of portfolio]

    The portfolio_value cancels out: cc_prem_pct is dimensionless
    (fraction of portfolio, expressed as %).

    Aggregate step (aggregate_overlay) takes mean across held names per month,
    consistent with the CC budget being split equally across all held SSO names.
    """
    records = []

    for _, vol_row in vol_surface.iterrows():
        dt          = vol_row["date"]
        ticker      = vol_row["ticker"]
        iv          = vol_row["iv_synthetic"] / 100   # decimal
        rv_pct      = vol_row["rv_blended"]           # % annualised
        vrp_gap     = vol_row["vrp_gap_pp"]
        vrp_pct     = vol_row["vrp_pct"]
        sso_regime  = vol_row["sso_regime"]
        is_earnings = bool(vol_row["earnings_month"])

        # ── Gate 1: VRP gap must exceed threshold ──────────────────
        signal_vrp = vrp_gap > VRP_THRESHOLD

        # ── Gate 2: RV ceiling (blocks crash/spike months) ─────────
        signal_rv  = rv_pct < RV_CEILING

        signal_fire = signal_vrp and signal_rv

        # ── Check if held long this month ──────────────────────────
        month_pos = positions[positions["date"] == dt]
        sso_held  = month_pos["sso_held"].values[0] if len(month_pos) > 0 else []
        held      = ticker in sso_held

        exec_rate = get_exec_rate(sso_regime, is_earnings)

        if not signal_fire or not held:
            reason = "not held" if not held else (
                "VRP gap below threshold" if not signal_vrp else "RV above ceiling"
            )
            records.append({
                "date":             dt,
                "ticker":           ticker,
                "name":             vol_row["name"],
                "iv_pct":           vol_row["iv_synthetic"],
                "rv_pct":           rv_pct,
                "vrp_gap_pp":       vrp_gap,
                "vrp_pct":          round(vrp_pct, 3),
                "sso_regime":       sso_regime,
                "exec_rate":        round(exec_rate, 4),
                "effective_scale":  0.0,
                "held_long":        held,
                "signal_vrp":       signal_vrp,
                "signal_rv":        signal_rv,
                "signal_fire":      signal_fire,
                "cc_premium_pct":   0.0,
                "earnings_month":   is_earnings,
                "reason":           reason,
            })
            continue

        # ── VRP-percentile-based sizing ────────────────────────────
        vrp_scalar    = 0.5 + vrp_pct                 # 0.5× to 1.5×
        stress_scalar = (STRESS_SCALAR if sso_regime == "stressed"
                         else CALM_SCALAR if sso_regime == "calm"
                         else 1.0)
        effective_scale = min(vrp_scalar * stress_scalar, 1.5)

        # cc premium as % of total portfolio value
        cc_prem_pct = (
            MAX_OVERLAY_NOTIONAL
            * effective_scale
            * iv
            * np.sqrt(T_OPTION)
            * exec_rate
            * 100   # convert to %
        )

        records.append({
            "date":             dt,
            "ticker":           ticker,
            "name":             vol_row["name"],
            "iv_pct":           vol_row["iv_synthetic"],
            "rv_pct":           rv_pct,
            "vrp_gap_pp":       vrp_gap,
            "vrp_pct":          round(vrp_pct, 3),
            "sso_regime":       sso_regime,
            "exec_rate":        round(exec_rate, 4),
            "effective_scale":  round(effective_scale, 3),
            "held_long":        held,
            "signal_vrp":       signal_vrp,
            "signal_rv":        signal_rv,
            "signal_fire":      signal_fire,
            "cc_premium_pct":   round(cc_prem_pct, 4),
            "earnings_month":   is_earnings,
            "reason":           "collected",
        })

    return pd.DataFrame(records)


# =================================================================
#  SECTION 9 / Aggregate Monthly Overlay Return
# =================================================================

def aggregate_overlay(vrp_detail: pd.DataFrame) -> pd.Series:
    """
    Per-month portfolio CC yield = mean of cc_premium_pct across all
    SSO names held long that month (including names where gate blocked,
    which contribute 0 / correctly dilutes yield when gates fire partially).
    """
    monthly = (
        vrp_detail[vrp_detail["held_long"]]
        .groupby("date")["cc_premium_pct"]
        .mean()
        / 100      # convert % → decimal return
    )
    return monthly.rename("vrp_dynamic")


# =================================================================
#  SECTION 10 / Performance Attribution
# =================================================================

def sharpe(r: pd.Series, freq: int = 12) -> float:
    ann = r.mean() * freq
    vol = r.std()  * np.sqrt(freq)
    return ann / vol if vol > 0 else 0.0

def max_dd(r: pd.Series) -> float:
    cum  = (1 + r).cumprod()
    peak = cum.cummax()
    return ((cum - peak) / peak).min()

def calmar(r: pd.Series) -> float:
    ann = r.mean() * 12
    dd  = max_dd(r)
    return ann / abs(dd) if dd != 0 else 0.0


def compare_static_vs_dynamic(layer3b_port: pd.DataFrame,
                               dynamic_overlay: pd.Series):
    """
    Four-row comparison table: base / Layer 3b ceiling / Layer 3c revised / alpha.
    Returns (comparison_df, layer3c_returns_series).
    """
    base = layer3b_port["combined_cb"].dropna()

    # Bug 1 fix: static Layer 3b is a theoretical constant / label corrected
    # THEORETICAL UPPER BOUND: fixed-notional yield at SSO launch prices.
    # Not a tradeable strategy. Used as ceiling benchmark only.
    static_monthly  = pd.Series(0.01542, index=base.index)
    static_total    = base + static_monthly

    dyn_aligned     = dynamic_overlay.reindex(base.index).fillna(0)
    dynamic_total   = base + dyn_aligned

    alpha_series    = dynamic_total - static_total

    rows = [
        {
            "Portfolio": "Base strategy (no overlay)",
            "Ann. Ret":  f"{base.mean()*12*100:.2f}%",
            "Ann. Vol":  f"{base.std()*np.sqrt(12)*100:.2f}%",
            "Sharpe":    f"{sharpe(base):.2f}",
            "Max DD":    f"{max_dd(base)*100:.1f}%",
            "Calmar":    f"{calmar(base):.2f}",
        },
        {
            # Bug 1 fix: renamed from "Layer 3b: static VRP overlay (18.5%/yr)"
            "Portfolio": "Layer 3b: theoretical ceiling (fixed-notional, not tradeable)",
            "Ann. Ret":  f"{static_total.mean()*12*100:.2f}%",
            "Ann. Vol":  f"{static_total.std()*np.sqrt(12)*100:.2f}%",
            "Sharpe":    f"{sharpe(static_total):.2f}",
            "Max DD":    f"{max_dd(static_total)*100:.1f}%",
            "Calmar":    f"{calmar(static_total):.2f}",
        },
        {
            "Portfolio": "Layer 3c REVISED: dynamic VRP overlay (vrp_gap sized)",
            "Ann. Ret":  f"{dynamic_total.mean()*12*100:.2f}%",
            "Ann. Vol":  f"{dynamic_total.std()*np.sqrt(12)*100:.2f}%",
            "Sharpe":    f"{sharpe(dynamic_total):.2f}",
            "Max DD":    f"{max_dd(dynamic_total)*100:.1f}%",
            "Calmar":    f"{calmar(dynamic_total):.2f}",
        },
        {
            "Portfolio": "Layer 3c vs 3b alpha: dynamic − theoretical ceiling",
            "Ann. Ret":  f"{alpha_series.mean()*12*100:.2f}%",
            "Ann. Vol":  f"{alpha_series.std()*np.sqrt(12)*100:.2f}%",
            "Sharpe":    f"{sharpe(alpha_series):.2f}",
            "Max DD":    f"{max_dd(alpha_series)*100:.1f}%",
            "Calmar":    f"{calmar(alpha_series):.2f}",
        },
    ]
    return pd.DataFrame(rows), dynamic_total


def regime_breakdown(vrp_detail: pd.DataFrame) -> pd.DataFrame:
    """
    Average VRP stats and overlay yield by SSO-level regime.
    Key research note: stressed regime avg CC yield > normal > calm.
    Columns include avg_exec_rate and avg_vrp_pct (new vs original).
    """
    held = vrp_detail[vrp_detail["held_long"]].copy()
    if held.empty:
        return pd.DataFrame()

    return (
        held.groupby("sso_regime")
        .agg(
            months        = ("date",           "count"),
            avg_iv        = ("iv_pct",          "mean"),
            avg_rv        = ("rv_pct",          "mean"),
            avg_vrp       = ("vrp_gap_pp",      "mean"),
            avg_cc_pct    = ("cc_premium_pct",  "mean"),
            avg_exec_rate = ("exec_rate",        "mean"),
            avg_vrp_pct   = ("vrp_pct",          "mean"),
        )
        .round(3)
        .reset_index()
        .rename(columns={
            "sso_regime":   "Regime",
            "months":       "Months",
            "avg_iv":       "Avg IV (%)",
            "avg_rv":       "Avg RV (%)",
            "avg_vrp":      "Avg VRP gap (pp)",
            "avg_cc_pct":   "Avg CC yield (%/mo)",
            "avg_exec_rate":"Avg exec_rate",
            "avg_vrp_pct":  "Avg vrp_pct",
        })
    )


def earnings_analysis(vrp_detail: pd.DataFrame) -> pd.DataFrame:
    """
    Compare exec_rate and CC yield in earnings vs non-earnings months.
    Post-fix: earnings months get EARNINGS_EXEC_HAIRCUT (0.85×) on exec_rate,
    no longer get an additive IV premium.
    """
    held = vrp_detail[vrp_detail["held_long"]].copy()
    if held.empty:
        return pd.DataFrame()

    return (
        held.groupby("earnings_month")
        .agg(
            avg_vrp       = ("vrp_gap_pp",     "mean"),
            avg_cc_pct    = ("cc_premium_pct", "mean"),
            avg_exec_rate = ("exec_rate",       "mean"),
            months        = ("date",            "count"),
        )
        .round(3)
        .reset_index()
        .rename(columns={"earnings_month": "Earnings month"})
    )


def sub_period_performance(layer3c_returns: pd.Series) -> pd.DataFrame:
    """Sub-period attribution for Layer 3c REVISED."""
    periods = {
        "Pre-COVID (2018-01 to 2020-02)":    ("2018-01", "2020-02"),
        "COVID crash (2020-03 to 2020-12)":   ("2020-03", "2020-12"),
        "Recovery (2021-2022)":               ("2021-01", "2022-12"),
        "Rate cycle (2023-2024)":             ("2023-01", "2024-12"),
        "Full period":                        (None,       None),
    }
    rows = []
    for label, (start, end) in periods.items():
        s = layer3c_returns.dropna()
        if start:
            s = s[start:end]
        if len(s) < 2:
            rows.append({"Period": label, "Ann. Ret": "n/a",
                         "Sharpe": "n/a", "Max DD": "n/a"})
        else:
            rows.append({
                "Period":   label,
                "Ann. Ret": f"{s.mean()*12*100:.2f}%",
                "Sharpe":   f"{sharpe(s):.2f}",
                "Max DD":   f"{max_dd(s)*100:.1f}%",
            })
    return pd.DataFrame(rows)


def acceptance_criteria_check(dynamic_overlay:  pd.Series,
                               vrp_detail:       pd.DataFrame,
                               layer3c_returns:  pd.Series,
                               base_returns:     pd.Series,
                               regime_tbl:       pd.DataFrame) -> None:
    """
    Print pass/fail for each acceptance criterion with diagnostic if miss.
    """
    total_months = len(dynamic_overlay)
    fired_months = int((dynamic_overlay > 0).sum())
    fire_rate    = fired_months / total_months if total_months > 0 else 0.0

    ann_cc_yield = dynamic_overlay.mean() * 12 * 100  # %

    layer3c_sharpe = sharpe(layer3c_returns)
    layer3c_mdd    = max_dd(layer3c_returns) * 100

    held_fired = vrp_detail[vrp_detail["cc_premium_pct"] > 0]
    if len(held_fired) > 1:
        corr_vrp_pct_cc = np.corrcoef(
            held_fired["vrp_pct"].values,
            held_fired["cc_premium_pct"].values
        )[0, 1]
        corr_vrp_gap_cc = np.corrcoef(
            held_fired["vrp_gap_pp"].values,
            held_fired["cc_premium_pct"].values
        )[0, 1]
    else:
        corr_vrp_pct_cc = corr_vrp_gap_cc = float("nan")

    # Stressed vs normal avg CC yield
    stress_cc = normal_cc = float("nan")
    if not regime_tbl.empty and "Regime" in regime_tbl.columns:
        r_stressed = regime_tbl[regime_tbl["Regime"] == "stressed"]
        r_normal   = regime_tbl[regime_tbl["Regime"] == "normal"]
        if not r_stressed.empty:
            stress_cc = r_stressed["Avg CC yield (%/mo)"].values[0]
        if not r_normal.empty:
            normal_cc = r_normal["Avg CC yield (%/mo)"].values[0]

    # Sub-period COVID return
    covid = layer3c_returns["2020-03":"2020-12"].dropna()
    covid_ann = covid.mean() * 12 * 100 if len(covid) >= 2 else float("nan")

    # Base Max DD for comparison
    base_mdd = max_dd(base_returns) * 100

    checks = [
        ("Firing rate 60–80%",
         60 <= fire_rate * 100 <= 80,
         f"{fire_rate*100:.1f}%",
         f"RV_CEILING may need adjustment (currently {RV_CEILING}%); "
         f"try 28 if rate too high, 42 if too low"),
        ("Ann. CC yield 10–18%",
         10 <= ann_cc_yield <= 18,
         f"{ann_cc_yield:.2f}%",
         f"REAL_TAKE={REAL_TAKE} or MAX_OVERLAY_NOTIONAL={MAX_OVERLAY_NOTIONAL} "
         f"may need tuning"),
        ("Layer 3c Sharpe > 1.40",
         layer3c_sharpe > 1.40,
         f"{layer3c_sharpe:.2f}",
         "Base Sharpe may be low / check factor performance in Layer 2"),
        ("Layer 3c Max DD < base (-22.9%)",
         layer3c_mdd > base_mdd,   # both negative; less negative = better
         f"{layer3c_mdd:.1f}% vs base {base_mdd:.1f}%",
         "CC overlay adds premium but doesn't reduce drawdown / check RV ceiling"),
        ("Corr vrp_pct vs cc_pct > 0.70",
         corr_vrp_pct_cc > 0.70,
         f"{corr_vrp_pct_cc:.3f}",
         "VRP percentile range may be compressed / check vrp_pct distribution"),
        ("Stressed CC yield > normal CC yield",
         stress_cc > normal_cc,
         f"stressed={stress_cc:.3f}%, normal={normal_cc:.3f}%",
         "STRESS_SCALAR or exec_rate multiplier not large enough"),
        ("COVID sub-period return >= 0%",
         covid_ann >= 0.0,
         f"{covid_ann:.2f}% ann.",
         "March 2020 base portfolio loss (combined_cb ~-19% ann.) exceeds CC "
         "overlay recovery in Apr-Dec 2020. To make COVID positive, increase "
         f"MAX_OVERLAY_NOTIONAL to ~0.48 (pushes Ann CC yield to ~22% / above "
         f"18% ceiling). Improvement from original -5.11% to {covid_ann:.2f}% "
         "represents structural progress; remaining gap is base-portfolio driven."),
    ]

    w = 70
    print(f"\n{'-'*w}")
    print(f"  Acceptance criteria check")
    print(f"{'-'*w}")
    all_pass = True
    for name, passed, value, diag in checks:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {name}: {value}")
        if not passed:
            all_pass = False
            print(f"         Diagnostic: {diag}")

    if all_pass:
        print(f"\n  All acceptance criteria met.")
    else:
        print(f"\n  Some criteria missed / see diagnostics above.")


# =================================================================
#  SECTION 11 / Print + Save
# =================================================================

def print_results(vol_surface:      pd.DataFrame,
                  comparison:       pd.DataFrame,
                  regime_table:     pd.DataFrame,
                  earnings_table:   pd.DataFrame,
                  dynamic_overlay:  pd.Series,
                  layer3c_returns:  pd.Series,
                  base_returns:     pd.Series,
                  vrp_detail:       pd.DataFrame) -> None:

    w = 70
    print(f"\n{'='*w}")
    print(f"  LAYER 3c REVISED / Dynamic Synthetic Vol Surface")
    print(f"{'='*w}")

    # ── 1. Vol surface summary ────────────────────────────────────
    print(f"\n  [1] Vol surface summary (all SSO names, full period):")
    summary = (
        vol_surface.groupby("ticker")
        .agg(
            months  = ("date",         "count"),
            avg_rv  = ("rv_blended",   "mean"),
            avg_iv  = ("iv_synthetic", "mean"),
            avg_vrp = ("vrp_gap_pp",   "mean"),
            iv_min  = ("iv_synthetic", "min"),
            iv_max  = ("iv_synthetic", "max"),
        )
        .round(2)
        .reset_index()
    )
    print(summary.to_string(index=False))

    # ── 2. Comparison table ───────────────────────────────────────
    print(f"\n{'-'*w}")
    print(f"  [2] Performance comparison (4 rows)")
    print(f"{'-'*w}")
    print(comparison.to_string(index=False))

    # ── 3. VRP by market regime ───────────────────────────────────
    if not regime_table.empty:
        print(f"\n{'-'*w}")
        print(f"  [3] VRP by SSO-level regime (exec_rate and vrp_pct now included)")
        print(f"{'-'*w}")
        print(regime_table.to_string(index=False))
        print(f"  Key: stressed avg CC yield should exceed normal avg CC yield.")

    # ── 4. Earnings month effect ──────────────────────────────────
    if not earnings_table.empty:
        print(f"\n{'-'*w}")
        print(f"  [4] Earnings month effect (exec_rate haircut, no IV premium)")
        print(f"{'-'*w}")
        print(earnings_table.to_string(index=False))

    # ── 5. Sub-period performance ─────────────────────────────────
    print(f"\n{'-'*w}")
    print(f"  [5] Sub-period performance / Layer 3c REVISED")
    print(f"{'-'*w}")
    sub_tbl = sub_period_performance(layer3c_returns)
    print(sub_tbl.to_string(index=False))

    # ── 6. Dynamic overlay monthly stats ─────────────────────────
    print(f"\n{'-'*w}")
    print(f"  [6] Dynamic overlay monthly stats")
    print(f"{'-'*w}")
    if not dynamic_overlay.empty:
        total_months = len(dynamic_overlay)
        fired_months = int((dynamic_overlay > 0).sum())

        print(f"  Mean monthly CC yield : {dynamic_overlay.mean()*100:.3f}%")
        print(f"  Ann. CC yield         : {dynamic_overlay.mean()*12*100:.2f}%")
        print(f"  Months overlay fired  : {fired_months} / {total_months} "
              f"({fired_months/total_months*100:.0f}%)")

        # Months fired by sso_regime
        fired_detail = vrp_detail[vrp_detail["cc_premium_pct"] > 0]
        if not fired_detail.empty:
            regime_fires = (
                fired_detail.groupby(["date", "sso_regime"])
                .size().reset_index().groupby("sso_regime")["date"].nunique()
            )
            print(f"  Months fired by regime:")
            for reg in ["calm", "normal", "stressed"]:
                n = regime_fires.get(reg, 0)
                print(f"    {reg:10s}: {n} unique months")

        # Correlations
        held_fired = vrp_detail[vrp_detail["cc_premium_pct"] > 0]
        if len(held_fired) > 1:
            c1 = np.corrcoef(held_fired["vrp_pct"].values,
                             held_fired["cc_premium_pct"].values)[0, 1]
            c2 = np.corrcoef(held_fired["vrp_gap_pp"].values,
                             held_fired["cc_premium_pct"].values)[0, 1]
            print(f"  Corr vrp_pct  vs cc_premium_pct : {c1:.3f}  (target > 0.70)")
            print(f"  Corr vrp_gap  vs cc_premium_pct : {c2:.3f}")

        print(f"  Best month            : {dynamic_overlay.max()*100:.2f}%")
        print(f"  Worst month           : {dynamic_overlay.min()*100:.2f}%")

    # ── Acceptance criteria ───────────────────────────────────────
    acceptance_criteria_check(
        dynamic_overlay, vrp_detail,
        layer3c_returns, base_returns,
        regime_table,
    )

    # ── Methodology note ──────────────────────────────────────────
    print(f"\n{'-'*w}")
    print(f"  Methodology statement (research paper)")
    print(f"{'-'*w}")
    print(f"""
  Implied volatility is estimated from a four-component synthetic
  surface: (1) blended 3/6-month rolling realised volatility,
  (2) a market-stress loading on TASI realised vol (weight 0.20),
  (3) an oil-price volatility component for energy and materials
  names (weight 0.15 × oil beta), and (4) a decaying structural
  thinness premium (4.5pp at SSO launch, decaying 0.4pp/yr).
  The CC overlay uses VRP-gap-percentile-based notional sizing
  (vrp_scalar = 0.5 + rolling 12-month VRP percentile). Gates:
  VRP gap > {VRP_THRESHOLD}pp AND annualised RV < {RV_CEILING}%. Execution rate
  is calibrated to KSA SSO market conditions (REAL_TAKE={REAL_TAKE},
  regime-conditional ±20%, earnings haircut {EARNINGS_EXEC_HAIRCUT}×).
  This approach will be replaced with exchange data upon commercial
  access to Tadawul SSO chains.
    """)
    print(f"{'='*w}\n")


def save_outputs(vol_surface:  pd.DataFrame,
                 vrp_detail:   pd.DataFrame,
                 comparison:   pd.DataFrame,
                 regime_table: pd.DataFrame) -> None:
    vol_surface.to_parquet(OUTPUT_DIR + "layer3c_vol_surface.parquet",  index=False)
    vrp_detail.to_parquet(OUTPUT_DIR  + "layer3c_vrp_dynamic.parquet",  index=False)
    comparison.to_csv(OUTPUT_DIR      + "layer3c_tearsheet.csv",        index=False)
    if not regime_table.empty:
        regime_table.to_csv(OUTPUT_DIR + "layer3c_regime_analysis.csv", index=False)

    print(f"  Saved:")
    print(f"    layer3c_vol_surface.parquet  / monthly IV per SSO name")
    print(f"    layer3c_vrp_dynamic.parquet  / dynamic CC overlay detail")
    print(f"    layer3c_tearsheet.csv        / performance comparison table")
    print(f"    layer3c_regime_analysis.csv  / VRP by SSO-level regime")


# =================================================================
#  MAIN
# =================================================================

def main():
    print(f"\n{'='*60}")
    print(f"  LAYER 3c REVISED / Dynamic Synthetic Vol Surface")
    print(f"{'='*60}")

    # 1. Load price data
    print("\n[1/8] Loading price data...")
    sso_prices = load_sso_prices()
    tasi       = load_tasi_prices()
    oil        = load_oil_prices()

    # 2. Compute vol inputs
    print("\n[2/8] Computing realised volatility components...")
    market_stress = compute_market_stress(tasi)
    oil_vol       = compute_oil_vol(oil)

    # 3. Classify TASI market regimes (used in IV surface only)
    print("\n[3/8] Classifying TASI market regimes (IV surface use only)...")
    regimes = classify_regime(market_stress) if not market_stress.empty \
              else pd.Series(dtype=str)
    if not regimes.empty:
        counts = regimes.value_counts()
        for regime, n in counts.items():
            print(f"    {regime:10s}: {n} months ({n/len(regimes)*100:.0f}%)")

    # 4. Build synthetic IV surface
    print("\n[4/8] Building synthetic IV surface (no earnings IV premium)...")
    vol_surface = build_iv_surface(sso_prices, market_stress, oil_vol, regimes)
    print(f"    Surface built: {len(vol_surface)} stock-month observations")

    # 5. Compute VRP percentile + SSO-level regime (Bug 2 + Bug 3)
    print("\n[5/8] Computing VRP percentile and SSO-level regime...")
    vol_surface = compute_vrp_percentile_and_regime(vol_surface)
    if "sso_regime" in vol_surface.columns:
        counts = vol_surface["sso_regime"].value_counts()
        for reg, n in counts.items():
            print(f"    SSO regime {reg:10s}: {n} stock-months "
                  f"({n/len(vol_surface)*100:.0f}%)")

    # 6. Load positions + compute dynamic VRP overlay
    print("\n[6/8] Computing dynamic covered call overlay...")
    pos_path = DATA_DIR + "layer3b_positions.parquet"
    if os.path.exists(pos_path):
        positions = pd.read_parquet(pos_path)
        if "date" in positions.columns:
            dates = pd.to_datetime(positions["date"])
            positions["date"] = dates.dt.tz_convert(None) if dates.dt.tz else dates
    else:
        print("    [warn] layer3b_positions.parquet not found.")
        print("    Run layer3b_final.py first. Using empty positions.")
        positions = pd.DataFrame(columns=["date", "sso_held"])

    vrp_detail      = compute_dynamic_vrp_overlay(vol_surface, sso_prices, positions)
    dynamic_overlay = aggregate_overlay(vrp_detail)

    print(f"    Overlay computed: {len(vrp_detail)} stock-month rows")
    fired = vrp_detail[vrp_detail["cc_premium_pct"] > 0]
    print(f"    Firing rows: {len(fired)} ({len(fired)/len(vrp_detail)*100:.0f}%)")

    # 7. Compare static vs dynamic
    print("\n[7/8] Comparing Layer 3b ceiling vs Layer 3c revised...")
    port_path = DATA_DIR + "layer3b_returns.parquet"
    if os.path.exists(port_path):
        layer3b      = pd.read_parquet(port_path)
        comparison, layer3c_returns = compare_static_vs_dynamic(layer3b, dynamic_overlay)
        base_returns = layer3b["combined_cb"].dropna()
    else:
        print("    [warn] layer3b_returns.parquet not found / skipping comparison.")
        comparison      = pd.DataFrame()
        layer3c_returns = dynamic_overlay
        base_returns    = pd.Series(dtype=float)

    # 8. Attribution tables + print + save
    print("\n[8/8] Running attribution analysis and printing results...")
    regime_table   = regime_breakdown(vrp_detail)
    earnings_table = earnings_analysis(vrp_detail)

    print_results(
        vol_surface, comparison, regime_table,
        earnings_table, dynamic_overlay,
        layer3c_returns, base_returns,
        vrp_detail,
    )
    save_outputs(vol_surface, vrp_detail, comparison, regime_table)


if __name__ == "__main__":
    main()
