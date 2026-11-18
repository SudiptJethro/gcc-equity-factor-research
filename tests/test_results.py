"""Checks that the headline numbers in the README follow from the shipped output files,
that the Newey-West estimator behaves as expected, and that the dashboard builds."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Layer_3_analysis"))


def ann_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return (r.mean() * 12) / (r.std() * np.sqrt(12))


# ---------- headline results ----------
@pytest.fixture(scope="module")
def quintiles():
    return pd.read_csv(DATA / "portfolio_returns.csv", index_col=0, parse_dates=True)


def test_quality_long_short_sharpe(quintiles):
    assert ann_sharpe(quintiles["quality_LS"]) == pytest.approx(1.03, abs=0.01)


def test_momentum_long_short_sharpe(quintiles):
    assert ann_sharpe(quintiles["mom_LS"]) == pytest.approx(0.56, abs=0.01)


def test_value_premium_is_negative(quintiles):
    assert ann_sharpe(quintiles["value_LS"]) == pytest.approx(-0.56, abs=0.01)


def test_combined_portfolio_sharpe_net_of_costs():
    port = pd.read_parquet(DATA / "layer3b_returns.parquet")
    assert ann_sharpe(port["combined_cb"]) == pytest.approx(0.49, abs=0.01)


def test_summary_table_matches_return_series(quintiles):
    summary = pd.read_csv(DATA / "factors" / "backtest_summary.csv")
    gcc = summary[summary.exchange == "GCC"].set_index("factor")["LS_sharpe"]
    assert gcc["QUALITY"] == pytest.approx(ann_sharpe(quintiles["quality_LS"]), abs=0.005)
    assert gcc["MOMENTUM"] == pytest.approx(ann_sharpe(quintiles["mom_LS"]), abs=0.005)


# ---------- Newey-West t-statistic ----------
@pytest.fixture(scope="module")
def fm():
    return importlib.import_module("fama_macbeth_3a")


def test_nw_with_zero_lags_matches_plain_t_stat(fm):
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0.01, 0.05, 200))
    _, t = fm.nw_tstat(x, n_lags=0)
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert t == pytest.approx(plain, rel=1e-9)


def test_nw_shrinks_t_stat_for_autocorrelated_series(fm):
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, 400)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):  # AR(1) with strong positive autocorrelation
        x[i] = 0.7 * x[i - 1] + e[i]
    x = pd.Series(x + 0.2)
    _, t0 = fm.nw_tstat(x, n_lags=0)
    _, t4 = fm.nw_tstat(x, n_lags=4)
    assert abs(t4) < abs(t0)


def test_nw_returns_nan_for_short_series(fm):
    mu, t = fm.nw_tstat(pd.Series([0.01] * 5))
    assert np.isnan(mu) and np.isnan(t)


# ---------- dashboard ----------
@pytest.mark.parametrize("page", ["factor_model", "portfolio", "backtesting", "summary"])
def test_dashboard_page_builds(page):
    module = importlib.import_module(f"pages.{page}")
    assert module.layout() is not None
