import numpy as np
import pandas as pd

from config import KUWAIT_MSCI_UPGRADE


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly log returns from a price DataFrame."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def kuwait_split(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split Kuwait prices into pre- and post-MSCI-upgrade subsamples.
    Returns (pre_df, post_df).
    Hypothesis: factor premia compress post-upgrade as institutional
    arbitrage capital enters the market.
    """
    split = pd.Timestamp(KUWAIT_MSCI_UPGRADE)
    pre  = monthly[monthly.index < split]
    post = monthly[monthly.index >= split]
    return pre, post
