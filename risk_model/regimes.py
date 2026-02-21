# risk_model/regimes.py

from typing import Tuple
import numpy as np
import pandas as pd


def rolling_realized_vol(
    returns: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Compute rolling realized volatility (e.g., 20-day rolling std) of portfolio returns.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns r_{p,t}.
    window : int
        Rolling window length.

    Returns
    -------
    realized_vol : pd.Series
        Rolling standard deviation series.
    """
    realized_vol = returns.rolling(window=window).std()
    realized_vol.name = "realized_vol"
    return realized_vol


def define_volatility_regimes(
    realized_vol: pd.Series,
    low_quantile: float = 0.2,
    high_quantile: float = 0.8,
) -> pd.Series:
    """
    Label each date as 'low', 'mid', or 'high' volatility regime based on quantiles.

    Parameters
    ----------
    realized_vol : pd.Series
        Realized vol measure (e.g., rolling std of portfolio returns).
    low_quantile : float
        Quantile threshold for 'low' regime (e.g., 0.2).
    high_quantile : float
        Quantile threshold for 'high' regime (e.g., 0.8).

    Returns
    -------
    regimes : pd.Series
        Series of strings in {'low', 'mid', 'high'}.
    """
    rv = realized_vol.dropna()
    q_low = rv.quantile(low_quantile)
    q_high = rv.quantile(high_quantile)

    def label(v: float) -> str:
        if v <= q_low:
            return "low"
        elif v >= q_high:
            return "high"
        else:
            return "mid"

    regimes = rv.apply(label)
    regimes.name = "regime"
    return regimes