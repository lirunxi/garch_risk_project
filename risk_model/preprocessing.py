# risk_model/preprocessing.py

from typing import Tuple
import numpy as np
import pandas as pd


def align_and_clean_prices(
    prices: pd.DataFrame,
    drop_if_any_nan: bool = True,
) -> pd.DataFrame:
    """
    Align and clean price data.

    Basic rules:
    - Ensure index is sorted.
    - Optionally drop rows where ANY ticker is NaN
      (strict, but clean).
    - Alternatively, only drop rows where ALL are NaN
      and keep partial (set drop_if_any_nan=False).

    Parameters
    ----------
    prices : pd.DataFrame
        Raw prices.
    drop_if_any_nan : bool
        If True, drop rows with any NaN.
        If False, only drop rows where all are NaN.

    Returns
    -------
    cleaned : pd.DataFrame
        Cleaned price data.
    """
    prices = prices.sort_index()

    if drop_if_any_nan:
        # strict: keep only days where we have all assets
        cleaned = prices.dropna(how="any")
    else:
        # looser: only drop days with all missing
        cleaned = prices.dropna(how="all")

    return cleaned


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from price data.

    r_t = ln(P_t) - ln(P_{t-1})

    Parameters
    ----------
    prices : pd.DataFrame
        Clean, aligned price data.

    Returns
    -------
    returns : pd.DataFrame
        Log returns aligned with prices (first row removed).
    """
    log_prices = np.log(prices)
    returns = log_prices.diff()
    returns = returns.dropna(how="all")
    return returns


def basic_return_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic descriptive statistics for each asset.

    Returns
    -------
    stats_df : pd.DataFrame
        DataFrame with mean, std, skew, kurt for each column.
    """
    stats = {
        "mean": returns.mean(),
        "std": returns.std(),
        "skew": returns.skew(),
        "kurtosis": returns.kurtosis(),
    }
    stats_df = pd.DataFrame(stats)
    return stats_df

def summarize_missing(prices: pd.DataFrame) -> pd.Series:
    """
    Count missing values per asset.
    """
    return prices.isna().sum()