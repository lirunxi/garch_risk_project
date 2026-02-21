# risk_model/eda.py

from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def summarize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Extended summary statistics for each asset.

    Includes mean, std, skew, kurtosis, min, max, and selected quantiles.
    """
    desc = returns.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    desc["skew"] = returns.skew()
    desc["kurtosis"] = returns.kurtosis()
    return desc


def plot_volatility_clustering(
    returns: pd.Series,
    asset_name: str = "",
    window: int = 20,
) -> None:
    """
    Visual check for volatility clustering:
      - plot returns
      - plot rolling std

    Parameters
    ----------
    returns : pd.Series
        Return series for a single asset.
    asset_name : str
        Optional name for titles.
    window : int
        Rolling window length for realized volatility.
    """
    r = returns.dropna()
    rv = r.rolling(window=window).std()

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax[0].plot(r.index, r.values)
    ax[0].set_title(f"{asset_name} returns")

    ax[1].plot(rv.index, rv.values)
    ax[1].set_title(f"{asset_name} {window}-day rolling volatility")
    ax[1].set_xlabel("Date")

    plt.tight_layout()
    plt.show()


def adf_test(
    returns: pd.Series,
    maxlag: int | None = None,
) -> Dict[str, float]:
    """
    Augmented Dickey-Fuller test on a return series.

    Null hypothesis: series has a unit root (non-stationary).
    For returns, we usually expect to REJECT the null (i.e., stationary).

    Returns a dict with statistic, pvalue, and some extras.
    """
    r = returns.dropna()
    result = adfuller(r, maxlag=maxlag, autolag="AIC")
    stat, pvalue, usedlag, nobs, crit_vals, icbest = result

    out = {
        "statistic": stat,
        "pvalue": pvalue,
        "used_lag": usedlag,
        "nobs": nobs,
        "crit_1%": crit_vals["1%"],
        "crit_5%": crit_vals["5%"],
        "crit_10%": crit_vals["10%"],
        "icbest": icbest,
    }
    return out


def plot_acf_pacf(
    returns: pd.Series,
    lags: int = 20,
    title_prefix: str = "",
) -> None:
    """
    Plot ACF and PACF of a return series.

    Useful for checking serial correlation structure.
    """
    r = returns.dropna()

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(r, lags=lags, ax=ax[0])
    ax[0].set_title(f"{title_prefix} ACF of returns")

    plot_pacf(r, lags=lags, ax=ax[1], method="ywm")
    ax[1].set_title(f"{title_prefix} PACF of returns")

    plt.tight_layout()
    plt.show()


def plot_acf_pacf_squared(
    returns: pd.Series,
    lags: int = 20,
    title_prefix: str = "",
) -> None:
    """
    Plot ACF and PACF of squared returns.

    This is standard for checking volatility clustering / ARCH effects.
    """
    r2 = (returns.dropna() ** 2)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(r2, lags=lags, ax=ax[0])
    ax[0].set_title(f"{title_prefix} ACF of squared returns")

    plot_pacf(r2, lags=lags, ax=ax[1], method="ywm")
    ax[1].set_title(f"{title_prefix} PACF of squared returns")

    plt.tight_layout()
    plt.show()