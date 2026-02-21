# risk_model/var_es.py

from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm


def parametric_var_es_normal(
    portfolio_vol: pd.Series,
    alpha: float = 0.99,
    mean_return: float = 0.0,
) -> pd.DataFrame:
    """
    Parametric 1-day VaR and ES for portfolio loss under normality.

    Assumptions
    -----------
    R_{p,t+1} ~ Normal(mean_return, portfolio_vol_t^2)
    Loss L_{t+1} = -R_{p,t+1}

    For confidence level alpha (e.g., 0.99):
      z = Phi^{-1}(1 - alpha)  (lower-tail quantile, negative)
      VaR_alpha = - (mean_return + sigma * z)
      ES_alpha  = sigma * phi(z) / (1 - alpha) - mean_return

    Parameters
    ----------
    portfolio_vol : pd.Series
        Time series of conditional portfolio volatility sigma_{p,t}.
    alpha : float
        Confidence level, e.g. 0.99 for 99% VaR.
    mean_return : float
        Assumed (or estimated) unconditional mean portfolio return.

    Returns
    -------
    var_es : pd.DataFrame
        DataFrame indexed like portfolio_vol with columns ["VaR", "ES"],
        representing *positive* loss numbers.
    """
    sigma = portfolio_vol

    # lower-tail quantile (e.g. 0.01 for alpha=0.99)
    z = norm.ppf(1 - alpha)
    phi_z = norm.pdf(z)

    var = -(mean_return + sigma * z)
    es = sigma * phi_z / (1 - alpha) - mean_return

    out = pd.DataFrame({"VaR": var, "ES": es})
    return out


def historical_var_es(
    portfolio_returns: pd.Series,
    window: int = 250,
    alpha: float = 0.99,
) -> pd.DataFrame:
    """
    Rolling Historical VaR and ES for portfolio loss.

    At each date t >= window, use the past 'window' returns:
      - compute empirical (1-alpha)-quantile q of returns (lower tail)
      - VaR = -q
      - ES  = -average of returns <= q

    Parameters
    ----------
    portfolio_returns : pd.Series
        Time series of portfolio returns r_{p,t}.
    window : int
        Size of the rolling lookback window (e.g., 250 trading days).
    alpha : float
        Confidence level.

    Returns
    -------
    var_es : pd.DataFrame
        Indexed by date (starting after 'window') with columns ["VaR", "ES"].
    """
    vars_ = []
    ess_ = []
    dates = []

    r = portfolio_returns.dropna()

    for i in range(window, len(r)):
        window_ret = r.iloc[i - window : i]
        # Lower-tail quantile for returns
        q = np.quantile(window_ret, 1 - alpha)
        tail = window_ret[window_ret <= q]

        var_val = -q
        es_val = -tail.mean() if len(tail) > 0 else -q

        vars_.append(var_val)
        ess_.append(es_val)
        dates.append(r.index[i])

    df = pd.DataFrame({"VaR": vars_, "ES": ess_}, index=pd.Index(dates))
    return df