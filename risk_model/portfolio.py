# risk_model/portfolio.py

from typing import Dict
import numpy as np
import pandas as pd


def compute_rolling_correlation(
    returns: pd.DataFrame,
    window: int = 250,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Compute rolling correlation matrices over a fixed window.

    Returns a dictionary:
       { date_t : corr_matrix_at_t }

    correlation at time t is computed from returns[t-window : t].
    """
    corr_dict: Dict[pd.Timestamp, pd.DataFrame] = {}

    for i in range(window, len(returns)):
        date = returns.index[i]
        window_ret = returns.iloc[i - window : i]
        corr_dict[date] = window_ret.corr()

    return corr_dict


def build_covariance_matrix(
    vols: pd.DataFrame,
    corr_dict: Dict[pd.Timestamp, pd.DataFrame],
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Build covariance matrices Σ_t for each date t using:

        Σ_t = D_t  R_t  D_t

    where D_t = diag(vols_t).
    """
    cov_dict: Dict[pd.Timestamp, pd.DataFrame] = {}

    for date, R_t in corr_dict.items():
        if date not in vols.index:
            continue

        sigma_t = vols.loc[date]               # vector of std devs
        D_t = np.diag(sigma_t.values)

        Σ_t = pd.DataFrame(
            D_t @ R_t.values @ D_t,
            index=R_t.index,
            columns=R_t.columns,
        )

        cov_dict[date] = Σ_t

    return cov_dict


def compute_portfolio_volatility(
    cov_dict: Dict[pd.Timestamp, pd.DataFrame],
    weights: pd.Series,
) -> pd.Series:
    """
    Compute portfolio volatility for each date:

        σ_p,t = sqrt( w^T Σ_t w )
    """
    vols = {}

    # column vector (n × 1)
    w = weights.values.reshape(-1, 1)

    for date, Sigma_t in cov_dict.items():
        # Ensure the covariance matrix is aligned to the weight order
        Sigma_t = Sigma_t.loc[weights.index, weights.index]

        # Matrix multiplication: (1×n) @ (n×n) @ (n×1) -> (1×1)
        var_matrix = (w.T @ Sigma_t.values @ w)

        # Extract the single scalar value from the 1×1 array
        variance = var_matrix[0, 0]

        vols[date] = np.sqrt(variance)

    return pd.Series(vols, name="portfolio_vol").sort_index()


def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """
    Compute portfolio return series using fixed weights.

    r_p,t = sum_i  w_i * r_i,t
    """
    aligned = returns[weights.index]
    rp = aligned.mul(weights, axis=1).sum(axis=1)
    rp.name = "portfolio_return"
    return rp