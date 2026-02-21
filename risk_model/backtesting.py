# risk_model/backtesting.py

from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import chi2


def compute_exceedances(
    portfolio_returns: pd.Series,
    var_series: pd.Series,
) -> pd.Series:
    """
    Compute exceedance indicators I_t for VaR backtesting.

    We define loss L_t = -r_t. A VaR exception occurs when:
        L_t > VaR_t   <=>   r_t < -VaR_t

    Parameters
    ----------
    portfolio_returns : pd.Series
        Series of realized portfolio returns r_t.
    var_series : pd.Series
        Series of VaR_t values (positive numbers, in same units as returns).

    Returns
    -------
    exceedances : pd.Series
        Series of 0/1 indicators with aligned index.
    """
    df = pd.concat(
        [portfolio_returns.rename("ret"), var_series.rename("VaR")],
        axis=1,
    ).dropna()

    r = df["ret"]
    var = df["VaR"]

    exceed = (r < -var).astype(int)
    exceed.name = "exceed"

    return exceed

def kupiec_test(
    exceedances: pd.Series,
    alpha: float,
) -> Dict[str, float]:
    """
    Kupiec (1995) Unconditional Coverage Test.

    Null hypothesis: the true exceedance probability is (1 - alpha).
    For example, alpha=0.99 -> expected 1% of days should be exceptions.

    Test statistic:
        LR_uc = -2 [ ln L(pi_0) - ln L(pi_hat) ] ~ chi2(1)

    Returns
    -------
    result : dict
        Contains n, x, pi_hat, lr_uc, p_value.
    """
    I = exceedances.dropna().values
    n = len(I)
    x = int(I.sum())
    pi_hat = x / n if n > 0 else np.nan
    pi_0 = 1.0 - alpha  # expected exception rate

    def log_likelihood(pi: float) -> float:
        if pi <= 0 or pi >= 1:
            return -np.inf
        return x * np.log(pi) + (n - x) * np.log(1 - pi)

    ll_hat = log_likelihood(pi_hat)
    ll_0 = log_likelihood(pi_0)

    lr_uc = -2.0 * (ll_0 - ll_hat)
    p_value = 1.0 - chi2.cdf(lr_uc, df=1)

    return {
        "n": n,
        "x": x,
        "pi_hat": pi_hat,
        "lr_uc": lr_uc,
        "p_value": p_value,
    }

def christoffersen_independence_test(
    exceedances: pd.Series,
) -> Dict[str, float]:
    """
    Christoffersen (1998) independence test.

    Null: exceptions are independent over time (no clustering).
    We build a 2x2 transition matrix for I_{t-1} -> I_t:

        N00: 0 -> 0
        N01: 0 -> 1
        N10: 1 -> 0
        N11: 1 -> 1

    Then compare:
      - Restricted model: single exception probability pi
      - Unrestricted: transition-specific probabilities pi01, pi11

    LR_ind ~ chi2(1)
    """
    I = exceedances.dropna().astype(int).values
    if len(I) < 2:
        return {
            "N00": np.nan,
            "N01": np.nan,
            "N10": np.nan,
            "N11": np.nan,
            "lr_ind": np.nan,
            "p_value": np.nan,
        }

    N00 = N01 = N10 = N11 = 0

    for t in range(1, len(I)):
        prev = I[t - 1]
        curr = I[t]
        if prev == 0 and curr == 0:
            N00 += 1
        elif prev == 0 and curr == 1:
            N01 += 1
        elif prev == 1 and curr == 0:
            N10 += 1
        elif prev == 1 and curr == 1:
            N11 += 1

    N0 = N00 + N01
    N1 = N10 + N11
    N = N0 + N1

    # Transition probabilities under unrestricted model
    pi01 = N01 / N0 if N0 > 0 else 0.0
    pi11 = N11 / N1 if N1 > 0 else 0.0

    # Overall exception probability under restricted model
    pi = (N01 + N11) / N if N > 0 else 0.0

    # Log-likelihood unrestricted
    def safe_log(p):
        return np.log(p) if p > 0 else -np.inf

    ll_unrestricted = (
        N00 * safe_log(1 - pi01)
        + N01 * safe_log(pi01)
        + N10 * safe_log(1 - pi11)
        + N11 * safe_log(pi11)
    )

    # Log-likelihood restricted
    ll_restricted = (
        (N00 + N10) * safe_log(1 - pi)
        + (N01 + N11) * safe_log(pi)
    )

    lr_ind = -2.0 * (ll_restricted - ll_unrestricted)
    p_value = 1.0 - chi2.cdf(lr_ind, df=1)

    return {
        "N00": N00,
        "N01": N01,
        "N10": N10,
        "N11": N11,
        "lr_ind": lr_ind,
        "p_value": p_value,
    }

def christoffersen_conditional_coverage_test(
    exceedances: pd.Series,
    alpha: float,
) -> Dict[str, float]:
    """
    Christoffersen Conditional Coverage Test.

    LR_cc = LR_uc + LR_ind  ~ chi2(2)

    Combines:
      - Unconditional coverage (correct exception rate)
      - Independence (no clustering)
    """
    uc = kupiec_test(exceedances, alpha=alpha)
    ind = christoffersen_independence_test(exceedances)

    lr_cc = uc["lr_uc"] + ind["lr_ind"]
    p_value = 1.0 - chi2.cdf(lr_cc, df=2)

    return {
        "lr_cc": lr_cc,
        "p_value": p_value,
        "lr_uc": uc["lr_uc"],
        "uc_p_value": uc["p_value"],
        "lr_ind": ind["lr_ind"],
        "ind_p_value": ind["p_value"],
        "n": uc["n"],
        "x": uc["x"],
        "pi_hat": uc["pi_hat"],
    }

def summarize_exceedances_by_regime(
    exceedances: pd.Series,
    regimes: pd.Series,
    alpha: float,
) -> pd.DataFrame:
    """
    Summarize VaR exceedances by volatility regime.

    For each regime:
      - n: number of observations
      - x: number of exceedances
      - pi_hat: empirical exception rate
      - expected: 1 - alpha
    """
    df = pd.concat(
        [exceedances.rename("exceed"), regimes.rename("regime")],
        axis=1,
    ).dropna()

    records = []
    for regime, group in df.groupby("regime"):
        I = group["exceed"]
        n = len(I)
        x = int(I.sum())
        pi_hat = x / n if n > 0 else np.nan
        records.append(
            {
                "regime": regime,
                "n": n,
                "x": x,
                "pi_hat": pi_hat,
                "expected": 1.0 - alpha,
            }
        )

    out = pd.DataFrame(records).set_index("regime")
    return out