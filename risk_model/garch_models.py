# risk_model/garch_models.py

from typing import Dict
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox


class UnivariateGarchResult:
    """
    Simple container for fitted GARCH results for one asset.

    Attributes
    ----------
    asset : str
        Ticker / asset name.
    model : arch.univariate.base.ARCHModel
        The underlying GARCH model specification.
    fit_result : arch.univariate.base.ARCHModelResult
        The fitted model result with parameters, diagnostics, etc.
    """

    def __init__(self, asset: str, model, fit_result):
        self.asset = asset
        self.model = model
        self.fit_result = fit_result

    def conditional_vol_series(self) -> pd.Series:
        """
        Full in-sample conditional volatility series sigma_t.

        NOTE: We fitted the model on returns * 100 (percent units),
        so conditional_volatility is also in "percent" units.
        We'll divide by 100 later when building the volatility matrix.
        """
        sigma = self.fit_result.conditional_volatility
        sigma.name = self.asset
        return sigma

    def standardized_residuals(self) -> pd.Series:
        """
        Standardized residuals = epsilon_t / sigma_t.

        Useful for diagnostic tests (autocorrelation, remaining ARCH effects).
        """
        std_resid = self.fit_result.std_resid
        std_resid.name = self.asset + "_std_resid"
        return std_resid

    def summary(self) -> str:
        """
        Text summary of the fitted model (parameters, etc.).
        """
        return self.fit_result.summary().as_text()


def fit_garch_for_asset(
    returns: pd.Series,
    dist: str = "t",
) -> UnivariateGarchResult:
    """
    Fit a GARCH(1,1) model with specified error distribution to a single asset.

    Parameters
    ----------
    returns : pd.Series
        Return series for one asset (log returns).
    dist : str
        Distribution for errors: "normal", "t", "skewt", etc.

    Returns
    -------
    result : UnivariateGarchResult
        Wrapper around the fitted model.
    """
    r = returns.dropna()

    am = arch_model(
        r * 100,          # scale to percentage returns
        mean="zero",
        vol="GARCH",
        p=1,
        q=1,
        dist=dist,
    )

    res = am.fit(disp="off")
    return UnivariateGarchResult(asset=returns.name, model=am, fit_result=res)


def fit_garch_for_all_assets(
    returns: pd.DataFrame,
    dist: str = "t",
) -> Dict[str, UnivariateGarchResult]:
    """
    Fit GARCH(1,1) to all assets in the returns DataFrame.
    """
    models: Dict[str, UnivariateGarchResult] = {}

    for col in returns.columns:
        print(f"Fitting GARCH(1,1) for {col} with dist='{dist}'...")
        models[col] = fit_garch_for_asset(returns[col], dist=dist)

    return models


def build_volatility_matrix(
    models: Dict[str, UnivariateGarchResult],
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a DataFrame of conditional volatilities for all assets over time.

    Each column = asset, each row = date, entries = sigma_t (daily vol)
    in the same units as the original returns (i.e. not *100).
    """
    vol_dict = {}

    for asset, res in models.items():
        sigma_pct = res.conditional_vol_series()  # in percent
        sigma = sigma_pct / 100.0                # back to raw return units
        vol_dict[asset] = sigma

    vols = pd.DataFrame(vol_dict)
    # Align to returns index & drop missing
    vols = vols.loc[returns.index].dropna(how="any")
    return vols


def summarize_garch_models(
    models: Dict[str, UnivariateGarchResult],
    lags: int = 10,
) -> pd.DataFrame:
    """
    Summarize key diagnostics for each fitted GARCH model.

    For each asset, we collect:
      - AIC
      - BIC
      - Ljung-Box p-value on standardized residuals (autocorrelation in mean)
      - Ljung-Box p-value on squared standardized residuals (remaining ARCH)
    """
    records = []

    for asset, res in models.items():
        fit_res = res.fit_result
        std_resid = res.standardized_residuals().dropna()

        lb_resid = acorr_ljungbox(std_resid, lags=[lags], return_df=True)
        lb_resid2 = acorr_ljungbox((std_resid**2), lags=[lags], return_df=True)

        p_lb_resid = float(lb_resid["lb_pvalue"].iloc[-1])
        p_lb_resid2 = float(lb_resid2["lb_pvalue"].iloc[-1])

        records.append(
            {
                "asset": asset,
                "aic": fit_res.aic,
                "bic": fit_res.bic,
                "lb_resid_pvalue": p_lb_resid,
                "lb_resid2_pvalue": p_lb_resid2,
            }
        )

    summary_df = pd.DataFrame(records).set_index("asset")
    return summary_df


def rolling_garch_vol_forecast_for_asset(
    returns: pd.Series,
    dist: str = "t",
    window: int | None = None,
    min_obs: int = 500,
) -> pd.Series:
    """
    True out-of-sample 1-step-ahead GARCH volatility forecasts for ONE asset.

    For each date t (after we have enough history), we:
      - fit GARCH(1,1) using ONLY data up to t-1
      - forecast 1-day ahead variance for date t
      - store sqrt(variance) as sigma_t

    Parameters
    ----------
    returns : pd.Series
        Full return series for the asset (log returns).
    dist : str
        Error distribution for GARCH (e.g. "t").
    window : int or None
        If None -> expanding window (use all data up to t-1).
        If int  -> rolling window of that length (use last 'window' obs).
    min_obs : int
        Minimum number of observations before we start forecasting.

    Returns
    -------
    sigma_oos : pd.Series
        Out-of-sample volatility forecasts sigma_t (in same units as returns),
        indexed by the dates of realized returns.
    """
    r = returns.dropna()
    n = len(r)
    if n < min_obs + 1:
        raise ValueError(f"Not enough observations for rolling GARCH: got {n}, need at least {min_obs + 1}")

    sigmas = {}

    # We'll start forecasting from index 'start_idx', i.e. we will produce
    # a forecast for date r.index[start_idx], using data strictly before it.
    start_idx = min_obs

    for i in range(start_idx, n):
        # Data used for estimation ends at i-1
        if window is None:
            r_window = r.iloc[:i]               # expanding window
        else:
            r_window = r.iloc[max(0, i - window): i]  # rolling window

        am = arch_model(
            r_window * 100,    # percent scaling
            mean="zero",
            vol="GARCH",
            p=1,
            q=1,
            dist=dist,
        )
        res = am.fit(disp="off")

        # 1-step-ahead variance forecast for "next" observation (which is index i)
        fcast = res.forecast(horizon=1, reindex=False)
        # last row, horizon 0 is the 1-step-ahead forecast
        var_next = float(fcast.variance.iloc[-1, 0])
        sigma_next = np.sqrt(var_next) / 100.0  # back to raw return units

        date_next = r.index[i]  # this is the date whose return we haven't used in estimation
        sigmas[date_next] = sigma_next

    sigma_oos = pd.Series(sigmas, name=returns.name)
    return sigma_oos


def build_rolling_volatility_matrix_oos(
    returns: pd.DataFrame,
    dist: str = "t",
    window: int | None = None,
    min_obs: int = 500,
) -> pd.DataFrame:
    """
    Out-of-sample 1-step-ahead volatility forecasts for ALL assets.

    For each asset, we call rolling_garch_vol_forecast_for_asset and then
    align the resulting series into a volatility matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns, one column per asset.
    dist : str
        Error distribution for all GARCH models.
    window : int or None
        None -> expanding window; int -> rolling window length.
    min_obs : int
        Minimum observations per asset before we start forecasting.

    Returns
    -------
    vols_oos : pd.DataFrame
        DataFrame of out-of-sample sigma_{i,t}, with rows = dates,
        columns = assets, entries in same units as returns.
    """
    vol_dict = {}

    for col in returns.columns:
        print(f"Rolling OOS GARCH(1,1) for {col} (dist={dist}, window={window}, min_obs={min_obs})...")
        sigma_oos = rolling_garch_vol_forecast_for_asset(
            returns[col],
            dist=dist,
            window=window,
            min_obs=min_obs,
        )
        vol_dict[col] = sigma_oos

    # Combine & align (inner join on dates)
    vols_oos = pd.DataFrame(vol_dict).dropna(how="any")
    return vols_oos