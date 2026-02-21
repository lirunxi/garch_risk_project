# phase6_regimes_sensitivity.py

import pandas as pd

from risk_model.data_loader import download_price_data
from risk_model.preprocessing import (
    align_and_clean_prices,
    compute_log_returns,
)
from risk_model.garch_models import (
    fit_garch_for_all_assets,
    build_volatility_matrix,
)
from risk_model.portfolio import (
    compute_rolling_correlation,
    build_covariance_matrix,
    compute_portfolio_volatility,
    compute_portfolio_returns,
)
from risk_model.var_es import parametric_var_es_normal
from risk_model.backtesting import (
    compute_exceedances,
    summarize_exceedances_by_regime,
)
from risk_model.regimes import (
    rolling_realized_vol,
    define_volatility_regimes,
)


def run_for_alpha(alpha: float):
    print(f"\n=== Analysis for alpha = {alpha} ===")

    tickers = ["SPY", "QQQ", "TLT", "GLD", "AAPL"]
    weights = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=tickers)

    start = "2015-01-01"
    end = "2025-01-01"

    # ---- Phases 1–3: Data, GARCH, portfolio vol/returns ----
    prices = download_price_data(tickers, start, end)
    prices = align_and_clean_prices(prices)
    returns = compute_log_returns(prices)

    models = fit_garch_for_all_assets(returns, dist="t")
    vols = build_volatility_matrix(models, returns)

    corr_dict = compute_rolling_correlation(returns, window=250)
    cov_dict = build_covariance_matrix(vols, corr_dict)

    port_vol = compute_portfolio_volatility(cov_dict, weights)
    port_ret = compute_portfolio_returns(returns, weights)

    # Align
    common_idx = port_vol.index.intersection(port_ret.index)
    port_vol = port_vol.loc[common_idx]
    port_ret = port_ret.loc[common_idx]

    # ---- Phase 4: Parametric VaR (normal) ----
    mean_ret = port_ret.mean()
    var_es_param = parametric_var_es_normal(
        portfolio_vol=port_vol,
        alpha=alpha,
        mean_return=mean_ret,
    )
    var_param = var_es_param["VaR"]

    # Align VaR with returns
    idx = var_param.index.intersection(port_ret.index)
    var_param = var_param.loc[idx]
    ret = port_ret.loc[idx]

    # ---- Regimes: realized volatility based on returns ----
    realized_vol = rolling_realized_vol(ret, window=20)
    regimes = define_volatility_regimes(realized_vol, low_quantile=0.2, high_quantile=0.8)

    # Align regimes with exceedances
    exceed = compute_exceedances(ret, var_param)

    # ---- Summary by regime ----
    summary = summarize_exceedances_by_regime(exceed, regimes, alpha=alpha)
    print("\nExceedance summary by regime:")
    print(summary)


def main():
    # Run analysis for two confidence levels as a simple sensitivity study
    run_for_alpha(0.99)
    run_for_alpha(0.95)


if __name__ == "__main__":
    main()