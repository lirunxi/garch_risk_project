# phase5_backtesting_check.py

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
from risk_model.var_es import (
    parametric_var_es_normal,
    historical_var_es,
)
from risk_model.backtesting import (
    compute_exceedances,
    kupiec_test,
    christoffersen_independence_test,
    christoffersen_conditional_coverage_test,
)


def main():
    tickers = ["SPY", "QQQ", "TLT", "GLD", "AAPL"]
    weights = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=tickers)

    start = "2015-01-01"
    end = "2025-01-01"
    alpha = 0.99

    # ----- Phases 1–3: Data, GARCH vols, portfolio vol/returns -----
    print("Downloading prices...")
    prices = download_price_data(tickers, start, end)
    prices = align_and_clean_prices(prices)
    returns = compute_log_returns(prices)

    print("Fitting GARCH models...")
    models = fit_garch_for_all_assets(returns, dist="t")
    vols = build_volatility_matrix(models, returns)

    print("Computing rolling correlations...")
    corr_dict = compute_rolling_correlation(returns, window=250)

    print("Building covariance matrices...")
    cov_dict = build_covariance_matrix(vols, corr_dict)

    print("Computing portfolio volatility...")
    port_vol = compute_portfolio_volatility(cov_dict, weights)

    print("Computing portfolio returns...")
    port_ret = compute_portfolio_returns(returns, weights)

    # Align
    common_idx = port_vol.index.intersection(port_ret.index)
    port_vol = port_vol.loc[common_idx]
    port_ret = port_ret.loc[common_idx]

    # ----- Phase 4: VaR & ES -----
    mean_ret = port_ret.mean()

    print(f"\nComputing parametric normal VaR/ES (alpha={alpha})...")
    var_es_param = parametric_var_es_normal(
        portfolio_vol=port_vol,
        alpha=alpha,
        mean_return=mean_ret,
    )

    print(f"Computing historical VaR/ES (alpha={alpha})...")
    var_es_hist = historical_var_es(
        portfolio_returns=port_ret,
        window=250,
        alpha=alpha,
    )

    # Align VaR series with returns
    param_var = var_es_param["VaR"]
    hist_var = var_es_hist["VaR"]

    # Make sure all are aligned
    idx_param = param_var.index.intersection(port_ret.index)
    idx_hist = hist_var.index.intersection(port_ret.index)

    param_var = param_var.loc[idx_param]
    hist_var = hist_var.loc[idx_hist]

    ret_param = port_ret.loc[idx_param]
    ret_hist = port_ret.loc[idx_hist]

    # ----- Phase 5: Backtesting -----
    print("\n=== Parametric Normal VaR Backtest ===")
    exceed_param = compute_exceedances(ret_param, param_var)
    uc_param = kupiec_test(exceed_param, alpha=alpha)
    ind_param = christoffersen_independence_test(exceed_param)
    cc_param = christoffersen_conditional_coverage_test(exceed_param, alpha=alpha)

    print("\nUnconditional coverage:")
    print(uc_param)
    print("\nIndependence test:")
    print(ind_param)
    print("\nConditional coverage:")
    print(cc_param)

    print("\n=== Historical VaR Backtest ===")
    exceed_hist = compute_exceedances(ret_hist, hist_var)
    uc_hist = kupiec_test(exceed_hist, alpha=alpha)
    ind_hist = christoffersen_independence_test(exceed_hist)
    cc_hist = christoffersen_conditional_coverage_test(exceed_hist, alpha=alpha)

    print("\nUnconditional coverage:")
    print(uc_hist)
    print("\nIndependence test:")
    print(ind_hist)
    print("\nConditional coverage:")
    print(cc_hist)


if __name__ == "__main__":
    main()