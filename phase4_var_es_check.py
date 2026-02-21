# phase4_var_es_check.py

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


def main():
    tickers = ["SPY", "QQQ", "TLT", "GLD", "AAPL"]
    weights = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=tickers)

    start = "2015-01-01"
    end = "2025-01-01"

    # ----- Phase 1: Data -----
    print("Downloading prices...")
    prices = download_price_data(tickers, start, end)
    prices = align_and_clean_prices(prices)
    returns = compute_log_returns(prices)

    # ----- Phase 2: GARCH vols -----
    print("Fitting GARCH models...")
    models = fit_garch_for_all_assets(returns, dist="t")
    vols = build_volatility_matrix(models, returns)
    print("Vols shape:", vols.shape)

    # ----- Phase 3: Covariance + portfolio volatility -----
    print("Computing rolling correlations...")
    corr_dict = compute_rolling_correlation(returns, window=250)

    print("Building covariance matrices...")
    cov_dict = build_covariance_matrix(vols, corr_dict)

    print("Computing portfolio volatility...")
    port_vol = compute_portfolio_volatility(cov_dict, weights)
    print("Portfolio volatility sample:")
    print(port_vol.head())

    # Portfolio returns with fixed weights
    port_ret = compute_portfolio_returns(returns, weights)

    # Align returns with volatility (since vol starts later due to 250-day window)
    common_index = port_vol.index.intersection(port_ret.index)
    port_vol = port_vol.loc[common_index]
    port_ret = port_ret.loc[common_index]

    print("\nAligned shapes:")
    print("  portfolio_vol:", port_vol.shape)
    print("  portfolio_ret:", port_ret.shape)

    # ----- Phase 4: VaR & ES -----
    alpha = 0.99
    mean_ret = port_ret.mean()

    print(f"\nComputing parametric normal VaR/ES (alpha={alpha})...")
    var_es_param = parametric_var_es_normal(
        portfolio_vol=port_vol,
        alpha=alpha,
        mean_return=mean_ret,
    )
    print("Parametric VaR/ES sample:")
    print(var_es_param.head())

    print(f"\nComputing historical VaR/ES (alpha={alpha})...")
    var_es_hist = historical_var_es(
        portfolio_returns=port_ret,
        window=250,
        alpha=alpha,
    )
    print("Historical VaR/ES sample:")
    print(var_es_hist.head())

    # Optionally align parametric & historical for comparison
    common_idx = var_es_param.index.intersection(var_es_hist.index)
    combined = pd.DataFrame(
        {
            "VaR_param": var_es_param.loc[common_idx, "VaR"],
            "VaR_hist": var_es_hist.loc[common_idx, "VaR"],
            "ES_param": var_es_param.loc[common_idx, "ES"],
            "ES_hist": var_es_hist.loc[common_idx, "ES"],
        }
    )

    print("\nComparison sample (parametric vs historical):")
    print(combined.head())

    # Save for later backtesting / plotting
    combined.to_csv("var_es_comparison.csv")
    print("\nSaved var_es_comparison.csv")


if __name__ == "__main__":
    main()