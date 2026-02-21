# phase3_covariance_check.py

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
import pandas as pd


def main():
    tickers = ["SPY", "QQQ", "TLT", "GLD", "AAPL"]
    weights = pd.Series([0.2,0.2,0.2,0.2,0.2], index=tickers)

    start = "2015-01-01"
    end   = "2025-01-01"

    # Phase 1 – Data
    print("Downloading prices...")
    prices = download_price_data(tickers, start, end)
    prices = align_and_clean_prices(prices)
    returns = compute_log_returns(prices)

    # Phase 2 – GARCH vols
    print("Fitting GARCH models...")
    models = fit_garch_for_all_assets(returns, dist="t")
    vols = build_volatility_matrix(models, returns)

    print("Vols shape:", vols.shape)

    # Phase 3 – Correlation + Covariance + Portfolio Vol
    print("Computing rolling correlations...")
    corr_dict = compute_rolling_correlation(returns, window=250)

    print("Building covariance matrices...")
    cov_dict = build_covariance_matrix(vols, corr_dict)

    print("Computing portfolio volatility...")
    port_vol = compute_portfolio_volatility(cov_dict, weights)
    print("Portfolio volatility (sample):")
    print(port_vol.head())

    print("Computing portfolio return series...")
    port_ret = compute_portfolio_returns(returns, weights)
    print(port_ret.head())


if __name__=="__main__":
    main()