# phase2_garch_check.py

import pandas as pd

from risk_model.data_loader import download_price_data
from risk_model.preprocessing import (
    align_and_clean_prices,
    compute_log_returns,
)
from risk_model.garch_models import (
    fit_garch_for_all_assets,
    build_volatility_matrix,
    summarize_garch_models,
)


def main():
    tickers = ["SPY", "QQQ", "TLT", "GLD", "AAPL"]
    start = "2015-01-01"
    end = "2025-01-01"

    # 1. Phase 1: data
    print("Downloading prices...")
    raw_prices = download_price_data(tickers, start, end)
    prices = align_and_clean_prices(raw_prices, drop_if_any_nan=True)
    returns = compute_log_returns(prices)

    print("Prices shape:", prices.shape)
    print("Returns shape:", returns.shape)

    # 2. Phase 2: fit GARCH models per asset
    print("\nFitting GARCH(1,1) per asset...")
    garch_models = fit_garch_for_all_assets(returns, dist="t")

    # 3. Summarize models (AIC, BIC, residual tests)
    print("\nGARCH model diagnostics:")
    diag_df = summarize_garch_models(garch_models, lags=10)
    print(diag_df)

    # 4. Build volatility matrix σ_{i,t}
    print("\nBuilding conditional volatility matrix...")
    vols = build_volatility_matrix(garch_models, returns)
    print("Volatility matrix shape:", vols.shape)
    print("\nSample of volatilities (first 5 rows):")
    print(vols.head())

    # Optionally save to CSV for later phases
    vols.to_csv("garch_vols.csv")
    print("\nSaved garch_vols.csv")


if __name__ == "__main__":
    main()