# phase2_eda_diagnostics.py

import pandas as pd
import matplotlib.pyplot as plt

from risk_model.data_loader import download_price_data
from risk_model.preprocessing import (
    align_and_clean_prices,
    compute_log_returns,
)
from risk_model.eda import (
    summarize_returns,
    plot_volatility_clustering,
    adf_test,
    plot_acf_pacf,
    plot_acf_pacf_squared,
)


def main():
    tickers = ["SPY", "QQQ", "TLT", "GLD", "AAPL"]
    start = "2015-01-01"
    end = "2025-01-01"

    print("Downloading prices...")
    prices = download_price_data(tickers, start, end)
    prices = align_and_clean_prices(prices)
    returns = compute_log_returns(prices)

    # 1. Summary statistics across all assets
    print("\n=== Summary statistics for returns ===")
    summary = summarize_returns(returns)
    print(summary)

    # Choose a couple of key assets for deeper diagnostics
    for asset in ["SPY", "QQQ"]:
        print(f"\n=== Diagnostics for {asset} ===")
        r = returns[asset]

        # 2. Volatility clustering visualization
        print("Plotting volatility clustering...")
        plot_volatility_clustering(r, asset_name=asset, window=20)

        # 3. ADF test
        adf_res = adf_test(r)
        print("ADF test results:")
        for k, v in adf_res.items():
            print(f"  {k}: {v}")

        # 4. ACF/PACF of returns
        print("Plotting ACF/PACF of returns...")
        plot_acf_pacf(r, lags=20, title_prefix=asset)

        # 5. ACF/PACF of squared returns
        print("Plotting ACF/PACF of squared returns...")
        plot_acf_pacf_squared(r, lags=20, title_prefix=asset)

    print("\nEDA diagnostics completed.")


if __name__ == "__main__":
    main()