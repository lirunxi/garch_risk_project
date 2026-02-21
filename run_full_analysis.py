# run_full_analysis.py

import os
import json
import pandas as pd

from risk_model.data_loader import download_price_data
from risk_model.preprocessing import (
    align_and_clean_prices,
    compute_log_returns,
)
from risk_model.eda import (
    summarize_returns,
    # plot_volatility_clustering,
    # plot_acf_pacf,
    # plot_acf_pacf_squared,
    adf_test,
)
from risk_model.garch_models import (
    fit_garch_for_all_assets,
    build_volatility_matrix,
    summarize_garch_models,
    build_rolling_volatility_matrix_oos,
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
    summarize_exceedances_by_regime,
)
from risk_model.regimes import (
    rolling_realized_vol,
    define_volatility_regimes,
)


def ensure_output_dir(path: str = "output") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def main():
    out_dir = ensure_output_dir()

    # ------------------------------
    # Configuration
    # ------------------------------
    tickers = ["SPY", "QQQ", "TLT", "GLD", "AAPL"]
    weights = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=tickers)
    start = "2015-01-01"
    end = "2025-01-01"
    alpha_param = 0.99      # main VaR confidence level
    alpha_hist = 0.99
    window_corr = 250       # for rolling correlations & historical VaR
    window_realized = 20    # for realized vol (regimes)

    print("=== GARCH – VaR Project: Full Run ===")

    # ------------------------------
    # Phase 1: Data
    # ------------------------------
    print("\n[Phase 1] Downloading and cleaning data...")
    prices = download_price_data(tickers, start, end)
    prices = align_and_clean_prices(prices, drop_if_any_nan=True)
    returns = compute_log_returns(prices)

    prices.to_csv(os.path.join(out_dir, "cleaned_prices.csv"))
    returns.to_csv(os.path.join(out_dir, "returns.csv"))

    # ------------------------------
    # Phase 2: EDA & Diagnostics
    # ------------------------------
    print("\n[Phase 2] Exploratory Data Analysis and Diagnostics...")

    # Summary stats across all assets
    summary_stats = summarize_returns(returns)
    summary_stats.to_csv(os.path.join(out_dir, "summary_returns.csv"))
    print("\nReturn summary (truncated):")
    print(summary_stats.head())

    # ADF tests (stationarity) per asset
    adf_records = []
    for asset in tickers:
        res = adf_test(returns[asset])
        res["asset"] = asset
        adf_records.append(res)

    adf_df = pd.DataFrame(adf_records).set_index("asset")
    adf_df.to_csv(os.path.join(out_dir, "adf_results.csv"))
    print("\nADF test results (truncated):")
    print(adf_df[["statistic", "pvalue"]])

    # (Optional) You can generate plots separately in phase2_eda_diagnostics.py

    # ------------------------------
    # Phase 3: Univariate GARCH per asset (for diagnostics)
    # ------------------------------
    print("\n[Phase 3] Fitting GARCH(1,1) models per asset (for diagnostics)...")
    garch_models = fit_garch_for_all_assets(returns, dist="t")

    # In-sample vols (for reference / optional plots)
    vols_in_sample = build_volatility_matrix(garch_models, returns)
    vols_in_sample.to_csv(os.path.join(out_dir, "garch_vols_in_sample.csv"))
    print("In-sample GARCH vol matrix shape:", vols_in_sample.shape)

    # GARCH diagnostics (AIC, BIC, Ljung-Box)
    garch_diag = summarize_garch_models(garch_models, lags=10)
    garch_diag.to_csv(os.path.join(out_dir, "garch_diagnostics.csv"))
    print("\nGARCH diagnostics (truncated):")
    print(garch_diag)

    # ------------------------------
    # Phase 3b: OUT-OF-SAMPLE portfolio volatility
    # ------------------------------
    print("\n[Phase 3b] Computing OUT-OF-SAMPLE GARCH volatilities...")
    vols = build_rolling_volatility_matrix_oos(
        returns,
        dist="t",
        window=None,      # expanding window; or set an int for rolling
        min_obs=500,      # first 500 days used to initialize
    )
    vols.to_csv(os.path.join(out_dir, "garch_vols.csv"))   # overwrite old name
    print("OOS GARCH vol matrix shape:", vols.shape)



    # ------------------------------
    # Phase 3 (cont'd): Portfolio covariance & volatility
    # ------------------------------
    print("\n[Phase 3c] Building time-varying covariance matrices and portfolio volatility...")
    corr_dict = compute_rolling_correlation(returns, window=window_corr)
    cov_dict = build_covariance_matrix(vols, corr_dict)

    port_vol = compute_portfolio_volatility(cov_dict, weights)
    port_ret = compute_portfolio_returns(returns, weights)

    # Align series
    common_idx = port_vol.index.intersection(port_ret.index)
    port_vol = port_vol.loc[common_idx]
    port_ret = port_ret.loc[common_idx]

    port_vol.to_csv(os.path.join(out_dir, "portfolio_vol.csv"))
    port_ret.to_csv(os.path.join(out_dir, "portfolio_returns.csv"))

    print("Portfolio vol/return samples:")
    print(pd.concat([port_vol.head(), port_ret.head()], axis=1))

    # ------------------------------
    # Phase 4: VaR & ES (Parametric Normal + Historical)
    # ------------------------------
    print("\n[Phase 4] Computing VaR and ES...")

    mean_ret = port_ret.mean()

    # Parametric normal VaR / ES
    var_es_param = parametric_var_es_normal(
        portfolio_vol=port_vol,
        alpha=alpha_param,
        mean_return=mean_ret,
    )
    var_es_param.to_csv(os.path.join(out_dir, "var_es_parametric_normal.csv"))
    print("\nParametric VaR/ES (sample):")
    print(var_es_param.head())

    # Historical VaR / ES
    var_es_hist = historical_var_es(
        portfolio_returns=port_ret,
        window=window_corr,
        alpha=alpha_hist,
    )
    var_es_hist.to_csv(os.path.join(out_dir, "var_es_historical.csv"))
    print("\nHistorical VaR/ES (sample):")
    print(var_es_hist.head())

    # Align parametric vs historical for comparison
    common_v = var_es_param.index.intersection(var_es_hist.index)
    var_es_compare = pd.DataFrame(
        {
            "VaR_param": var_es_param.loc[common_v, "VaR"],
            "ES_param": var_es_param.loc[common_v, "ES"],
            "VaR_hist": var_es_hist.loc[common_v, "VaR"],
            "ES_hist": var_es_hist.loc[common_v, "ES"],
        }
    )
    var_es_compare.to_csv(os.path.join(out_dir, "var_es_comparison.csv"))
    print("\nVaR/ES comparison (sample):")
    print(var_es_compare.head())

    # ------------------------------
    # Phase 5: Backtesting – Parametric vs Historical
    # ------------------------------
    print("\n[Phase 5] Backtesting VaR models...")

    # Align VaR with returns for backtesting
    param_var = var_es_param["VaR"]
    hist_var = var_es_hist["VaR"]

    # Let pandas align on the intersection of dates and drop rows with NaNs
    param_df = pd.concat(
        [port_ret.rename("ret"), param_var.rename("VaR")],
        axis=1
    ).dropna()

    hist_df = pd.concat(
        [port_ret.rename("ret"), hist_var.rename("VaR")],
        axis=1
    ).dropna()

    # Parametric
    exceed_param = compute_exceedances(param_df["ret"], param_df["VaR"])
    kupiec_param = kupiec_test(exceed_param, alpha=alpha_param)
    ind_param = christoffersen_independence_test(exceed_param)
    cc_param = christoffersen_conditional_coverage_test(exceed_param, alpha=alpha_param)

    # Historical
    exceed_hist = compute_exceedances(hist_df["ret"], hist_df["VaR"])
    kupiec_hist = kupiec_test(exceed_hist, alpha=alpha_hist)
    ind_hist = christoffersen_independence_test(exceed_hist)
    cc_hist = christoffersen_conditional_coverage_test(exceed_hist, alpha=alpha_hist)

    # Save backtest results as JSON
    with open(os.path.join(out_dir, "backtest_parametric.json"), "w") as f:
        json.dump(
            {
                "kupiec": kupiec_param,
                "independence": ind_param,
                "conditional_coverage": cc_param,
            },
            f,
            indent=2,
        )

    with open(os.path.join(out_dir, "backtest_historical.json"), "w") as f:
        json.dump(
            {
                "kupiec": kupiec_hist,
                "independence": ind_hist,
                "conditional_coverage": cc_hist,
            },
            f,
            indent=2,
        )

    print("\nParametric VaR backtest (Kupiec / Christoffersen):")
    print("Kupiec:", kupiec_param)
    print("Independence:", ind_param)
    print("Conditional coverage:", cc_param)

    print("\nHistorical VaR backtest (Kupiec / Christoffersen):")
    print("Kupiec:", kupiec_hist)
    print("Independence:", ind_hist)
    print("Conditional coverage:", cc_hist)

    # ------------------------------
    # Phase 6: Regime Analysis (using Parametric VaR at alpha_param)
    # ------------------------------
    print("\n[Phase 6] Regime-based performance analysis...")

    # Use parametric VaR since it's driven by GARCH
    var_param_bt = param_df["VaR"]
    ret_bt = param_df["ret"]

    # Realized vol & regimes
    realized_vol = rolling_realized_vol(ret_bt, window=window_realized)
    regimes = define_volatility_regimes(realized_vol, low_quantile=0.2, high_quantile=0.8)

    # Exceedances
    exceed_bt = compute_exceedances(ret_bt, var_param_bt)

    regime_summary = summarize_exceedances_by_regime(
        exceedances=exceed_bt,
        regimes=regimes,
        alpha=alpha_param,
    )
    regime_summary.to_csv(os.path.join(out_dir, "regime_exceedance_summary.csv"))

    print("\nExceedance summary by regime:")
    print(regime_summary)

    print("\n=== Full analysis complete. Outputs written to 'output/' directory. ===")


if __name__ == "__main__":
    main()