# plot_dashboards.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


OUTPUT_DIR = "output"


def ensure_output_dir(path: str = OUTPUT_DIR) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_core_data():
    """
    Load the key CSV outputs from run_full_analysis.py
    """
    prices = pd.read_csv(
        os.path.join(OUTPUT_DIR, "cleaned_prices.csv"),
        index_col=0,
        parse_dates=True,
    )
    returns = pd.read_csv(
        os.path.join(OUTPUT_DIR, "returns.csv"),
        index_col=0,
        parse_dates=True,
    )
    
     # Load portfolio returns (as DataFrame) then squeeze to Series
    port_ret = pd.read_csv(
        os.path.join(OUTPUT_DIR, "portfolio_returns.csv"),
        index_col=0,
        parse_dates=True,
    )
    port_ret = port_ret.iloc[:, 0]  # convert to Series

    # Load portfolio volatility
    port_vol = pd.read_csv(
        os.path.join(OUTPUT_DIR, "portfolio_vol.csv"),
        index_col=0,
        parse_dates=True,
    )
    port_vol = port_vol.iloc[:, 0]  # convert to Series
        
    
    var_es_param = pd.read_csv(
        os.path.join(OUTPUT_DIR, "var_es_parametric_normal.csv"),
        index_col=0,
        parse_dates=True,
    )
    var_es_hist = pd.read_csv(
        os.path.join(OUTPUT_DIR, "var_es_historical.csv"),
        index_col=0,
        parse_dates=True,
    )
    var_es_comp = pd.read_csv(
        os.path.join(OUTPUT_DIR, "var_es_comparison.csv"),
        index_col=0,
        parse_dates=True,
    )
    regime_summary = pd.read_csv(
        os.path.join(OUTPUT_DIR, "regime_exceedance_summary.csv"),
        index_col=0,
    )

    return {
        "prices": prices,
        "returns": returns,
        "port_ret": port_ret.squeeze(),
        "port_vol": port_vol.squeeze(),
        "var_es_param": var_es_param,
        "var_es_hist": var_es_hist,
        "var_es_comp": var_es_comp,
        "regime_summary": regime_summary,
    }


def plot_portfolio_returns_with_var(
    port_ret: pd.Series,
    var_es_param: pd.DataFrame,
    var_es_hist: pd.DataFrame,
    filename: str = "dashboard_returns_var.png",
):
    """
    Plot portfolio returns with parametric & historical VaR bands
    and mark exceedances.
    """
    # Align everything
    df = pd.DataFrame({"ret": port_ret})
    df = df.join(var_es_param[["VaR"]].rename(columns={"VaR": "VaR_param"}), how="left")
    df = df.join(var_es_hist[["VaR"]].rename(columns={"VaR": "VaR_hist"}), how="left")
    df = df.dropna()

    # Loss = -return
    df["loss"] = -df["ret"]
    df["exceed_param"] = df["loss"] > df["VaR_param"]
    df["exceed_hist"] = df["loss"] > df["VaR_hist"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot returns
    ax.plot(df.index, df["ret"], label="Portfolio return")

    # Plot -VaR (drawn on return scale)
    ax.plot(df.index, -df["VaR_param"], label="Parametric VaR (99%)", linestyle="--")
    ax.plot(df.index, -df["VaR_hist"], label="Historical VaR (99%)", linestyle=":")

    # Mark exceedances (where return < -VaR)
    exceed_param_idx = df.index[df["exceed_param"]]
    exceed_hist_idx = df.index[df["exceed_hist"]]

    ax.scatter(
        exceed_param_idx,
        df.loc[exceed_param_idx, "ret"],
        label="Parametric exceedance",
        marker="o",
        s=20,
    )

    ax.scatter(
        exceed_hist_idx,
        df.loc[exceed_hist_idx, "ret"],
        label="Historical exceedance",
        marker="x",
        s=20,
    )

    ax.set_title("Portfolio Returns with VaR Bands and Exceedances")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_portfolio_volatility(
    port_vol: pd.Series,
    filename: str = "dashboard_portfolio_volatility.png",
):
    """
    Plot rolling portfolio volatility over time.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(port_vol.index, port_vol.values)
    ax.set_title("Portfolio Conditional Volatility (GARCH-based)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (daily std dev)")
    ax.grid(True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_var_es_comparison(
    var_es_comp: pd.DataFrame,
    filename: str = "dashboard_var_es_comparison.png",
):
    """
    Plot average VaR/ES for parametric vs historical methods.
    """
    # Compute simple averages over time
    avg = var_es_comp.mean()

    labels = ["Param VaR", "Hist VaR", "Param ES", "Hist ES"]
    values = [
        avg["VaR_param"],
        avg["VaR_hist"],
        avg["ES_param"],
        avg["ES_hist"],
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_title("Average VaR and ES (99%) – Parametric vs Historical")
    ax.set_ylabel("Loss (absolute units)")
    ax.grid(axis="y")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_regime_exceedance_summary(
    regime_summary: pd.DataFrame,
    filename: str = "dashboard_regime_exceedances.png",
):
    """
    Plot empirical exception rates by volatility regime vs expected rate.
    """
    # regime_summary has columns: n, x, pi_hat, expected
    summary = regime_summary.copy()
    regimes = summary.index.tolist()
    pi_hat = summary["pi_hat"].values
    expected = summary["expected"].iloc[0]

    x_pos = np.arange(len(regimes))

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(x_pos, pi_hat, width=0.5, label="Empirical exception rate")
    ax.axhline(expected, linestyle="--", label="Expected exception rate")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("Exception rate")
    ax.set_title("VaR Exception Rates by Volatility Regime (99%)")
    ax.grid(axis="y")
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ensure_output_dir()

    print("Loading data from 'output/'...")
    data = load_core_data()

    print("Generating portfolio returns + VaR dashboard...")
    plot_portfolio_returns_with_var(
        port_ret=data["port_ret"],
        var_es_param=data["var_es_param"],
        var_es_hist=data["var_es_hist"],
    )

    print("Generating portfolio volatility dashboard...")
    plot_portfolio_volatility(
        port_vol=data["port_vol"],
    )

    print("Generating VaR/ES comparison dashboard...")
    plot_var_es_comparison(
        var_es_comp=data["var_es_comp"],
    )

    print("Generating regime exceedance summary dashboard...")
    plot_regime_exceedance_summary(
        regime_summary=data["regime_summary"],
    )

    print("Dashboards saved to 'output/'.")


if __name__ == "__main__":
    main()