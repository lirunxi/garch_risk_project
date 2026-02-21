# risk_model/data_loader.py

from typing import List
import pandas as pd
import yfinance as yf


def download_price_data(
    tickers: List[str],
    start: str,
    end: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers using yfinance.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols, e.g. ["SPY", "QQQ", "TLT", "GLD", "AAPL"].
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    auto_adjust : bool
        Whether to download auto-adjusted prices (dividends/splits).

    Returns
    -------
    prices : pd.DataFrame
        DataFrame of adjusted close prices with datetime index and
        tickers as columns.
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )

    # yfinance returns a multiindex for multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        # prefer Adjusted Close if available
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"].copy()
        else:
            prices = data["Close"].copy()
    else:
        # Single ticker case
        prices = data.copy()
        prices = prices.to_frame(name=tickers[0])  # ensure column named by ticker

    # Drop days where all prices are NaN (full market holiday, etc.)
    prices = prices.dropna(how="all")

    # Optional: sort index just in case
    prices = prices.sort_index()

    return prices