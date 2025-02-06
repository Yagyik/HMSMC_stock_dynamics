# FILE: HMSMC_stock_dynamics/src/data/load_data.py

import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def fetch_metadata(ticker):
    # Example: Fetch sector classification and fundamental ratios
    sector = "Technology"  # Replace with actual data fetching logic
    pe_ratio = 25.0  # Replace with actual data fetching logic
    return {"sector": sector, "pe_ratio": pe_ratio}

def load_data(tickers, start_date, end_date):
    all_data = {}
    for ticker in tickers:
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        metadata = fetch_metadata(ticker)
        all_data[ticker] = {"stock_data": stock_data, "metadata": metadata}
    return all_data