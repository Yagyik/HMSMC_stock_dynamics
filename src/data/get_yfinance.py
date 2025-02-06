# FILE: HMSMC_stock_dynamics/src/data/get_yfinance.py

import yfinance as yf

def fetch_yfinance_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data



def fetch_yfinance_data_multiple(tickers, start_date, end_date):
    all_stock_data = {}
    for ticker in tickers:
        stock_data = fetch_yfinance_data(ticker, start_date, end_date)
        all_stock_data[ticker] = stock_data
    return all_stock_data