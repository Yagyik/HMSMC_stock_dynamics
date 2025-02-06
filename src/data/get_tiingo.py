import requests

def fetch_tiingo_data(ticker, start_date, end_date, api_key):
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        'startDate': start_date,
        'endDate': end_date,
        'token': api_key
    }
    response = requests.get(url, params=params)
    return response.json()

def fetch_tiingo_data_multiple(tickers, start_date, end_date, api_key):
    all_data = {}
    for ticker in tickers:
        data = fetch_tiingo_data(ticker, start_date, end_date, api_key)
        all_data[ticker] = data
    return all_data
