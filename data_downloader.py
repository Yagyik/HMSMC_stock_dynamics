# project/data/data_downloader.py
# --------------------------------
# This module contains functions to download stock market data from yfinance and Tiingo,
# and store the downloaded data on disk in various formats (CSV, Parquet, or HDF5).
#
# Key features:
# - Download data for a given ticker between specified start and end dates.
# - Support for both yfinance and Tiingo as data sources.
# - Save data in a user-specified format.
#
# Usage:
#   - Instantiate the DataDownloader with a configuration object.
#   - Call download_yfinance_data() or download_tiingo_data() with appropriate arguments.
#
# Note:
#   - For Tiingo, ensure you set the API key in your configuration (as TIINGO_API_KEY).
#   - The downloaded files are stored in designated folders under "data/yfinance" or "data/tiingo".

import os
import pandas as pd
import yfinance as yf

# For Tiingo, install the tiingo package: pip install tiingo
from tiingo import TiingoClient

class DataDownloader:
    def __init__(self, config):
        self.config = config
        # Set up output directories for yfinance and Tiingo data downloads
        self.yfinance_output_dir = os.path.join("data", "yfinance")
        self.tiingo_output_dir = os.path.join("data", "tiingo")
        os.makedirs(self.yfinance_output_dir, exist_ok=True)
        os.makedirs(self.tiingo_output_dir, exist_ok=True)
    
    def download_yfinance_data(self, ticker, start_date, end_date, interval="1d", output_format="csv"):
        """
        Download stock data from yfinance and save it to disk.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            interval (str): Data interval (e.g., '1d', '1h').
            output_format (str): Format to save file. Options: "csv", "parquet", "hdf5".
        
        Returns:
            file_path (str): Path to the saved file.
        """
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        # Select the desired columns (first level of MultiIndex)
        data.columns = data.columns.get_level_values(0)
        # Keep only the columns you are interested in
        # data = data[['Open', 'Close', 'Volume', 'Low', 'High']]
        # If the index already contains the dates, rename the index
        data.index.name = 'Date'  # Ensure the index is named "Date"
            
        # Resetting the index if necessary
        data.reset_index(inplace=True)
        # Ensure that the index is of type datetime
        data['Date'] = pd.to_datetime(data['Date'])
        # Set the 'Date' column as the index again (in case it's reset)
        data.set_index('Date', inplace=True)
        
        if data.empty:
            print(f"No data returned for ticker {ticker} from yfinance.")
            return None

        filename = f"{ticker}_{start_date}_{end_date}.{output_format}"
        file_path = os.path.join(self.yfinance_output_dir, filename)
        
        if output_format == "csv":
            data.to_csv(file_path)
        elif output_format == "parquet":
            data.to_parquet(file_path)
        elif output_format == "hdf5":
            data.to_hdf(file_path, key='data', mode='w', complevel=9, complib='blosc')
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        print(f"Downloaded yfinance data for {ticker} saved to {file_path}")
        return file_path

    def download_tiingo_data(self, ticker, start_date, end_date, output_format="csv"):
        """
        Download stock data from Tiingo and save it to disk.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            output_format (str): Format to save file. Options: "csv", "parquet", "hdf5".
        
        Returns:
            file_path (str): Path to the saved file.
        """
        # Set up the Tiingo client with API key from configuration
        tiingo_config = {'api_key': getattr(self.config, 'TIINGO_API_KEY', None)}
        client = TiingoClient(tiingo_config)
        
        try:
            data = client.get_dataframe(ticker, startDate=start_date, endDate=end_date)
        except Exception as e:
            print(f"Error downloading data for {ticker} from Tiingo: {e}")
            return None
        
        if data.empty:
            print(f"No data returned for ticker {ticker} from Tiingo.")
            return None

        filename = f"{ticker}_{start_date}_{end_date}.{output_format}"
        file_path = os.path.join(self.tiingo_output_dir, filename)
        
        if output_format == "csv":
            data.to_csv(file_path)
        elif output_format == "parquet":
            data.to_parquet(file_path)
        elif output_format == "hdf5":
            data.to_hdf(file_path, key='data', mode='w', complevel=9, complib='blosc')
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        print(f"Downloaded Tiingo data for {ticker} saved to {file_path}")
        return file_path

if __name__ == "__main__":
    # Example usage:
    import sys
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'config_simple')))
    # print(sys.path)


    from HMSMC_stock_dynamics.src.config.config_simple import Config
    config = Config()
    downloader = DataDownloader(config)
    
    # Download sample data from yfinance
    # downloader.download_yfinance_data(ticker="AAPL", start_date="2020-01-01", end_date="2022-12-31", output_format="csv")
    # Download data for multiple tickers and combine into a single DataFrame
    tickers = ["AAPL", "MSFT", "IBM"]
    combined_data = pd.DataFrame()

    for ticker in tickers:
        data = downloader.download_yfinance_data(ticker=ticker, start_date="2017-01-01", end_date="2022-12-31", output_format="csv")
        if data is not None:
            df = pd.read_csv(data, index_col=0)
            df.columns = [f"{ticker}_{col}" for col in df.columns]
            if combined_data.empty:
                combined_data = df
            else:
                combined_data = combined_data.join(df, how='outer')

    # Save the combined DataFrame to a CSV file
    output_path = os.path.join("datasets", "raw", "yfinance_test.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_data.to_csv(output_path)
    print(f"Combined data saved to {output_path}")

    # To use Tiingo, ensure TIINGO_API_KEY is set in your configuration and uncomment the line below:
    # downloader.download_tiingo_data(ticker="AAPL", start_date="2022-01-01", end_date="2022-12-31", output_format="csv")
