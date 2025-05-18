from src.benchmark.constants import BenchmarkInferenceSchema

import pandas as pd
from datetime import datetime, time
from typing import List
import numpy as np


def generate_market_hours_timestamps(
        start_date: datetime,
        end_date: datetime,
        market_open: time = time(9, 30),
        market_close: time = time(16, 0),
        interval: str = "1min"  # pandas-compatible freq: "1min", "5min", "15min", etc.
) -> pd.DatetimeIndex:
    """
    Generate intraday timestamps between market open and close for a given date.

    Args:
        date (datetime): The date to generate timestamps for (date part is used).
        market_open (time): Market open time (default 9:30 AM).
        market_close (time): Market close time (default 4:00 PM).
        interval (str): Frequency string compatible with pd.date_range.

    Returns:
        pd.DatetimeIndex: Timestamps during market hours.
    """
    start = datetime.combine(start_date.date(), market_open)
    end = datetime.combine(end_date.date(), market_close)
    df = pd.date_range(start=start, end=end, freq=interval)

    # remove time-stamps after market close/open each day
    df = df[(df.time <= market_close)]
    df = df[(df.time >= market_open)]

    return df


class MockInferenceDataGenerator:
    """
    Generates synthetic time-series OHLC (Open, High, Low, Close) data for financial symbols
    across specified trading dates and hours.

    This mock data generator is useful for benchmarking time-series forecasting models or
    testing data pipelines in financial applications. It simulates random timestamped
    price data for one or more symbols, over a given date range and market hours.

    Attributes:
        symbols (List[str]): List of stock symbols to generate data for.
        start_date (datetime): Start date of the simulation period.
        end_date (datetime): End date of the simulation period.
        interval (str): Time interval between generated data points (e.g., "1min").
        market_open (datetime.time): Market open time (e.g., 9:30 AM).
        market_close (datetime.time): Market close time (e.g., 4:00 PM).
    """

    def __init__(
            self,
            symbols: List[str] = None,
            start_date: datetime = datetime(2025, 1, 1),
            end_date: datetime = datetime(2025, 3, 1),
            interval="1min",
            market_open: time = time(9, 30),
            market_close: time = time(16, 0),
    ):
        if symbols is None:
            symbols = ["AAPL", "AMZN", "GOOG"]

        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.market_open = market_open
        self.market_close = market_close

    def generate(self) -> pd.DataFrame:
        """
        Generate a mock time-series price dataset for a set of symbols over a specified date range.

        This function simulates minute-level OHLC (Open, High, Low, Close) market data for each symbol
        and timestamp within the trading hours defined by `market_open` and `market_close`. The resulting
        DataFrame is indexed by symbol and timestamp.

        Returns:
            pd.DataFrame: A multi-indexed DataFrame with levels ['symbol', 'timestamp'] and columns
            ['open', 'high', 'low', 'close']. Each row contains synthetic OHLC data for a given symbol
            at a specific timestamp.
        """

        timestamps = pd.DataFrame(
            {
                BenchmarkInferenceSchema.TIMESTAMP.value: generate_market_hours_timestamps(
                    start_date=self.start_date,
                    end_date=self.end_date,
                    market_open=self.market_open,
                    market_close=self.market_close
                )
            }
        )

        symbols_df = pd.DataFrame({BenchmarkInferenceSchema.SYMBOL.value: self.symbols})

        df = timestamps.merge(symbols_df, how="cross")

        open_price = np.random.uniform(100, 200, len(df))
        high_price = open_price + (np.random.uniform(0, 5, len(df)))
        low_price = open_price - np.random.uniform(0, 5)
        close_price = np.random.uniform(low_price, high_price)

        df[BenchmarkInferenceSchema.OPEN.value] = open_price
        df[BenchmarkInferenceSchema.HIGH.value] = high_price
        df[BenchmarkInferenceSchema.LOW.value] = low_price
        df[BenchmarkInferenceSchema.CLOSE.value] = close_price

        df = df.set_index(
            [
                BenchmarkInferenceSchema.SYMBOL.value,
                BenchmarkInferenceSchema.TIMESTAMP.value
            ]
        ).sort_index()

        return df


if __name__ == "__main__":
    vals = MockInferenceDataGenerator(
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 2),
    ).generate()