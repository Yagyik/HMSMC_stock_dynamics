from src.benchmark.benchmark_schema import BenchmarkInferenceSchema

import pandas as pd
from datetime import datetime, time
from typing import List
import numpy as np


def generate_market_hours_timestamps(
        date: datetime,
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
    start = datetime.combine(date.date(), market_open)
    end = datetime.combine(date.date(), market_close)
    return pd.date_range(start=start, end=end, freq=interval)


class MockInferenceDataGenerator:
    def __init__(
            self,
            symbols: List[str] = ["AAPL", "AMZN", "GOOG"],
            start_date: datetime = datetime(2025, 1, 1),
            end_date: datetime = datetime(2025, 3, 1),
            interval="1min",
            market_open: time = time(9, 30),
            market_close: time = time(16, 0),
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.market_open = market_open
        self.market_close = market_close

    def generate(self) -> pd.DataFrame:
        timestamps = pd.date_range(self.start_date, self.end_date, freq="D")
        result = []
        symbols_df = pd.DataFrame({BenchmarkInferenceSchema.SYMBOL.value: self.symbols})

        for day_ts in timestamps:
            min_ts = generate_market_hours_timestamps(
                day_ts,
                self.market_open,
                self.market_close,
                self.interval
            )
            start_end_time = pd.DataFrame(
                {
                    BenchmarkInferenceSchema.TIMESTAMP.value: min_ts
                }
            )

            df = symbols_df.merge(start_end_time, how="cross")

            open_price = np.random.uniform(100, 200, len(df))
            high_price = open_price + (np.random.uniform(0, 5, len(df)))
            low_price = open_price - np.random.uniform(0, 5)
            close_price = np.random.uniform(low_price, high_price)

            df[BenchmarkInferenceSchema.OPEN.value] = open_price
            df[BenchmarkInferenceSchema.HIGH.value] = high_price
            df[BenchmarkInferenceSchema.LOW.value] = low_price
            df[BenchmarkInferenceSchema.CLOSE.value] = close_price

            result.append(df)

        result_df = pd.concat(result, axis=0)
        result_df = result_df.set_index(
            [
                BenchmarkInferenceSchema.SYMBOL.value,
                BenchmarkInferenceSchema.TIMESTAMP.value
            ]
        ).sort_index()

        return result_df
