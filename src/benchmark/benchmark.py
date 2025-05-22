import pandas as pd
import numpy as np
from src.benchmark.constants import BenchmarkInferenceSchema, BenchmarkAggregation, BenchmarkReduction
from sklearn.metrics import r2_score
from datetime import time

import logging

logger = logging.getLogger(__name__)


class RegressionEvaluator:
    """
    Evaluate time-series regression model outputs using various error metrics.

    This class compares predicted and actual price-related fields from two DataFrames
    indexed by timestamp and symbol. It supports optional aggregation (e.g., by minute, hour, day)
    and reduction (e.g., mean, sum) to summarize errors over time.

    Attributes:
        actual_df (pd.DataFrame): Ground-truth values indexed by timestamp and symbol.
        predicted_df (pd.DataFrame): Predicted values indexed by timestamp and symbol.
        aggregation (str): Temporal aggregation granularity ('none', 'minute', 'hour', 'day').
        reduction (str): Reduction function to apply during aggregation ('none', 'mean', 'sum', etc.).
    """

    def __init__(
            self,
            actual_df: pd.DataFrame,
            predicted_df: pd.DataFrame,
            aggregation: str = BenchmarkAggregation.NONE.value,
            reduction: str = BenchmarkReduction.NONE.value,
            market_open_time: time = time(9, 30),
            market_close_time: time = time(16, 0),
    ):
        """
        Initialize the RegressionEvaluator.

        Args:
            actual_df (pd.DataFrame): Ground-truth values with a MultiIndex of [timestamp, symbol].
            predicted_df (pd.DataFrame): Predicted values with the same index and columns.
            aggregation (str): How to group timestamps (e.g., 'minute', 'hour', 'day', 'none').
            reduction (str): Aggregation function to apply (e.g., 'mean', 'sum', 'none').
            # TODO: Allow for different start / end time per day
            market_open_time (time): Time at which the market is open. default is (9, 30)
            market_close_time (time): Time at which the market is close. default is (16, 0)

        Raises:
            AssertionError: If aggregation or reduction is not in the defined enums.
            ValueError: If index or column structure of the inputs is incorrect.
        """

        assert aggregation in {agg.value for agg in BenchmarkAggregation}, f"unsupported aggregation {aggregation}"
        assert reduction in {red.value for red in BenchmarkReduction}, f"unsupported reduction {reduction}"

        self.market_open_time = market_open_time
        self.market_close_time = market_close_time
        self.aggregation = aggregation
        self.reduction = reduction
        self.columns = BenchmarkInferenceSchema.price_fields()

        self.actual_df = actual_df.sort_index()
        self.predicted_df = predicted_df.sort_index()

        self._validate_inputs()
        self._aggregation_key = "time_group"

    @staticmethod
    def _validate_index(df: pd.DataFrame):
        """
        Validate that the DataFrame has the required MultiIndex of [timestamp, symbol].

        Args:
            df (pd.DataFrame): Input DataFrame.

        Raises:
            ValueError: If required index levels are missing.
        """

        expected_indices = {
            BenchmarkInferenceSchema.TIMESTAMP.value,
            BenchmarkInferenceSchema.SYMBOL.value
        }
        if not (set(df.index.names) == expected_indices):
            missing_indices = expected_indices - set(df.index.names)
            raise ValueError(f"Input df is missing indices: {missing_indices}.")

    def _filter_market_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Only retains timestamps within market open and close (inclusive)
        """
        df = df[df.index.get_level_values(BenchmarkInferenceSchema.TIMESTAMP.value).time >= self.market_open_time]
        df = df[df.index.get_level_values(BenchmarkInferenceSchema.TIMESTAMP.value).time <= self.market_close_time]
        return df

    def _validate_columns(self, df: pd.DataFrame, df_name: str):
        """
        Validate that the DataFrame has the expected columns.

        Args:
            df (pd.DataFrame): Input DataFrame.
            df_name (str): Descriptive name for error messages.

        Raises:
            ValueError: If the columns do not match expected schema.
        """

        if not (set(self.columns) == set(df.columns)):
            raise ValueError(f"Expected columns {self.columns} but got {set(df.columns)} in {df_name}.")

    def _validate_inputs(self):
        """
        Run full input validation on both actual and predicted DataFrames.
        """

        self._validate_index(self.actual_df)
        self._validate_index(self.predicted_df)

        if not self.actual_df.index.equals(self.predicted_df.index):
            raise ValueError("actual_df and predicted_df must have identical indices.")

        self._validate_columns(self.actual_df, "ground-truth dataframe")
        self._validate_columns(self.predicted_df, "predicted dataframe")

        self.actual_df = self._filter_market_trading_hours(self.actual_df)
        self.predicted_df = self._filter_market_trading_hours(self.predicted_df)

    def _get_timestamp_aggregation_key(self) -> str | None:
        """
        Translate aggregation string to a pandas time frequency string.

        Returns:
            str or None: Pandas frequency string like 'D', 'h', or 'M'.
        """

        key = self.aggregation
        if key == BenchmarkAggregation.DAY.value:
            return "D"
        elif key == BenchmarkAggregation.HOUR.value:
            return "h"
        elif key == BenchmarkAggregation.MINUTE.value:
            return "M"
        elif key == BenchmarkAggregation.NONE.value:
            return None
        else:
            raise ValueError(f"Unrecognized aggregation key {key}")

    def _get_timestamp_reduction_key(self) -> str | None:
        """
        Return the reduction key if set.

        Returns:
            str or None: The reduction function name, e.g., 'mean'.
        """

        key = self.reduction
        if key == BenchmarkReduction.NONE.value:
            return None
        else:
            return key

    def _set_timestamp_aggregation(self, df):
        """
        Add a temporary time_group column to the DataFrame based on aggregation frequency.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with added 'time_group' column.
        """

        timestamps = df.index.get_level_values(BenchmarkInferenceSchema.TIMESTAMP.value)
        time_group = timestamps.floor(self._get_timestamp_aggregation_key())
        df[self._aggregation_key] = time_group
        return df

    def _aggregate_and_reduce(self, df):
        """
        Aggregate and reduce the input DataFrame by timestamp and symbol.

        Args:
            df (pd.DataFrame): Input DataFrame of error values.

        Returns:
            pd.DataFrame: Aggregated and reduced DataFrame.
        """

        # Check that both are none or both are non-null
        assert (self.aggregation == BenchmarkAggregation.NONE.value) == (
                self.reduction == BenchmarkReduction.NONE.value)

        if self.aggregation == BenchmarkAggregation.NONE.value:
            return df

        df = self._set_timestamp_aggregation(df)
        reduction = self._get_timestamp_reduction_key()

        df = df.groupby([BenchmarkInferenceSchema.SYMBOL.value, self._aggregation_key]).agg(reduction)
        df = df.rename_axis(index={self._aggregation_key: BenchmarkInferenceSchema.TIMESTAMP.value})

        return df

    def absolute_error(self, percentage=False):
        """
        Compute absolute error between actual and predicted values.

        Args:
            percentage (bool): If True, computes absolute error as a percentage.

        Returns:
            pd.DataFrame: Error values (or percentage errors), optionally aggregated and reduced.
        """

        _error = (self.actual_df - self.predicted_df).abs()
        if percentage:
            _error = (_error / self.actual_df.abs()) * 100

        _error = self._aggregate_and_reduce(_error)

        return _error

    def root_squared_error(self):
        """
        Compute root of squared error between actual and predicted values.

        Returns:
            pd.DataFrame: Root squared error values, optionally aggregated and reduced.
        """

        _error = (self.actual_df - self.predicted_df) ** 2
        _error = self._aggregate_and_reduce(_error)
        return np.sqrt(_error)

    def r2(self):
        """
        Compute R² (coefficient of determination) for each price field across groups.

        Aggregation must be enabled to group by timestamp intervals, since r^2 is computed over time.

        Returns:
            pd.DataFrame: R² values for each symbol and timestamp group.

        Raises:
            AssertionError: If aggregation is 'none', as R² requires grouping.
        """

        # Join the two dataframes
        assert self.aggregation != BenchmarkAggregation.NONE.value, "aggregation cannot be none for r2"

        joined = self.actual_df.join(self.predicted_df, lsuffix="_actual", rsuffix="_pred")
        joined = self._set_timestamp_aggregation(joined)

        def _column_wise_r2(group, columns):
            if len(group) < 2:
                return {col: np.nan for col in columns}

            col_to_r2 = {}
            for col in columns:
                y_true = group[f"{col}_actual"].values
                y_pred = group[f"{col}_pred"].values
                col_to_r2[col] = r2_score(y_true, y_pred)
            return col_to_r2

        # Groupby and apply r2 for each column
        result = joined.groupby(
            [
                BenchmarkInferenceSchema.SYMBOL.value,
                self._aggregation_key
            ]
        ).apply(
            _column_wise_r2,
            columns=self.columns,
            include_groups=False
        )

        result = pd.DataFrame(result.tolist(), index=result.index)
        result = result.rename_axis(index={self._aggregation_key: BenchmarkInferenceSchema.TIMESTAMP.value})

        return result
