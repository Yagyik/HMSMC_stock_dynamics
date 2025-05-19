import pandas as pd
import numpy as np
from src.benchmark.constants import BenchmarkInferenceSchema, BenchmarkAggregation, BenchmarkReduction


class RegressionEvaluator:
    def __init__(
            self,
            actual_df: pd.DataFrame,
            predicted_df: pd.DataFrame,
            aggregation: str = BenchmarkAggregation.NONE.value,
            reduction: str = BenchmarkReduction.NONE.value,
    ):
        assert aggregation in {agg.value for agg in BenchmarkAggregation}
        assert reduction in {red.value for red in BenchmarkReduction}

        self.aggregation = aggregation
        self.reduction = reduction
        self.columns = BenchmarkInferenceSchema.price_fields()
        self.actual_df = actual_df.sort_index()
        self.predicted_df = predicted_df.sort_index()
        self._validate_inputs(self.actual_df, self.predicted_df)
        self._aggregation_key = "time_group"

    @staticmethod
    def _validate_index(df: pd.DataFrame):
        expected_indices = {
            BenchmarkInferenceSchema.TIMESTAMP.value,
            BenchmarkInferenceSchema.SYMBOL.value
        }
        if not (set(df.index.names) == expected_indices):
            missing_indices = expected_indices - set(df.index.names)
            raise ValueError(f"Input df is missing indices: {missing_indices}.")

    def _validate_columns(self, df: pd.DataFrame, df_name: str):
        if not (set(self.columns) == set(df.columns)):
            raise ValueError(f"Expected columns {self.columns} but got {set(df.columns)} in {df_name}.")

    def _validate_inputs(self, actual_df: pd.DataFrame, predicted_df: pd.DataFrame):
        self._validate_index(actual_df)
        self._validate_index(predicted_df)
        self._validate_columns(actual_df, "ground-truth dataframe")
        self._validate_columns(predicted_df, "predicted dataframe")

    def _get_timestamp_aggregation_key(self) -> str | None:
        key = self.aggregation
        if key == BenchmarkAggregation.DAY.value:
            return "D"
        elif key == BenchmarkAggregation.HOUR.value:
            return "h"
        elif key == BenchmarkAggregation.MINUTE.value:
            return "M"
        elif key == BenchmarkAggregation.NONE.value:
            return
        else:
            raise ValueError(f"Unrecognized aggregation key {key}")

    def _get_timestamp_reduction_key(self) -> str | None:
        key = self.reduction
        if key == BenchmarkReduction.NONE.value:
            return
        else:
            return key

    def _set_timestamp_aggregation(self, df):
        timestamps = df.index.get_level_values(BenchmarkInferenceSchema.TIMESTAMP.value)
        time_group = timestamps.floor(self._get_timestamp_aggregation_key())
        df[self._aggregation_key] = time_group
        return df

    def _aggregate_and_reduce(self, df):
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
        _error = (self.actual_df - self.predicted_df).abs()
        if percentage:
            _error = (_error / self.actual_df.abs()) * 100

        _error = self._aggregate_and_reduce(_error)

        return _error

    def root_squared_error(self):
        _error = (self.actual_df - self.predicted_df) ** 2
        _error = self._aggregate_and_reduce(_error)
        return np.sqrt(_error)

    def r2(self):
        # Join the two dataframes
        assert self.aggregation != BenchmarkAggregation.NONE.value, "aggregation cannot be none for r2"

        joined = self.actual_df.join(self.predicted_df, lsuffix="_actual", rsuffix="_pred")
        joined = self._set_timestamp_aggregation(joined)

        def _column_wise_r2(group, columns):
            if len(group) < 2:
                return {col: np.nan for col in columns}

            result = {
                col: group[f"{col}_actual"].corr(group[f"{col}_pred"])
                for col in columns
            }
            return result

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
