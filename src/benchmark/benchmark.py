import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.benchmark.benchmark_schema import BenchmarkInferenceSchema


class RegressionEvaluator:
    def __init__(self, actual_df: pd.DataFrame, predicted_df: pd.DataFrame):
        self.columns = BenchmarkInferenceSchema.price_fields()
        self.actual_df = actual_df.sort_index()
        self.predicted_df = predicted_df.sort_index()
        self._validate_inputs(actual_df, predicted_df)

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

    @staticmethod
    def _average_metric_by_date(df):
        df["date"] = df.index.get_level_values(BenchmarkInferenceSchema.TIMESTAMP.value).date
        df = df.groupby([BenchmarkInferenceSchema.SYMBOL.value, "date"]).mean()
        df.index.names = [BenchmarkInferenceSchema.SYMBOL.value, BenchmarkInferenceSchema.TIMESTAMP.value]
        return df

    def absolute_error(self, average_by_day: bool = False):
        abs_error = (self.actual_df - self.predicted_df).abs()
        if average_by_day:
            abs_error = self._average_metric_by_date(abs_error)

        return abs_error

    def squared_error(self):
        return self._compute_metric(lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False))

    def mape(self):
        return self._compute_metric(lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    def r2(self):
        return self._compute_metric(r2_score)
