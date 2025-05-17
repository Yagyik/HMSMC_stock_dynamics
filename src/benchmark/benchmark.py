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

    def _validate_inputs(self, actual_df: pd.DataFrame, predicted_df: pd.DataFrame):
        if not actual_df.index.equals(predicted_df.index):
            raise ValueError("Indices of actual and predicted dataframes must match exactly (symbol, timestamp).")
        for col in self.columns:
            if (col not in actual_df.columns) or (col not in predicted_df.columns):
                raise ValueError(f"Missing required column: {col}")

    def _compute_metric(self, metric_func, **kwargs):
        results = {}
        for col in self.columns:
            actual = self.actual_df[col]
            predicted = self.predicted_df[col]
            results[col] = metric_func(actual, predicted, **kwargs)
        return results

    def mae(self):
        return self._compute_metric(mean_absolute_error)

    def rmse(self):
        return self._compute_metric(lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False))

    def mape(self):
        return self._compute_metric(lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    def r2(self):
        return self._compute_metric(r2_score)
