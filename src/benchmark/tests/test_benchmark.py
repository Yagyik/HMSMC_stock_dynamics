import datetime

import pytest
import pandas as pd
import numpy as np

from src.benchmark.testing import MockInferenceDataGenerator
from src.benchmark.benchmark import RegressionEvaluator
from src.benchmark.constants import BenchmarkAggregation, BenchmarkReduction, BenchmarkInferenceSchema


@pytest.fixture
def mock_data():
    generator = MockInferenceDataGenerator(symbols=["AAPL", "GOOG"], interval="1min")
    df = generator.generate()
    return df


def test_regression_evaluator_accepts_mock_data(mock_data):
    # Create a predicted DataFrame with small noise
    noise = np.random.normal(0, 0.5, size=mock_data.shape)
    predicted_df = mock_data.copy() + noise

    evaluator = RegressionEvaluator(
        actual_df=mock_data,
        predicted_df=predicted_df,
    )

    # Run metrics and assert outputs are non-empty
    assert isinstance(evaluator.absolute_error(), pd.DataFrame)
    assert isinstance(evaluator.root_squared_error(), pd.DataFrame)

    evaluator = RegressionEvaluator(
        actual_df=mock_data,
        predicted_df=predicted_df,
        aggregation="day",
        reduction="mean",
    )
    assert isinstance(evaluator.r2(), pd.DataFrame)


def test_absolute_error_zero_for_same_data(mock_data):
    evaluator = RegressionEvaluator(
        actual_df=mock_data,
        predicted_df=mock_data,
    )
    assert (evaluator.absolute_error() == 0).all().all()


def test_absolute_error_constant_offset(mock_data):
    actual = mock_data
    predicted = actual + 1  # uniform offset
    evaluator = RegressionEvaluator(actual, predicted)
    error_df = evaluator.absolute_error()
    assert np.allclose(error_df.values, 1)


def test_hourly_aggregation_mean_reduction(mock_data):
    actual = mock_data
    predicted = actual + 1
    evaluator = RegressionEvaluator(
        actual_df=actual,
        predicted_df=predicted,
        aggregation=BenchmarkAggregation.HOUR.value,
        reduction=BenchmarkReduction.MEAN.value
    )
    error_df = evaluator.absolute_error()

    assert np.allclose(error_df.values, 1)

    time_index = error_df.index.get_level_values(BenchmarkInferenceSchema.TIMESTAMP.value)

    # check that time-index is floored to hour
    assert (time_index.second == 0).all() & (time_index.minute == 0).all(), "timestamp should be hourly aligned"


def test_r2_perfect(mock_data):
    actual = mock_data
    predicted = actual.copy()

    evaluator = RegressionEvaluator(
        actual_df=actual,
        predicted_df=predicted,
        aggregation=BenchmarkAggregation.HOUR.value,
        reduction=BenchmarkReduction.MEAN.value
    )
    r2_df = evaluator.r2().dropna()
    assert np.allclose(r2_df.values, 1)

    time_index = r2_df.index.get_level_values(BenchmarkInferenceSchema.TIMESTAMP.value)
    # check that time-index is floored to hour
    assert (time_index.second == 0).all() and (time_index.minute == 0).all(), "Timestamp should be hourly-aligned"


def test_invalid_index_raises():
    df = pd.DataFrame({
        "open": [1], "high": [2], "low": [0.5], "close": [1.5]
    }, index=[0])  # Not a MultiIndex

    with pytest.raises(ValueError):
        RegressionEvaluator(df, df)


def test_missing_column_raises():
    df = pd.DataFrame(
        {
            "open": [1],
            "low": [0.5],
            "close": [1.5],
            "symbol": ["AAPL"],
            "timestamp": [datetime.datetime(2025, 3, 2, 1, 23)],
        }
    )  # Not a MultiIndex
    df = df.set_index(["symbol", "timestamp"])

    with pytest.raises(ValueError):
        RegressionEvaluator(df, df)
