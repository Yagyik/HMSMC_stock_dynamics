from datetime import datetime, time
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


@pytest.mark.parametrize("offset", [0, 1, 10])
def test_absolute_error_constant_offset(mock_data, offset):
    actual = mock_data
    predicted = actual + offset  # uniform offset
    evaluator = RegressionEvaluator(actual, predicted)
    error_df = evaluator.absolute_error()
    assert np.allclose(error_df.values, offset)


@pytest.mark.parametrize("offset", [0, 1, 10])
def test_hourly_aggregation_mean_reduction(mock_data, offset):
    actual = mock_data
    predicted = actual + offset
    evaluator = RegressionEvaluator(
        actual_df=actual,
        predicted_df=predicted,
        aggregation=BenchmarkAggregation.HOUR.value,
        reduction=BenchmarkReduction.MEAN.value
    )
    error_df = evaluator.absolute_error()

    assert np.allclose(error_df.values, offset)

    time_index = error_df.index.get_level_values(BenchmarkInferenceSchema.TIMESTAMP.value)

    # check that time-index is floored to hour
    assert (time_index.second == 0).all() & (time_index.minute == 0).all(), "timestamp should be hourly aligned"

    assert isinstance(error_df.index, pd.MultiIndex)
    assert BenchmarkInferenceSchema.TIMESTAMP.value in error_df.index.names
    assert BenchmarkInferenceSchema.SYMBOL.value in error_df.index.names


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


def test_r2_with_noise(mock_data):
    np.random.seed(42)
    actual = mock_data
    predicted = actual + np.random.normal(0, 0.5, size=actual.shape)

    evaluator = RegressionEvaluator(
        actual_df=actual,
        predicted_df=predicted,
        aggregation=BenchmarkAggregation.DAY.value,
        reduction=BenchmarkReduction.MEAN.value
    )
    r2_df = evaluator.r2().dropna()
    assert (r2_df.values <= 1.0).all()
    assert (r2_df.values >= -1.0).all()

    time_index = r2_df.index.get_level_values(BenchmarkInferenceSchema.TIMESTAMP.value)
    assert ((time_index.hour == 0).all() & (time_index.minute == 0).all() & (
            time_index.second == 0).all()), "Timestamp should be day-aligned"


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
            "timestamp": [datetime(2025, 3, 2, 1, 23)],
        }
    )  # Not a MultiIndex
    df = df.set_index(["symbol", "timestamp"])

    with pytest.raises(ValueError):
        RegressionEvaluator(df, df)


def test_sort_index_order(mock_data):
    actual = mock_data
    predcted = actual.iloc[np.random.permutation(len(actual))]

    evaluator = RegressionEvaluator(actual, predcted)

    assert evaluator.actual_df.index.equals(evaluator.predicted_df.index)


def test_market_open_close():
    start_dt = datetime(2024, 1, 1)
    end_dt = datetime(2024, 1, 1)  # market opens at 9:30 AM

    data_generator = MockInferenceDataGenerator(
        symbols=["AAPL", "GOOG"],
        start_date=start_dt,
        end_date=end_dt,
        market_open=time(9, 30),
        market_close=time(16, 0),
    )

    actual_df = data_generator.generate()
    predicted_df = data_generator.generate()

    assert (
            actual_df.index.get_level_values(BenchmarkInferenceSchema.TIMESTAMP.value).time >= time(11, 0)
    ).any()
    assert (
            actual_df.index.get_level_values(BenchmarkInferenceSchema.TIMESTAMP.value).time <= time(13, 0)
    ).any()

    evaluator = RegressionEvaluator(
        actual_df=actual_df,
        predicted_df=predicted_df,
        aggregation=BenchmarkAggregation.NONE.value,
        reduction=BenchmarkReduction.NONE.value,
        market_open_time=time(11, 0),
        market_close_time=time(13, 0)
    )

    assert (evaluator.actual_df.index.get_level_values(BenchmarkInferenceSchema.TIMESTAMP.value).time >= time(11,
                                                                                                              0)).all()
    assert (evaluator.actual_df.index.get_level_values(BenchmarkInferenceSchema.TIMESTAMP.value).time <= time(13,
                                                                                                              0)).all()

    assert evaluator.actual_df.index.equals(evaluator.predicted_df.index)
