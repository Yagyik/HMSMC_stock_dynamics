import pytest
import pandas as pd
import numpy as np

from src.benchmark.testing import MockInferenceDataGenerator
from src.benchmark.benchmark import RegressionEvaluator
from src.benchmark.benchmark_schema import BenchmarkInferenceSchema


@pytest.fixture
def mock_data():
    generator = MockInferenceDataGenerator(symbols=["AAPL", "GOOG"], interval="1min")
    df = generator.generate()
    return df


def test_regression_evaluator_accepts_mock_data(mock_data):
    # Create a predicted DataFrame with small noise
    noise = np.random.normal(0, 0.5, size=mock_data.shape)
    predicted_df = mock_data.copy() + noise

    evaluator = RegressionEvaluator(actual_df=mock_data, predicted_df=predicted_df)

    # Run metrics and assert outputs are non-empty dictionaries with float values
    evaluator.absolute_error(average_by_day=True)
