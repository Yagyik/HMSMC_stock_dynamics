# project/data/feature_engineering.py
# ---------------------------------------------------
# This module provides functions to transform or augment a Pandas DataFrame with additional
# financial features (technical indicators, etc.). Each function takes a DataFrame, calculates
# the desired indicators, and returns the augmented DataFrame.

import pandas as pd
import numpy as np

def add_moving_average(df, column="Close", window=5, prefix="ma"):
    """
    Add a simple moving average feature to the dataframe for a specified column and window.
    
    Args:
        df (DataFrame): The original DataFrame.
        column (str): The column to use for calculating the moving average.
        window (int): The rolling window size.
        prefix (str): Prefix for the new feature name.
    
    Returns:
        df (DataFrame): Augmented DataFrame with the new feature.
    """
    feature_name = f"{prefix}_{window}"
    df[feature_name] = df[column].rolling(window=window).mean()
    return df

def add_exponential_moving_average(df, column="Close", span=12, prefix="ema"):
    """
    Add an exponential moving average feature.
    
    Args:
        df (DataFrame): The original DataFrame.
        column (str): The column to use (often 'Close').
        span (int): Span parameter for ewm.
        prefix (str): Prefix for the new feature name.
        
    Returns:
        df (DataFrame): Augmented DataFrame with the new feature.
    """
    feature_name = f"{prefix}_{span}"
    df[feature_name] = df[column].ewm(span=span, adjust=False).mean()
    return df

def add_rsi(df, column="Close", period=14, prefix="rsi"):
    """
    Add the Relative Strength Index (RSI) as a feature.
    
    RSI = 100 - (100 / (1 + RS)), where
    RS = average_gain / average_loss over 'period' lookback.
    
    Args:
        df (DataFrame): The original DataFrame.
        column (str): The column used for RSI calculation.
        period (int): Lookback period.
        prefix (str): Feature prefix.
    
    Returns:
        df (DataFrame): DataFrame with 'RSI' feature.
    """
    delta = df[column].diff(1)
    gain = (delta.clip(lower=0)).abs()
    loss = (-delta.clip(upper=0)).abs()
    
    # Calculate rolling mean of gains/losses
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)  # Avoid division by zero
    rsi = 100.0 - (100.0 / (1.0 + rs))
    feature_name = f"{prefix}_{period}"
    df[feature_name] = rsi
    return df

def add_dummy_features(df):
    """
    Add any number of example/dummy features for demonstration.
    """
    # Example: daily returns
    if "Close" in df.columns:
        df["returns"] = df["Close"].pct_change()
    return df

def engineer_features(df):
    """
    Master function to apply a series of transformations or additions to the DataFrame.
    This orchestrates multiple smaller subroutines to create a comprehensive feature set.
    
    Returns the augmented DataFrame (with NaNs at the beginning for any rolling calculations).
    Consider dropping those NaNs after feature creation.
    """
    df = add_dummy_features(df)
    if "Close" in df.columns:
        df = add_moving_average(df, column="Close", window=5, prefix="ma")
        df = add_exponential_moving_average(df, column="Close", span=12, prefix="ema")
        df = add_rsi(df, column="Close", period=14, prefix="rsi")
        # Add more custom features as needed
    
    # Optionally, drop NaN rows created by rolling calculations
    df.dropna(inplace=True)
    return df
