# project/data/clean_data.py
# --------------------------
# This module provides a DataCleaner class to handle missing values, scaling, 
# timestamp alignment, and optional normalization for financial time series.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataCleaner:
    def __init__(self, 
                 strategy="ffill",   # Missing value handling strategy
                 scaling="minmax",   # "minmax", "standard", or None
                 freq=None,          # Frequency for aligning timestamps, e.g. "D" for daily
                 normalize=False     # Whether to normalize data to zero mean, unit variance
                ):
        """
        Args:
            strategy (str): Missing value handling strategy. 
                            Options: "ffill", "bfill", "dropna", "mean".
            scaling (str): Scaling method (applied after missing values are handled).
                           Options: "minmax", "standard", or None.
            freq (str or None): If not None, reindex to this frequency (e.g. 'D') for alignment.
            normalize (bool): Whether to normalize data to zero mean, unit variance 
                              (in addition to or instead of scaling).
                              If True, we'll store a mean/std per column. 
        """
        self.strategy = strategy
        self.scaling = scaling
        self.freq = freq
        self.normalize = normalize

        # Internal references for scaling/normalization
        self.scaler = None
        self.norm_means_ = None
        self.norm_stds_ = None

    def handle_missing(self, df):
        """
        Handle missing values in the DataFrame based on the specified strategy.
        """
        if self.strategy == "ffill":
            df = df.fillna(method="ffill").fillna(method="bfill")
        elif self.strategy == "bfill":
            df = df.fillna(method="bfill").fillna(method="ffill")
        elif self.strategy == "dropna":
            df = df.dropna()
        elif self.strategy == "mean":
            df = df.fillna(df.mean())
        else:
            print(f"Unrecognized missing strategy: {self.strategy}. No missing-value handling performed.")
        return df

    def fit_scaler(self, df):
        """
        Fit a scaler to the DataFrame's numeric columns if self.scaling is 'minmax' or 'standard'.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return

        if self.scaling == "minmax":
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[numeric_cols])
        elif self.scaling == "standard":
            self.scaler = StandardScaler()
            self.scaler.fit(df[numeric_cols])
        else:
            self.scaler = None

    def apply_scaler(self, df):
        """
        Apply the previously fitted scaler to the DataFrame. 
        """
        if self.scaler is None:
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        return df

    def align_timestamps(self, df):
        """
        Align the DataFrame to a regular frequency if self.freq is not None.
        This method assumes 'Date' or 'Datetime' is in the DataFrame or is its index.

        Steps:
          1. Ensure the index is datetime (or convert a 'Date' column).
          2. Reindex to a regular frequency, forward-filling or backward-filling as needed.
        """
        if self.freq is None:
            return df

        # Identify or convert a date/datetime index
        if df.index.dtype.kind not in ['M']:  # not a datetime type
            # Attempt to convert a 'Date' or 'Datetime' column if present:
            possible_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if len(possible_cols) > 0:
                df = df.set_index(pd.to_datetime(df[possible_cols[0]]))
            else:
                # Force reindex with integer index => might not do anything.
                print(f"No date-like column found, cannot align to freq '{self.freq}' properly.")
                return df

        # Now assume index is datetime
        df = df.sort_index()
        # Reindex to the chosen freq
        df = df.asfreq(self.freq, method='ffill')  # forward fill
        return df

    def fit_normalizer(self, df):
        """
        Compute mean and std for each numeric column, to be used in normalize_data.
        """
        if not self.normalize:
            return
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.norm_means_ = df[numeric_cols].mean()
        self.norm_stds_ = df[numeric_cols].std().replace(0, 1e-9)  # avoid division by zero

    def normalize_data(self, df):
        """
        Subtract mean and divide by std for numeric columns.
        """
        if not self.normalize or self.norm_means_ is None or self.norm_stds_ is None:
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = (df[numeric_cols] - self.norm_means_[numeric_cols]) / self.norm_stds_[numeric_cols]
        return df

    def clean_data(self, df):
        """
        Orchestrate the data cleaning steps:
          1) Align timestamps (if freq is specified).
          2) Handle missing values.
          3) Fit & apply scaler (if configured).
          4) Fit & apply optional normalization.
        """
        # 1) Align timestamps
        df = self.align_timestamps(df)



        # # 2) Handle missing values
        # df = self.handle_missing(df)

        # 3) Fit scaler on the cleaned data, then apply
        self.fit_scaler(df)
        df = self.apply_scaler(df)

        # 4) Fit and apply optional normalization
        self.fit_normalizer(df)
        df = self.normalize_data(df)

        return df
