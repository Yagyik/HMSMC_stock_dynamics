# project/data/data_loader.py
# ---------------------------
# A consolidated TimeSeriesDataset class that:
#   1) Includes static methods to load a file into a DataFrame.
#   2) Includes a classmethod (load_as_dataframe) to retrieve only the DataFrame.
#   3) Allows direct instantiation from a file_path or from an existing DataFrame.
#   4) Stores numeric data in self.data as float32, and retains date/time info in self.dates.

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self,
                 file_path=None,
                 source=None,
                 data_frame=None,
                 seq_len=10,
                 chunk_size=None,
                 skip_dates=False):
        """
        A PyTorch Dataset for time-series prediction using a sliding window approach.

        There are two primary ways to instantiate:
            1) Provide file_path + source, so the dataset will load the file internally.
            2) Provide an existing data_frame that has already been loaded/cleaned/processed.

        Args:
            file_path (str, optional): Path to the data file on disk.
            source (str, optional): Type of data source ("csv", "yfinance", "tiingo", "compressed").
            data_frame (DataFrame, optional): A Pandas DataFrame already in memory (e.g. after feature engineering).
            seq_len (int): Length of each sequence for the sliding window.
            chunk_size (int, optional): If not None, build sequences in chunks for large data.
            skip_dates (bool): If True, we do not attempt to store date/time info in self.dates.
                               This can be useful if the data does not contain any date or you do not need it.
        """
        super().__init__()
        self.seq_len = seq_len
        self.chunk_size = chunk_size

        # 1) If data_frame is not provided, load from file
        if data_frame is None:
            if file_path is None or source is None:
                raise ValueError("Must provide either data_frame or (file_path + source).")
            # Load the file into a DataFrame
            df = self.load_file_to_df(file_path, source)
        else:
            # We already have a DataFrame
            df = data_frame

        # 2) Process the DataFrame to separate out date/time columns from numeric columns
        self.dates = None
        if not skip_dates:
            if isinstance(df.index, pd.DatetimeIndex):
                # If the DataFrame index is datetime, we store it
                self.dates = df.index
            else:
                # If there's a date/datetime column (e.g. 'Date')
                possible_date_cols = [col for col in df.columns if 'date' in col.lower()]
                if possible_date_cols:
                    date_col = possible_date_cols[0]
                    try:
                        # Attempt to parse as datetime
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        self.dates = df[date_col]
                    except Exception:
                        pass

        # 3) Extract numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.columns = list(numeric_cols)  # store column names
        self.numeric_data = df[numeric_cols].values.astype(np.float32)

        # 4) Create sequences & targets
        self.sequences, self.targets = self._create_sequences(self.numeric_data, seq_len, chunk_size)

    @staticmethod
    def load_file_to_df(file_path, source):
        """
        Static method to load data from disk into a DataFrame, depending on the source type.
        e.g., "csv", "yfinance", "tiingo", "compressed".
        """
        ext = os.path.splitext(file_path)[1].lower()

        if source in ["csv", "yfinance", "tiingo"]:
            df = pd.read_csv(file_path)
        elif source == "compressed":
            if ext == ".parquet":
                df = pd.read_parquet(file_path)
            elif ext in [".h5", ".hdf5"]:
                df = pd.read_hdf(file_path, key="data")
            else:
                raise ValueError(f"Unsupported compressed file format: {ext}")
        else:
            raise ValueError(f"Unsupported source type: {source}")

        return df

    @classmethod
    def load_as_dataframe(cls, file_path, source):
        """
        Classmethod that returns only the DataFrame loaded from disk,
        without constructing the TimeSeriesDataset sequences. Useful
        if you want to do cleaning, feature engineering, etc. yourself
        before creating the dataset object.
        """
        return cls.load_file_to_df(file_path, source)

    @staticmethod
    def _create_sequences(data, seq_len, chunk_size=None):
        """
        Build sequences and targets via a sliding window.
        If chunk_size is provided, we do it in smaller chunks for large data.
        """
        num_samples = len(data)
        if chunk_size is None:
            # Single pass
            sequences = []
            targets = []
            for i in range(num_samples - seq_len):
                seq = data[i : i + seq_len]
                tgt = data[i + seq_len]
                sequences.append(seq)
                targets.append(tgt)
            return np.array(sequences), np.array(targets)
        else:
            # Chunk-based approach
            seq_list, tgt_list = [], []
            start_idx = 0
            while start_idx < num_samples - seq_len:
                end_idx = min(start_idx + chunk_size, num_samples - seq_len)
                for i in range(start_idx, end_idx):
                    seq = data[i : i + seq_len]
                    tgt = data[i + seq_len]
                    seq_list.append(seq)
                    tgt_list.append(tgt)
                start_idx = end_idx
            return np.array(seq_list), np.array(tgt_list)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        tgt = self.targets[index]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)
