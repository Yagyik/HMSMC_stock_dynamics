import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

class DataCleaner:
    def __init__(self, missing_method="ffill", outlier_method="zscore", normalization=True):
        """
        Initializes the DataCleaner with methods for handling missing data and outliers.
        
        Args:
            missing_method (str): Method to handle missing data. Options: "ffill", "bfill", "interpolate", "drop".
            outlier_method (str): Method to handle outliers. Options: "zscore", "winsorize", None.
            normalization (bool): Whether to normalize numerical data.
        """
        self.missing_method = missing_method
        self.outlier_method = outlier_method
        self.normalization = normalization
        self.scaler = StandardScaler() if normalization else None

    def clean_missing_data(self, df):
        """Handles missing data according to the selected method."""
        if self.missing_method == "ffill":
            df = df.fillna(method="ffill").fillna(method="bfill")  # Forward then backward fill
        elif self.missing_method == "bfill":
            df = df.fillna(method="bfill")
        elif self.missing_method == "interpolate":
            df = df.interpolate()
        elif self.missing_method == "drop":
            df = df.dropna()
        return df

    def detect_outliers(self, df, threshold=3.0):
        """Detects and handles outliers based on z-score or winsorization."""
        if self.outlier_method == "zscore":
            return df[(np.abs(zscore(df)) < threshold).all(axis=1)]
        elif self.outlier_method == "winsorize":
            df = df.clip(lower=df.quantile(0.05), upper=df.quantile(0.95), axis=1)
        return df

    def normalize_data(self, df):
        """Applies standard scaling to numerical features."""
        if self.normalization:
            df[:] = self.scaler.fit_transform(df)
        return df

    def align_timestamps(self, dfs):
        """Ensures all time-series data have aligned timestamps."""
        common_index = set(dfs[0].index)
        for df in dfs[1:]:
            common_index &= set(df.index)
        common_index = sorted(common_index)
        return [df.loc[common_index] for df in dfs]

    def clean_text_metadata(self, texts):
        """Processes textual metadata for embedding (lowercasing, removing stopwords, etc.)."""
        import re
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        cleaned_texts = []
        for text in texts:
            text = text.lower()
            text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
            words = word_tokenize(text)
            words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
            cleaned_texts.append(" ".join(words))

        return cleaned_texts

    def clean_dataset(self, df, is_time_series=True):
        """Performs full cleaning on the dataset."""
        if is_time_series:
            df = self.clean_missing_data(df)
            df = self.detect_outliers(df)
        if self.normalization:
            df = self.normalize_data(df)
        return df

    def clean_dataset(self, data):
        """Performs full cleaning on the dataset."""
        data = data.dropna()
        data = data[data['Volume'] > 0]
        return data

    def normalize_data(self, data):
        """Applies standard scaling to numerical features."""
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def zscore_data(self, data):
        """Applies z-score normalization to numerical features."""
        return zscore(data)


