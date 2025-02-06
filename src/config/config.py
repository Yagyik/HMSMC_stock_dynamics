# project/config/config.py
# -------------------------
# This configuration file consolidates hyperparameters and toggles
# for data cleaning, wavelet usage, gradient monitoring, and logging.

class Config:
    # -----------------------------
    # Data & Source Settings
    # -----------------------------
    DATA_PATH = "data/dataset.csv"   # Path to your data file
    SOURCE_TYPE = "csv"             # "csv", "yfinance", "tiingo", "compressed"

    # Large Sequence Handling (Chunking)
    LARGE_SEQ_CHUNK_SIZE = None      # e.g., 100000 or None for smaller data

    # -----------------------------
    # Data Cleaning
    # -----------------------------
    PERFORM_CLEANING = True          # Toggle data cleaning
    CLEANING_STRATEGY = "ffill"      # "ffill", "bfill", "dropna", or "mean"
    SCALING_METHOD = "minmax"        # "minmax", "standard", or None
    TIMESTAMP_FREQ = None            # e.g. "D" for daily alignment if your data has a datetime index
    NORMALIZE = False                # Whether to do additional zero-mean, unit-std normalization

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    USE_FEATURE_AUGMENTATION = True  # Toggle advanced feature engineering

    # -----------------------------
    # Wavelet Decomposition
    # -----------------------------
    USE_WAVELET = False             # True/False for wavelet decomposition
    WAVELET_TYPE = "db4"            # e.g. "db4"
    WAVELET_LEVEL = 2               # Decomposition level

    # -----------------------------
    # Model Selection & Hyperparams
    # -----------------------------
    MODEL_TYPE = "LSTM"             # "LSTM" or "MZ"
    USE_ATTENTION = True            # If using LSTM, whether to enable attention

    # LSTM Hyperparams
    LSTM_INPUT_DIM = 10
    LSTM_HIDDEN_DIM = 64
    TIME_STEP = 1.0

    # MZ Model Hyperparams
    MZ_OPERATOR_DIM = 64

    # -----------------------------
    # Training Settings
    # -----------------------------
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3

    # Gradient Monitoring
    USE_GRAD_MONITORING = True      # If True, register hooks to track gradient norms, etc.

    # Logging
    USE_TRAINING_LOGGER = True      # If True, log metrics every epoch
    LOG_SAVE_DIR = "logs"           # Directory to save CSV or JSON logs
    LOG_FILE_FORMAT = "csv"         # "csv" or "json"

    # -----------------------------
    # Noise
    # -----------------------------
    NOISE_DISTRIBUTION = "normal"
    NOISE_PARAMS = {
        "mean": 0.0,
        "std": 0.1
    }

    # -----------------------------
    # Regularization
    # -----------------------------
    REG_ALPHA_SMOOTHNESS = 1e-3
    REG_GRAPH_SPARSITY = 1e-3

    # -----------------------------
    # Grid Search (example)
    # -----------------------------
    GRID_SEARCH_PARAMS = {
        "learning_rate": [1e-3, 1e-4],
        "LSTM_HIDDEN_DIM": [32, 64]
    }
