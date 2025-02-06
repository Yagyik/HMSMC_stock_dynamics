# project/config/config.py
# ----------------------------------------------------
# Configuration for a simple LSTM model test
# with data cleaning and feature engineering,
# but no wavelet decomposition nor attention.

class Config:
    # ------------------------------------------------
    # Data & Source Settings
    # ------------------------------------------------
    DATA_PATH = "data/dataset.csv"     # Update with your dataset path
    SOURCE_TYPE = "csv"                # "csv", "yfinance", "tiingo", or "compressed"

    # Large Sequence Handling (if you have very large data)
    LARGE_SEQ_CHUNK_SIZE = None        # None by default for smaller data

    # ------------------------------------------------
    # Data Cleaning
    # ------------------------------------------------
    PERFORM_CLEANING = True            # Enable data cleaning
    CLEANING_STRATEGY = "mean"         # "ffill", "bfill", "dropna", or "mean"
    SCALING_METHOD = "minmax"          # "minmax", "standard", or None
    TIMESTAMP_FREQ = None              # e.g., "D" to align daily data; None if not needed
    NORMALIZE = False                  # Whether to do zero-mean, unit-std after scaling

    # ------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------
    USE_FEATURE_AUGMENTATION = True    # Enable feature engineering (e.g., moving averages, RSI, etc.)

    # ------------------------------------------------
    # Wavelet Decomposition
    # ------------------------------------------------
    USE_WAVELET = False               # Disable wavelet decomposition

    # ------------------------------------------------
    # Model Selection & Hyperparams
    # ------------------------------------------------
    MODEL_TYPE = "LSTM"               # Use the LSTM model
    USE_ATTENTION = False             # **No attention** for this simple test case

    # LSTM Hyperparams
    LSTM_INPUT_DIM = 10               # Adjust based on your dataset's number of features
    LSTM_HIDDEN_DIM = 32              # Smaller hidden dimension for testing
    TIME_STEP = 1.0                   # Delta t for residual updates

    # MZ Model Hyperparams (not used here, but must exist)
    MZ_OPERATOR_DIM = 64

    # ------------------------------------------------
    # Training Settings
    # ------------------------------------------------
    BATCH_SIZE = 32
    NUM_EPOCHS = 5                    # Fewer epochs for a quick test run
    LEARNING_RATE = 1e-3

    # Gradient Monitoring
    USE_GRAD_MONITORING = True        # Optionally monitor gradients

    # Logging
    USE_TRAINING_LOGGER = True
    LOG_SAVE_DIR = "logs"
    LOG_FILE_FORMAT = "csv"

    # ------------------------------------------------
    # Noise
    # ------------------------------------------------
    NOISE_DISTRIBUTION = "normal"
    NOISE_PARAMS = {
        "mean": 0.0,
        "std": 0.01
    }

    # ------------------------------------------------
    # Regularization
    # ------------------------------------------------
    REG_ALPHA_SMOOTHNESS = 1e-3
    REG_GRAPH_SPARSITY = 1e-3

    # ------------------------------------------------
    # Grid Search (optional)
    # ------------------------------------------------
    GRID_SEARCH_PARAMS = {
        "learning_rate": [1e-3],
        "LSTM_HIDDEN_DIM": [32]
    }
