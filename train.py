# project/experiments/train.py
# ---------------------------
import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

# PYTHONPATH="$PYTHONPATH:$HOME/Dropbox/finance_trials/HMSMC_stock_dynamics/HMSMC_stock_dynamics"
# sys.path.append('/home/yagyik/Dropbox/finance_trials/HMSMC_stock_dynamics')
# import HMSMC_stock_dynamics.src as hmsmc
# sys.path.append(os.path.abspath(os.path.join('..', 'src')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append('~/HMSMC_repo_and_aux/HMSMC_stock_dynamics')

# print(sys.path)
# from HMSMC_stock_dynamics.src.config.config_simple import Config
from HMSMC_stock_dynamics.src.config.config_tiingo_extensive import Config
from HMSMC_stock_dynamics.src.data.data_loader import TimeSeriesDataset
from HMSMC_stock_dynamics.src.data.clean_data import DataCleaner
from HMSMC_stock_dynamics.src.data.feature_engineering import engineer_features
from HMSMC_stock_dynamics.src.data.wavelet_decomp import wavelet_transform_df, inverse_wavelet_transform_df, identity_transform_df, identity_inverse_transform_df

from HMSMC_stock_dynamics.src.models.losses import *
from HMSMC_stock_dynamics.src.models.residual_noise import sample_noise
from HMSMC_stock_dynamics.src.utils.gradient_monitoring import GradientMonitor
from HMSMC_stock_dynamics.src.utils.logging import TrainingLogger
from HMSMC_stock_dynamics.src.utils.visualization import plot_training_curves
from HMSMC_stock_dynamics.src.utils.misc import set_random_seed

def train_model(config,args):
    """
    Main training function using the pipeline steps:
      - Gradient monitoring (optional)
      - Logging (optional)
      - Data loading -> cleaning -> feature engineering -> wavelet decomposition
      - Train-test split
      - Dataset creation
      - Model training
    """

    # 1) Set random seed for reproducibility (optional)
    set_random_seed(42)

    # 5) Load raw data as a DataFrame (via static method) from disk
    # df = TimeSeriesDataset.load_file_to_df(file_path=config.DATA_PATH,
    #                                         source=config.SOURCE_TYPE)
    df = TimeSeriesDataset.load_file_to_df(file_path=args.train_data_path,
                                            source=config.SOURCE_TYPE)
    print(f"Initial DataFrame shape: {df.shape}")

    spam_df = TimeSeriesDataset.load_file_to_df(file_path=args.test_data_path,
                                            source=config.SOURCE_TYPE)
    
    # Keep only columns that are present in both df and spam_df
    common_columns = df.columns.intersection(spam_df.columns)
    df = df[common_columns]
    spam_df = spam_df[common_columns]

    # 6) Data Cleaning
    if getattr(config, "PERFORM_CLEANING", False):
        cleaner = DataCleaner(
            strategy=config.CLEANING_STRATEGY,
            scaling=config.SCALING_METHOD,
            freq=config.TIMESTAMP_FREQ,
            normalize=config.NORMALIZE
        )
        df = cleaner.clean_data(df)
        print(f"After cleaning, DataFrame shape: {df.shape}")

    # 7) Feature Engineering
    if getattr(config, "USE_FEATURE_AUGMENTATION", False):
        df = engineer_features(df)
        print(f"After feature engineering, DataFrame shape: {df.shape}")

    # 8) Wavelet Decomposition
    if config.USE_WAVELET:
        df = wavelet_transform_df(df, wavelet_type=config.WAVELET_TYPE, level=config.WAVELET_LEVEL)
    else:
        df = identity_transform_df(df)
        # If you need to revert later, use wavelet_reconstruct(...) 

    
   # 9) Train-test split
    total_len = len(df)
    train_size = int(total_len * 0.8)  # 80% train, 20% test
    df_train = df.iloc[:train_size].copy()
    df_test  = df.iloc[train_size:].copy()

    print(f"Train set shape: {df_train.shape}, Test set shape: {df_test.shape}")

    # 10) Build TimeSeriesDataset from the DataFrames
    seq_len = 10
    chunk_size = config.LARGE_SEQ_CHUNK_SIZE  # can be None or an int

    train_dataset = TimeSeriesDataset(
        data_frame=df_train,
        seq_len=seq_len,
        chunk_size=chunk_size
    )
    test_dataset = TimeSeriesDataset(
        data_frame=df_test,
        seq_len=seq_len,
        chunk_size=chunk_size
    )

    # Update config.LSTM_INPUT_DIM based on the shape of df data columns
    print(train_dataset.numeric_data.shape)
    config.LSTM_INPUT_DIM = train_dataset.numeric_data.shape[1]



    # 2) Instantiate or import the model based on config.MODEL_TYPE
    if config.MODEL_TYPE == "LSTM":
        from HMSMC_stock_dynamics.src.models.lstm_model import LSTMWithAttention
        model = LSTMWithAttention(config)
    elif config.MODEL_TYPE == "MZ":
        from HMSMC_stock_dynamics.src.models.mz_model import MZModel
        model = MZModel(config)
    else:
        raise ValueError(f"Unsupported model type: {config.MODEL_TYPE}")
    
    # 3) Gradient Monitoring
    grad_monitor = None
    if config.USE_GRAD_MONITORING:
        grad_monitor = GradientMonitor()
        grad_monitor.register_hooks(model)

    # 4) Training Logger
    # training_logger = True
    # config.USE_TRAINING_LOGGER = True
    if config.USE_TRAINING_LOGGER:
        os.makedirs(config.LOG_SAVE_DIR, exist_ok=True)
        training_logger = TrainingLogger(save_dir=config.LOG_SAVE_DIR)


    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Train Loader: {len(train_loader)} batches, Test Loader: {len(test_loader)} batches")

    # 11) Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_model = None
    best_test_loss = float("inf")

    # 12) Training Loop
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        # Reset gradient monitor stats for this epoch
        if grad_monitor:
            grad_monitor.reset()

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.transpose(0, 1)  # shape => (seq_len, batch_size, num_features)
            optimizer.zero_grad()

            # Forward pass
            X_pred, _ = model(X_batch)

            # Residual noise
            noise = sample_noise(X_pred.shape, config)
            # noise = 0
            X_pred_noisy = X_pred + noise

            # Compute forecasting loss
            loss_forecast = forecasting_loss(X_pred_noisy, Y_batch)

            # Add optional L1/L2 reg
            reg_loss = l1_regularizer(model.parameters(), weight=1e-5)
            reg_loss += l2_regularizer(model.parameters(), weight=1e-5)
            # reg_loss = 0

            total_loss = loss_forecast + reg_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        # Average training loss
        avg_train_loss = epoch_loss / len(train_loader)

        # Evaluate on test set
        model.eval()
        test_loss_sum = 0.0
        with torch.no_grad():
            for X_batch_test, Y_batch_test in test_loader:
                X_batch_test = X_batch_test.transpose(0, 1)
                X_pred_test, _ = model(X_batch_test)
                test_loss_sum += forecasting_loss(X_pred_test, Y_batch_test).item()
        avg_test_loss = test_loss_sum / len(test_loader) if len(test_loader) > 0 else 0.0
        # print(X_batch_test.shape,X_pred_test.shape)
        print(f"[Epoch {epoch}/{config.NUM_EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.6f}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model = model.state_dict()
            print(f"Best model saved with test loss: {best_test_loss:.4f}")
        # Log metrics
        if training_logger:
            metrics_dict = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "test_loss": avg_test_loss,
                "data": X_batch_test.cpu().numpy()[0,0,0],
                "pred": X_pred_test.cpu().numpy()[0,0],

            }
            # Optionally add gradient stats
            if grad_monitor:
                # e.g. store average grad_norm for all parameters:
                grad_info = grad_monitor.get_stats()
                if len(grad_info) > 0:
                    avg_grad_norm = sum(s["grad_norm"] for s in grad_info) / len(grad_info)
                    metrics_dict["avg_grad_norm"] = avg_grad_norm

            training_logger.log_metrics(metrics_dict,step=epoch)

    # Save logs
    if training_logger:
        training_logger.save_logs("training_run")
        training_logger.close()

    # Optionally, plot curves if desired:
    # from utils.visualization import plot_training_curves
    # logs = training_logger.get_logs()
    # plot_training_curves(logs, x_key="epoch", y_keys=["train_loss", "test_loss"], title="Training Curves")

    # save the best model to a file
    best_model_path = os.path.join(config.LOG_SAVE_DIR,
                                    "best_model.pth")
    torch.save(best_model, best_model_path)
    print(f"Best model saved to {best_model_path}")
    # Optionally, save the model architecture
    # torch.save(model, os.path.join(config.LOG_SAVE_DIR, "model_architecture.pth"))

    return model

if __name__ == "__main__":
    print(sys.path)
    config = Config()
    # config.DATA_PATH = "datasets/raw/yfinance_test.csv"
    config.DATA_PATH = "/home/yagyik/Dropbox/finance_trials/HMSMC_repo_and_aux/tiingo_downloader/dataset_extensive/NYSE_1_Jan_2016_to_1_Jan_2024_1min/"
    config.SOURCE_TYPE = "compressed"
    config.PERFORM_CLEANING = True
    config.LOG_SAVE_DIR = "logs"
    config.NUM_EPOCHS = 200
    config.LEARNING_RATE = 1e-5
    config.USE_TRAINING_LOGGER = True



    parser = argparse.ArgumentParser(description="Evaluate LSTM model predictions.")
    # parser.add_argument("--model_path", type=str, required=True, help="Path to the best model file.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the train dataset file.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset file.")

    
    final_model = train_model(config,args=parser.parse_args())

