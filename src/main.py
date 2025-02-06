# FILE: hierarchical_model/src/main.py

import argparse
import os
import pandas as pd
from src.data.clean_data import DataCleaner
from src.data.load_timeseries_data import get_time_series_dataloader
from src.data.load_metadata import get_metadata_dataloader
from src.data.embed_metadata import MetadataEmbedder
from src.training.train import train_model
from src.models.lstm import LSTMModel
import torch
import torch.nn as nn
import torch.optim as optim

def fetch_data(source, ticker, start_date, end_date, api_key=None):
    if source == 'yfinance':
        from src.data.get_yfinance import fetch_stock_data
        return fetch_stock_data(ticker, start_date, end_date)
    elif source == 'tiingo':
        from src.data.get_tiingo import fetch_tiingo_data
        return fetch_tiingo_data(ticker, start_date, end_date, api_key)
    else:
        raise ValueError("Unsupported data source. Choose 'yfinance' or 'tiingo'.")

def main(args):
    # Step 1: Fetch data
    data = fetch_data(args.source, args.ticker, args.start_date, args.end_date, args.api_key)
    data.to_csv(os.path.join(args.data_dir, 'raw_data.csv'))

    # Step 2: Clean data
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_dataset(data)
    cleaned_data.to_csv(os.path.join(args.data_dir, 'cleaned_data.csv'))

    # Step 3: Load and embed metadata
    text_metadata = ["Example text metadata"] * len(cleaned_data)  # Replace with actual text metadata
    numerical_metadata = cleaned_data[['Open', 'High', 'Low', 'Close', 'Volume']].values  # Example numerical features
    embedder = MetadataEmbedder()
    text_embeddings = embedder.embed_text(text_metadata)
    numerical_embeddings = embedder.normalize_numerical(numerical_metadata)
    combined_embeddings = embedder.combine_features(text_embeddings, numerical_embeddings)
    metadata_loader = get_metadata_dataloader(text_metadata, numerical_metadata, embedder, args.batch_size)

    # Step 4: Prepare time series data loader
    time_series_loader = get_time_series_dataloader(cleaned_data, args.window_size, args.batch_size)

    # Step 5: Train model
    model = LSTMModel(input_size=combined_embeddings.shape[1], hidden_size=50, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_model(model, time_series_loader, criterion, optimizer, args.num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hierarchical Multi-Scale Metadata-Coupled Stock Dynamics')
    parser.add_argument('--source', type=str, required=True, choices=['yfinance', 'tiingo'], help='Data source')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, required=True, help='Start date for data fetching')
    parser.add_argument('--end_date', type=str, required=True, help='End date for data fetching')
    parser.add_argument('--api_key', type=str, help='API key for Tiingo (if using Tiingo)')
    parser.add_argument('--data_dir', type=str, default='HMSMC_stock_dynamics/data/processed', help='Directory to save processed data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loaders')
    parser.add_argument('--window_size', type=int, default=60, help='Window size for time series data')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    args = parser.parse_args()
    main(args)