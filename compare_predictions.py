# FILE: compare_predictions.py

import os
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from HMSMC_stock_dynamics.src.models.lstm import LSTMModel
from HMSMC_stock_dynamics.src.data.clean_data import DataCleaner
# from HMSMC_stock_dynamics.src.data.embed_metadata import MetadataEmbedder
from HMSMC_stock_dynamics.src.utils.visualization import plot_time_series

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, selected_column):
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_dataset(data)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cleaned_data[[selected_column]])
    return scaled_data, scaler

def create_dataloader(data, window_size, batch_size):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
    sequences = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def generate_predictions(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0]
            outputs = model(inputs)
            predictions.append(outputs.numpy())
    predictions = np.concatenate(predictions, axis=0)
    return predictions

def main():
    # Load data
    data_file_path = 'datasets/processed/your_data_file.csv'  # Replace with your actual data file path
    data = load_data(data_file_path)
    
    # Preprocess data
    selected_column = 'Close'  # Replace with the column you want to predict
    scaled_data, scaler = preprocess_data(data, selected_column)
    
    # Create dataloader
    window_size = 60
    batch_size = 32
    dataloader = create_dataloader(scaled_data, window_size, batch_size)
    
    # Load model
    model_path = 'logs/best_model.pkl'
    model = load_model(model_path)
    
    # Generate predictions
    predictions = generate_predictions(model, dataloader)
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions)
    
    # Plot predictions vs actual values
    actual_values = data[selected_column].values[window_size:]
    plot_time_series(actual_values, title='Actual vs Predicted', xlabel='Time', ylabel='Value')
    plot_time_series(predictions, title='Actual vs Predicted', xlabel='Time', ylabel='Value')

if __name__ == "__main__":
    main()