import os
import numpy as np
import pickle
import argparse
import torch
import matplotlib.pyplot as plt
from HMSMC_stock_dynamics.src.data.data_loader import TimeSeriesDataset
from HMSMC_stock_dynamics.src.models.lstm_model import LSTMWithAttention  # Replace with the correct model class
from HMSMC_stock_dynamics.src.config.config_tiingo_extensive import Config

def load_model(model_path, config):
    """
    Load the best model from the specified path.
    """
    # with open(model_path, "rb") as f:
    #     state_dict = pickle.load(f)
    state_dict = torch.load(model_path) #['state_dict']
    print(state_dict)
    model = LSTMWithAttention(config)  # Replace with the correct model class
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    return model

def load_dataset(file_path, seq_len, chunk_size=None):
    """
    Load a dataset from the specified file path.
    """
    df = TimeSeriesDataset.load_file_to_df(file_path=file_path, source="compressed")
    dataset = TimeSeriesDataset(data_frame=df, seq_len=seq_len, chunk_size=chunk_size)
    return dataset

def rolling_forecast(model, dataset, seq_len, start_index=0):
    """
    Perform rolling forecast origin predictions and return datetime info.
    """
    predictions = np.empty((0, model.output_dim))  # Initialize empty numpy array for predictions
    actuals = np.empty((0, model.output_dim))  # Initialize empty numpy array for actuals
    datetimes = []  # List to store datetime indices
    print(start_index,len(dataset),seq_len)
    for i in range(start_index, len(dataset) - seq_len,10):
        input_seq, target_seq = dataset[i]  # Assuming dataset[i] also returns datetime info
        
        # print(datetime)
        print(input_seq.numpy().shape,target_seq.numpy().shape)
        input_seq = input_seq.unsqueeze(1)  # Add batch dimension
        with torch.no_grad():
            pred_seq, _ = model(input_seq)
        spam_pred = pred_seq.squeeze(0).numpy()
        spam_target = target_seq.numpy()
        # print("norm error",np.linalg.norm(spam_pred - spam_target))
        predictions = np.vstack((predictions, spam_pred))  # Stack predictions vertically
        actuals = np.vstack((actuals, spam_target))  # Stack actuals vertically
        # print(predictions.shape)
        # print(spam_pred[:10])
        # print(pred_seq.numpy()[:10])
        # print(dataset.dates[i])
        # datetimes.append(datetime)  # Append the last datetime of the sequence
    # datetimes = [str(dt) for dt in datetimes]
    datetimes = dataset.dates[start_index:len(dataset) - seq_len:10]
    print(len(datetimes))
    return predictions, actuals, datetimes


def multi_predict(model,dataset,input_len,predict_len,start_index=0):
    predictions = np.empty((0, model.output_dim))  # Initialize empty numpy array for predictions
    actuals = np.empty((0, model.output_dim))  # Initialize empty numpy array for actuals
    datetimes = []  # List to store datetime indices
    print(start_index,len(dataset),input_len,predict_len)
    chunk_len = input_len + predict_len
    for i in range(start_index, len(dataset) - chunk_len,input_len+chunk_len):
        target_seq = []
        for j in range(chunk_len):
            _ , target_spot = dataset[i+j]
            target_seq.append(target_spot)
        input_seq, target_seq = dataset[i]  # Assuming dataset[i] also returns datetime info
        
        # print(datetime)
        # print(input_seq.numpy().shape,target_seq.numpy().shape)
        input_seq = input_seq.unsqueeze(1)  # Add batch dimension
        with torch.no_grad():
            pred_seq, _ = model(input_seq)
        spam_pred = pred_seq.squeeze(0).numpy()
        spam_target = target_seq.numpy()
        # print("norm error",np.linalg.norm(spam_pred - spam_target))
        predictions = np.vstack((predictions, spam_pred))  # Stack predictions vertically
        actuals = np.vstack((actuals, spam_target))  # Stack actuals vertically
        # print(predictions.shape)
        # print(spam_pred[:10])
        # print(pred_seq.numpy()[:10])
        # print(dataset.dates[i])
        # datetimes.append(datetime)  # Append the last datetime of the sequence
    # datetimes = [str(dt) for dt in datetimes]
    datetimes = dataset.dates[start_index:len(dataset) - seq_len]
    print(datetimes)
    return predictions, actuals, datetimes

def plot_predictions(predictions, actuals, datetimes, columns, seq_len, output_dir):
    """
    Plot predictions vs. actuals for the specified columns against datetime.
    """
    os.makedirs(output_dir, exist_ok=True)
    # print(actuals.shape,predictions.shape)
    # print(actuals[0,:],actuals[-1,:])
    # print(predictions[0,:],predictions[-1,:])
    for col_idx, col_name in columns:
        # Plot predictions vs actuals
        plt.figure(figsize=(15, 12))
        print(col_idx,col_name)
        plt.plot(datetimes, actuals[:,col_idx], label=f"Actual ({col_name})", linewidth=2)
        plt.plot(datetimes, predictions[:,col_idx], '--', label="Prediction")
        plt.title(f"Predictions vs Actuals for {col_name} (Seq Len: {seq_len})",fontsize=30)
        plt.xlabel("Datetime",fontsize=20)
        plt.ylabel("Value",fontsize=20)
        plt.xticks(datetimes[::50], rotation=60,fontsize=15)  # Show every 50th entry and rotate the x-axis labels by 90 degrees
        plt.yticks(fontsize=20)
        plt.legend(fontsize=30)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"predictions_{col_name}_seq{seq_len}.png"))
        plt.close()

        # Plot difference between actuals and predictions
        plt.figure(figsize=(15, 12))
        differences = actuals[:,col_idx] - predictions[:,col_idx]
        plt.plot(datetimes, differences, label=f"Difference (Actual - Prediction) ({col_name})", color="red")
        plt.title(f"Difference between Actuals and Predictions for {col_name} (Seq Len: {seq_len})")
        plt.xlabel("Datetime",fontsize=20)
        plt.xticks(datetimes[::50], rotation=60,fontsize=15)  # Show every 50th entry and rotate the x-axis labels by 90 degrees
        plt.ylabel("Difference",fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=30)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"difference_{col_name}_seq{seq_len}.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate LSTM model predictions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the best model file.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the train dataset file.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset file.")
    parser.add_argument("--seq_lens", type=int, nargs="+", required=True, help="Sequence lengths for predictions.")
    parser.add_argument("--columns", type=str, nargs="+", required=True, help="Columns to visualize.")
    parser.add_argument("--output_dir", type=str, default="prediction_outputs", help="Directory to save plots.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for rolling forecast origin.")
    args = parser.parse_args()

    # Load config
    config = Config()

    

    # Load datasets
    df = TimeSeriesDataset.load_file_to_df(file_path=args.test_data_path,
                                            source=config.SOURCE_TYPE)
    
    spam_df = TimeSeriesDataset.load_file_to_df(file_path=args.train_data_path,
                                                source=config.SOURCE_TYPE)
    print(f"Train dataset shape: {spam_df.shape}")
    print(f"Test dataset shape: {df.shape}")

    # train_dataset = load_dataset(args.train_data_path, seq_len=max(args.seq_lens))
    # test_dataset = load_dataset(args.test_data_path, seq_len=max(args.seq_lens))

    total_len = len(df)
    train_size = int(total_len * 0.1)  # 80% train, 20% test
    df_train = df.iloc[:train_size].copy()
    df_test  = df.iloc[train_size:].copy()



    # Generate predictions for each sequence length
    for seq_len in args.seq_lens:
        # chunk_size = config.LARGE_SEQ_CHUNK_SIZE  # can be None or an int
        chunk_size = total_len - train_size - seq_len


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

        missing_columns = set(train_dataset.columns) - set(test_dataset.columns)
        print(f"Missing columns in test dataset: {missing_columns}")

        # Update config.LSTM_INPUT_DIM based on the shape of df data columns
        print(train_dataset.numeric_data.shape)
        config.LSTM_INPUT_DIM = train_dataset.numeric_data.shape[1]
        print(f"Generating predictions for sequence length: {seq_len}")
        # print(test_dataset)

        # Load model
        model = load_model(args.model_path, config)

        predictions, actuals, datetimes = rolling_forecast(model, test_dataset, seq_len,
                                                            start_index=args.start_index)

        col_list = [int(x) for x in args.columns]
        print(args.columns,col_list)


        print_cols = [(i,test_dataset.columns[i]) for i in col_list]
        print(print_cols)

        plot_predictions(predictions, actuals, datetimes, 
                         print_cols, seq_len, args.output_dir)

if __name__ == "__main__":
    main()