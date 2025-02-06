import matplotlib.pyplot as plt
import seaborn as sns

# project/utils/visualization.py
# ------------------------------
# This file provides basic plotting utilities for training curves, model predictions, etc.
# It uses matplotlib. You can extend or modify for advanced visualizations with seaborn, plotly, etc.


def plot_training_curves(logs, x_key="epoch", y_keys=("train_loss", "test_loss"), title="Training Curves"):
    """
    Plot one or more metrics over time (epochs).
    
    Args:
        logs (list of dict): The training logs from the TrainingLogger, e.g. [{"epoch":1, "train_loss":0.5, ...}, ...].
        x_key (str): The key in the dictionary to use for the x-axis (often "epoch").
        y_keys (tuple or list): The metrics to plot on the y-axis.
        title (str): Plot title.
    """
    if not logs:
        print("No logs to plot.")
        return
    
    plt.figure(figsize=(8,6))
    x_values = [log[x_key] for log in logs]
    
    for y_key in y_keys:
        y_values = [log.get(y_key, None) for log in logs]
        plt.plot(x_values, y_values, label=y_key)
    
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(true_values, predicted_values, title="Model Predictions"):
    """
    Plot predicted vs. true values for a time series or regression task.
    
    Args:
        true_values (list or ndarray): True target values.
        predicted_values (list or ndarray): Model predictions.
        title (str): Plot title.
    """
    plt.figure(figsize=(8,6))
    plt.plot(true_values, label="True", marker='o')
    plt.plot(predicted_values, label="Predicted", marker='x')
    plt.title(title)
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_time_series(data, title='Time Series Data', xlabel='Time', ylabel='Value'):
    plt.figure(figsize=(12, 6))
    plt.plot(data, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def plot_correlation_matrix(data, title='Correlation Matrix'):
    plt.figure(figsize=(10, 8))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title(title)
    plt.show()

def plot_histogram(data, title='Histogram', xlabel='Value', ylabel='Frequency', bins=30):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()