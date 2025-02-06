import matplotlib.pyplot as plt
import seaborn as sns

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