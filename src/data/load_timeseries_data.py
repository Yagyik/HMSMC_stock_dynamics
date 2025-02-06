import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, target_column='Close'):
        """
        Args:
            data (pd.DataFrame): Time-series data.
            window_size (int): Length of the input sequence.
            target_column (str): Column to use as the target.
        """
        self.data = data
        self.window_size = window_size
        self.target_column = target_column

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        """
        Returns:
            inputs (torch.Tensor): Input sequence of size (window_size, features).
            target (torch.Tensor): Target value at the next time step.
        """
        inputs = self.data.iloc[idx:idx + self.window_size].drop(columns=[self.target_column]).values
        target = self.data.iloc[idx + self.window_size][self.target_column]
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def get_time_series_dataloader(data, window_size, batch_size):
    dataset = TimeSeriesDataset(data, window_size)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
