import torch
import torch.nn as nn

class LSTMDrift(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMDrift, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMMemoryKernel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMMemoryKernel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel:
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the last time step
        return out

    def train_model(self, train_loader, criterion, optimizer, num_epochs):
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def evaluate_model(self, test_loader,criterion):
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.forward(inputs)
                total_loss += criterion(outputs, labels).item()
        return total_loss / len(test_loader)