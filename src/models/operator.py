# FILE: /hierarchical_model/hierarchical_model/src/models/operator.py

# This file will include generalized operators for dynamics.

import numpy as np
import torch
import torch.nn as nn

class NeuralOperator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralOperator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class MemoryKernel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MemoryKernel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

