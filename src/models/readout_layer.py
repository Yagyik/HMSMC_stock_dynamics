# project/models/readout_layer.py
# -------------------------------
# This module implements the read-out layer for the forecasting models.
# The purpose of the read-out layer is to map the concatenated latent representation [H; C]
# (where H is the hidden state and C is the effective cell state) to a forecasted state update.
#
# Key features:
# - The layer decomposes its weight matrix into two parts:
#     • W_self: Intended to capture self-interactions (diagonal or block-diagonal structure).
#     • W_cross: Intended to capture cross-interactions (off-diagonal contributions).
# - A bias term is added to complete the linear mapping.
# - Extraction hooks (via the get_weights() method) allow downstream analysis of W_self and W_cross.
#
# This modular design enables detailed investigation into how much each model relies on a variable’s
# own past (self) versus influences from other variables (cross).
#
# Usage:
#   - Instantiate the ReadOutLayer with the appropriate input and output dimensions.
#   - In the forward pass, pass the concatenated tensor [H; C] (shape: (batch_size, input_dim)).
#   - Use the get_weights() method to retrieve W_self and W_cross after training for analysis.

import torch
import numpy as np
import os
import torch.nn as nn

class ReadOutLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize the read-out layer.
        
        Args:
            input_dim (int): Dimensionality of the concatenated latent representation [H; C].
            output_dim (int): Dimensionality of the output state (should match the state dimension).
        """
        super(ReadOutLayer, self).__init__()
        # Weight matrix for self-interactions (intended to be diagonal or block-diagonal)
        self.W_self = nn.Parameter(torch.randn(output_dim, input_dim))
        # Weight matrix for cross-interactions (captures off-diagonal influences)
        self.W_cross = nn.Parameter(torch.randn(output_dim, input_dim))
        # Bias term for the linear mapping
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Optionally, you can enforce a diagonal structure on W_self.
        # For example, by registering a mask:
        # self.register_buffer("mask_self", torch.eye(output_dim, input_dim))
        # And then using: self.W_self = self.W_self * self.mask_self in the forward pass.
    
    def forward(self, x):
        """
        Forward pass of the read-out layer.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim) representing [H; C].
        
        Returns:
            output (Tensor): The computed forecast update, shape (batch_size, output_dim).
        """
        # Compute contribution from self-interactions
        y_self = torch.matmul(x, self.W_self.t())
        # Compute contribution from cross-interactions
        y_cross = torch.matmul(x, self.W_cross.t())
        # Sum contributions and add bias term
        output = y_self + y_cross + self.bias
        return output

    def get_weights(self):
        """
        Retrieve the weight matrices for analysis.
        
        Returns:
            W_self, W_cross (ndarray, ndarray): Numpy arrays for the self and cross weight matrices.
        """
        return self.W_self.detach().cpu().numpy(),self.W_cross.detach().cpu().numpy()

if __name__ == "__main__":
    # Example usage:
    # Assume latent representation [H; C] has dimension 128 and the output state dimension is 10.
    input_dim = 128
    output_dim = 10
    dummy_input = torch.randn(4, input_dim)  # Simulated batch of 4 samples.
    
    readout = ReadOutLayer(input_dim, output_dim)
    output = readout(dummy_input)
    # print("Output shape:", output.shape)
    
    # Extract and display the weight matrices.
    W_self, W_cross = readout.get_weights()
    # print("W_self shape:", W_self.shape)
    # print("W_cross shape:", W_cross.shape)

    # Define output directory for CSV files
    output_dir = "/home/yagyik/Dropbox/finance_trials/HMSMC_repo_and_aux/HMSMC_stock_dynamics/weights"
    os.makedirs(output_dir, exist_ok=True)

    # Write W_self to a CSV file
    W_self_path = os.path.join(output_dir, "W_self.csv")
    np.savetxt(W_self_path, W_self, delimiter=",")
    print(f"W_self saved to {W_self_path}")

    # Write W_cross to a CSV file
    W_cross_path = os.path.join(output_dir, "W_cross.csv")
    np.savetxt(W_cross_path, W_cross, delimiter=",")
    print(f"W_cross saved to {W_cross_path}")
