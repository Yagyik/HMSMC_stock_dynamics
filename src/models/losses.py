# project/losses/regularizers.py
# ------------------------------
# This module defines various regularization functions and additional loss functions
# that can be used to constrain and guide the learning of our forecasting models.
#
# Functions included:
#   - smoothness_regularizer: Enforces smoothness in the memory kernel by penalizing
#         large differences between consecutive kernel values.
#
#   - sparsity_regularizer: Encourages sparsity in a graph (e.g., cross-interaction matrix)
#         via L1 regularization.
#
#   - l1_regularizer: Standard L1 regularization on a list of model parameters.
#
#   - l2_regularizer: Standard L2 regularization on a list of model parameters.
#
#   - msr_loss: Computes an MSR-inspired loss for a chunk of the trajectory.
#         It enforces that the discrete time derivative of the trajectory aligns with the
#         predicted dynamics (the sum of the instantaneous operator and the memory term).
#
# Each function takes input arguments appropriate for the model and returns a scalar loss term.
# These functions are intended to be added to the overall loss during training.


import torch
import torch.nn as nn
from ..data.data_loader import TimeSeriesDataset

def forecasting_loss(predictions, targets):
    """
    Compute the Mean Squared Error (MSE) loss between predictions and targets.
    
    Args:
        predictions (Tensor): Predicted values, shape (batch_size, num_features).
        targets (Tensor): Ground truth values, shape (batch_size, num_features).
        
    Returns:
        loss (Tensor): A scalar tensor representing the MSE loss.
    """
    mse_loss = nn.MSELoss()
    return mse_loss(predictions, targets)


    


def smoothness_regularizer(kernel_values, weight=1e-3):
    """
    Enforce smoothness in the memory kernel by penalizing large differences between consecutive values.
    
    Args:
        kernel_values (Tensor): Tensor of kernel values of shape (num_lags, ...) where consecutive indices
                                represent increasing lag.
        weight (float): Regularization weight.
        
    Returns:
        loss (Tensor): A scalar tensor representing the smoothness penalty.
    """
    # Compute the differences between consecutive kernel values.
    # For a 1-D kernel, kernel_values should have shape (num_lags,)
    # For multi-dimensional kernels, the difference is computed elementwise.
    diff = kernel_values[1:] - kernel_values[:-1]
    loss = weight * torch.mean(diff ** 2)
    return loss

def sparsity_regularizer(graph_matrix, weight=1e-3):
    """
    Encourage sparsity in a graph (e.g., the cross-interaction matrix) using L1 regularization.
    
    Args:
        graph_matrix (Tensor): The graph or matrix (e.g., cross-coupling matrix) for which to enforce sparsity.
        weight (float): Regularization weight.
        
    Returns:
        loss (Tensor): A scalar tensor representing the sparsity penalty.
    """
    # L1 penalty on all elements of the matrix.
    loss = weight * torch.sum(torch.abs(graph_matrix))
    return loss

def l1_regularizer(parameters, weight=1e-3):
    """
    Apply L1 regularization to a list (or iterator) of parameters.
    
    Args:
        parameters (iterable): An iterable of model parameters (e.g., from model.parameters()).
        weight (float): Regularization weight.
        
    Returns:
        loss (Tensor): A scalar tensor representing the cumulative L1 penalty.
    """
    loss = 0.0
    for param in parameters:
        loss += weight * torch.sum(torch.abs(param))
    return loss

def l2_regularizer(parameters, weight=1e-3):
    """
    Apply L2 regularization to a list (or iterator) of parameters.
    
    Args:
        parameters (iterable): An iterable of model parameters.
        weight (float): Regularization weight.
        
    Returns:
        loss (Tensor): A scalar tensor representing the cumulative L2 penalty.
    """
    loss = 0.0
    for param in parameters:
        loss += weight * torch.sum(param ** 2)
    return loss

def msr_loss(trajectory, predicted_dynamics, dt, weight=1.0):
    """
    Compute an MSR-inspired loss over a chunk of the trajectory.
    
    The loss enforces that the discrete time derivative of the trajectory approximates
    the predicted dynamics. In continuous time, the system dynamics are given by:
        dx/dt = instantaneous_operator(x) + memory_term(x)
    In discrete form, we approximate:
        (x_{t+1} - x_t) / dt ≈ predicted_dynamics_t
    The loss is the mean squared error between these quantities.
    
    Args:
        trajectory (Tensor): Tensor of shape (T+1, batch_size, num_features) containing a chunk of the trajectory.
        predicted_dynamics (Tensor): Tensor of shape (T, batch_size, num_features) representing the model's predicted dynamics
                                     for each time step in the chunk.
        dt (float): Time step size.
        weight (float): Loss weight multiplier.
    
    Returns:
        loss (Tensor): A scalar tensor representing the MSR loss.
    """
    # Compute the finite difference (discrete time derivative) of the trajectory.
    # This yields a tensor of shape (T, batch_size, num_features).
    discrete_derivative = (trajectory[1:] - trajectory[:-1]) / dt
    
    # Compute the residual between the observed derivative and the predicted dynamics.
    residual = discrete_derivative - predicted_dynamics
    
    # Compute the mean squared error of the residual.
    loss = weight * torch.mean(residual ** 2)
    return loss

def msr_loss_from_file(file_path, source, chunk_start, chunk_length, model, dt, config, weight=1.0):
    """
    Load a contiguous chunk of trajectory data from disk using the data loader and compute the MSR loss on that chunk.
    
    This function uses the TimeSeriesDataset to load the data. It extracts a chunk of length (chunk_length + 1)
    starting at chunk_start (so that there are chunk_length differences available). Then, it uses the model's
    predict_dynamics() method (which must be implemented by the model) to compute the predicted dynamics on the chunk.
    
    Args:
        file_path (str): Path to the data file.
        source (str): Source of the data ("csv", "yfinance", "tiingo", "compressed").
        chunk_start (int): Starting index for the chunk.
        chunk_length (int): Number of time steps in the chunk (T); the trajectory will have T+1 points.
        model: Forecasting model that provides a predict_dynamics() method.
        dt (float): Time step size.
        config: Configuration object.
        weight (float): Loss weight multiplier.
    
    Returns:
        loss (Tensor): The computed MSR loss for the chunk.
    """
    # Load the entire dataset using the TimeSeriesDataset (this dataset builds sequences,
    # but we can access the full underlying data array).
    # Here, we set seq_len to a large number to force the loader to load the full trajectory.
    # Alternatively, you could write a dedicated loader function for contiguous data.
    dataset = TimeSeriesDataset(file_path, source, seq_len=chunk_length + 1, config=config)
    
    # The underlying data is stored in dataset.data (a NumPy array of shape (num_samples, num_features)).
    # Ensure we have enough data.
    if dataset.data.shape[0] < chunk_start + chunk_length + 1:
        raise ValueError("Not enough data to extract the requested chunk.")
    
    # Extract the chunk: shape (chunk_length + 1, num_features)
    chunk_np = dataset.data[chunk_start : chunk_start + chunk_length + 1]
    # Convert to a torch tensor and add a batch dimension (assume batch_size = 1 for MSR loss computation).
    chunk = torch.tensor(chunk_np, dtype=torch.float32).unsqueeze(1)  # Shape: (T+1, 1, num_features)
    
    # Use the model's predict_dynamics method to compute predicted dynamics for the chunk.
    # This method should take an input sequence of shape (T, batch_size, num_features) and return a tensor
    # of shape (T, batch_size, num_features) representing the predicted dynamics.
    # For example, predicted_dynamics = model.predict_dynamics(chunk[:-1])
    if not hasattr(model, "predict_dynamics"):
        raise AttributeError("The model does not have a predict_dynamics() method required for MSR loss computation.")
    
    predicted_dynamics = model.predict_dynamics(chunk[:-1])
    
    # Compute the MSR loss on the chunk.
    loss = msr_loss(chunk, predicted_dynamics, dt, weight)
    return loss

if __name__ == "__main__":
    # Example usage for individual loss components.
    # Smoothness regularization example:
    kernel = torch.linspace(1.0, 0.1, steps=10)
    smooth_loss = smoothness_regularizer(kernel, weight=1e-3)
    print("Smoothness regularization loss:", smooth_loss.item())

    # Sparsity regularization example:
    graph_matrix = torch.randn(5, 5)
    sparsity_loss = sparsity_regularizer(graph_matrix, weight=1e-3)
    print("Sparsity regularization loss:", sparsity_loss.item())

    # L1 and L2 regularizers example:
    dummy_params = [torch.randn(3, 3), torch.randn(4, 2)]
    l1_loss = l1_regularizer(dummy_params, weight=1e-3)
    l2_loss = l2_regularizer(dummy_params, weight=1e-3)
    print("L1 regularization loss:", l1_loss.item())
    print("L2 regularization loss:", l2_loss.item())

    # MSR loss example:
    # Simulate a trajectory chunk: (T+1, batch_size, num_features)
    T_plus_one, batch_size, num_features = 11, 4, 3
    trajectory = torch.randn(T_plus_one, batch_size, num_features)
    predicted_dynamics = torch.randn(T_plus_one - 1, batch_size, num_features)
    dt = 1.0
    msr_loss_val = msr_loss(trajectory, predicted_dynamics, dt, weight=1.0)
    print("MSR loss:", msr_loss_val.item())

    # Note: To test msr_loss_from_file, you must have a valid data file on disk,
    # a configuration object, and a model with a predict_dynamics() method.
