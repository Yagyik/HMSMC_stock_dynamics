# project/models/residual_noise.py
# -------------------------------
# This module provides a function to sample residual noise from various distributions.
# The choice of distribution is configurable via the configuration object.
#
# Supported distributions:
# - "normal": Gaussian distribution with specified mean and standard deviation.
# - "lognormal": Log-Normal distribution.
# - "poisson": Poisson distribution.
# - "heavy_tailed": A heavy-tailed distribution (here, implemented as a Cauchy distribution).
#
# The function sample_noise() returns a PyTorch tensor of the specified shape.
# This module is used during training to inject residual noise into the state updates.
#
# Usage:
#     noise = sample_noise(shape, config)
# where shape is a tuple (e.g., (batch_size, num_features)) and config is the configuration object.

import torch

def sample_noise(shape, config):
    """
    Sample residual noise based on the configuration.
    
    Args:
        shape (tuple): Desired shape of the noise tensor.
        config: Configuration object containing noise settings.
        
    Returns:
        noise (Tensor): A PyTorch tensor containing noise samples.
    """
    dist_type = config.NOISE_DISTRIBUTION.lower()
    params = config.NOISE_PARAMS

    if dist_type == "normal":
        # Normal distribution: sample from N(mean, std^2)
        mean = params.get("mean", 0.0)
        std = params.get("std", 1.0)
        noise = torch.randn(shape) * std + mean
    elif dist_type == "lognormal":
        # Lognormal distribution: sample from lognormal with specified mean and std.
        # Note: torch.lognormal_ uses the log-mean and log-std.
        mean = params.get("mean", 0.0)
        std = params.get("std", 1.0)
        noise = torch.empty(shape).log_normal_(mean=mean, std=std)
    elif dist_type == "poisson":
        # Poisson distribution: lambda is provided in the parameters.
        lam = params.get("lam", 1.0)
        # torch.poisson expects a tensor of rates, so we create one filled with lam.
        noise = torch.poisson(torch.full(shape, lam))
    elif dist_type == "heavy_tailed":
        # Heavy-tailed distribution: using a standard Cauchy distribution.
        cauchy = torch.distributions.Cauchy(loc=0.0, scale=1.0)
        noise = cauchy.sample(shape)
    else:
        raise ValueError(f"Unsupported noise distribution: {dist_type}")
    
    return noise

if __name__ == "__main__":
    # Example usage:
    from config.config import Config
    config = Config()
    # Define a sample shape, for instance (batch_size, LSTM_INPUT_DIM)
    shape = (4, config.LSTM_INPUT_DIM)
    noise_tensor = sample_noise(shape, config)
    print("Sampled noise tensor:", noise_tensor)
