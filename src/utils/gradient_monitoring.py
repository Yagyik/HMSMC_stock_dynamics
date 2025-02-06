# project/utils/gradient_monitoring.py
# ------------------------------------
# This file provides classes or functions to monitor gradients during training.
# A typical use case is to register hooks on model parameters to log gradient norms
# or detect exploding/vanishing gradients. The code below illustrates a simple
# "GradientMonitor" class.

import torch

class GradientMonitor:
    """
    A utility class that registers backward hooks on a model’s parameters
    to monitor gradient statistics such as mean, std, or max norm.
    """
    def __init__(self):
        self.gradient_stats = []

    def register_hooks(self, model):
        """
        Register hooks on the model’s parameters to collect gradient stats after backprop.
        Args:
            model: A PyTorch nn.Module
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(self._make_hook(name))

    def _make_hook(self, param_name):
        """
        Returns a hook function that computes stats from gradients.
        """
        def hook(grad):
            # Compute gradient statistics
            grad_norm = grad.norm(p=2).item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()

            self.gradient_stats.append({
                "param_name": param_name,
                "grad_norm": grad_norm,
                "grad_mean": grad_mean,
                "grad_std": grad_std
            })
        return hook

    def reset(self):
        """ Reset the stored gradient stats. """
        self.gradient_stats = []

    def get_stats(self):
        """ Return a list of collected gradient statistics. """
        return self.gradient_stats

if __name__ == "__main__":
    # Example usage
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(10, 1)
        def forward(self, x):
            return self.lin(x)

    model = SimpleModel()
    grad_monitor = GradientMonitor()
    grad_monitor.register_hooks(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)

    for epoch in range(2):
        grad_monitor.reset()
        optimizer.zero_grad()
        preds = model(x)
        loss = (preds - y).pow(2).mean()
        loss.backward()
        optimizer.step()

        # Access gradient statistics
        stats = grad_monitor.get_stats()
        print(f"Epoch {epoch}: Collected gradient stats = {stats}")
