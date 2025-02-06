# project/models/mz_model.py
# --------------------------
# This module implements the coupled Mori–Zwanzig (MZ) model for multidimensional time-series forecasting.
# The MZ model decomposes the state update into two parts:
#   1. An instantaneous operator, implemented as an MLP.
#   2. An explicit memory term, computed as a weighted sum (convolution) over past states.
#
# The memory term consists of:
#   - A learnable memory kernel K_φ that modulates the influence of past states as a function of the lag.
#   - A dynamic coupling matrix G_ψ, which is decomposed into self (diagonal) and cross (off-diagonal) components.
#
# The final update is given by:
#   X_{t+1} = X_t + Δt * (instantaneous operator + memory term) + ε_t,
# where ε_t is a residual noise term (handled separately).
#
# Hooks are provided to extract the effective kernel values and the coupling matrices for analysis.

# project/models/mz_model.py
# --------------------------
import torch
import torch.nn as nn

class MZModel(nn.Module):
    def __init__(self, config):
        super(MZModel, self).__init__()
        self.config = config
        input_dim = config.LSTM_INPUT_DIM
        self.delta_t = config.TIME_STEP

        # Instantaneous operator
        self.inst_operator = nn.Sequential(
            nn.Linear(input_dim, config.MZ_OPERATOR_DIM),
            nn.ReLU(),
            nn.Linear(config.MZ_OPERATOR_DIM, input_dim)
        )

        # Memory kernel
        self.memory_kernel = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Coupling
        self.coupling_self = nn.Parameter(torch.randn(input_dim))
        self.coupling_cross = nn.Parameter(torch.randn(input_dim, input_dim))

    def forward(self, X_seq):
        """
        X_seq: shape (seq_len, batch_size, input_dim)
        Return:
            X_next: shape (batch_size, input_dim)
            internal_states: dict with K_values, G_self, G_cross
        """
        seq_len, batch_size, input_dim = X_seq.shape
        device = X_seq.device

        X_t = X_seq[-1]
        inst = self.inst_operator(X_t)

        memory_sum = torch.zeros_like(X_t)
        K_list = []
        G_self_list = []
        G_cross_list = []

        for j in range(seq_len):
            lag = float(seq_len - j)
            lag_tensor = torch.tensor([[lag]], dtype=torch.float32, device=device)
            K_val = self.memory_kernel(lag_tensor)
            K_list.append(K_val.squeeze())

            G_self = torch.diag(self.coupling_self)
            G_cross = self.coupling_cross - torch.diag(torch.diag(self.coupling_cross))
            G_total = G_self + G_cross
            G_self_list.append(G_self)
            G_cross_list.append(G_cross)

            X_j = X_seq[j]
            memory_sum += K_val * torch.matmul(X_j, G_total.t())

        X_next = X_t + self.delta_t * (inst + memory_sum)
        internal_states = {
            "K_values": torch.stack(K_list),
            "G_self": G_self_list,
            "G_cross": G_cross_list
        }
        return X_next, internal_states

    def predict_dynamics(self, X_seq):
        """
        For each t in [0..T-2], estimate d_pred_t = instant_op(X_t) + sum_{j=0..t} K(t-j)*G(t,j)*X_j
        Return shape: (T-1, batch_size, input_dim)
        """
        seq_len, batch_size, input_dim = X_seq.shape
        device = X_seq.device
        dt = self.delta_t

        d_preds = []
        for t in range(seq_len - 1):
            X_t = X_seq[t]
            inst = self.inst_operator(X_t)
            memory_sum = torch.zeros_like(X_t)

            for j in range(t + 1):
                lag = float((t+1) - j)
                lag_tensor = torch.tensor([[lag]], dtype=torch.float32, device=device)
                K_val = self.memory_kernel(lag_tensor)

                G_self = torch.diag(self.coupling_self)
                G_cross = self.coupling_cross - torch.diag(torch.diag(self.coupling_cross))
                G_total = G_self + G_cross
                X_j = X_seq[j]
                memory_sum += K_val * torch.matmul(X_j, G_total.t())

            d_pred = inst + memory_sum  # shape (batch_size, input_dim)
            d_preds.append(d_pred)

        d_preds = torch.stack(d_preds, dim=0)  # shape (T-1, batch_size, input_dim)
        return d_preds


if __name__ == "__main__":
    # Example usage:
    from config.config import Config
    config = Config()
    # Create a dummy sequence: (seq_len, batch_size, input_dim)
    seq_len, batch_size, input_dim = 10, 4, config.LSTM_INPUT_DIM
    dummy_seq = torch.randn(seq_len, batch_size, input_dim)
    
    model = MZModel(config)
    X_next, internal_states = model(dummy_seq)
    print("Predicted next state shape:", X_next.shape)
    print("Kernel values shape:", internal_states["K_values"].shape)
    # Note: G_self and G_cross are lists of matrices; print the shape of the first matrix for reference.
    print("G_self matrix shape:", internal_states["G_self"][0].shape)
    print("G_cross matrix shape:", internal_states["G_cross"][0].shape)
