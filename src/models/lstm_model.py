# project/models/lstm_model.py
# ----------------------------
# This module implements the LSTM model with an optional attention mechanism.
# The model is designed for time-series forecasting, exposing its internal
# memory by unrolling the cell state update. It extracts key internal states:
# - The implicit memory kernel (α), obtained from the product of forget gates.
# - Optional explicit attention weights (β) from an attention module.
# - The hidden state H_t and the effective cell state C_effective.
#
# A modular read-out layer (imported from models/readout_layer.py) is used
# to compute the residual update for the observable state X_{t+1} from the
# concatenated latent representation [H_t; C_effective].
#
# Hooks are provided to extract the read-out weight matrices (W_self and W_cross)
# for further analysis of self (diagonal) and cross (off-diagonal) contributions.

# project/models/lstm_model.py
# ----------------------------
import torch
import torch.nn as nn
from ..models.readout_layer import ReadOutLayer

class LSTMWithAttention(nn.Module):
    def __init__(self, config):
        super(LSTMWithAttention, self).__init__()
        self.config = config
        input_dim = config.LSTM_INPUT_DIM
        hidden_dim = config.LSTM_HIDDEN_DIM
        self.use_attention = config.USE_ATTENTION

        # Single-layer LSTMCell
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)

        if self.use_attention:
            self.attention_layer = nn.Sequential(
                nn.Linear(input_dim + hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        self.readout = ReadOutLayer(input_dim=2*hidden_dim, output_dim=config.LSTM_INPUT_DIM)

    def forward(self, X):
        """
        X: shape (seq_len, batch_size, input_dim)
        Return:
            X_next: (batch_size, input_dim)
            internal_states: dictionary
        """
        seq_len, batch_size, _ = X.shape
        device = X.device
        H_t = torch.zeros(batch_size, self.config.LSTM_HIDDEN_DIM, device=device)
        C_t = torch.zeros(batch_size, self.config.LSTM_HIDDEN_DIM, device=device)

        alpha_list = []
        beta_list = []
        candidate_list = []

        for t in range(seq_len):
            X_t = X[t]
            # print(X_t.shape, H_t.shape, C_t.shape)
            # print(self.config.LSTM_HIDDEN_DIM, self.config.LSTM_INPUT_DIM)
            H_t, C_t = self.lstm_cell(X_t, (H_t, C_t))

            alpha_t = torch.ones(batch_size, self.config.LSTM_HIDDEN_DIM, device=device)
            alpha_list.append(alpha_t)

            candidate = C_t
            candidate_list.append(candidate)

            if self.use_attention:
                attn_input = torch.cat([X_t, H_t], dim=1)
                beta_t = self.attention_layer(attn_input)
                beta_list.append(beta_t)
            else:
                beta_list.append(torch.ones(batch_size, 1, device=device))

        # effective cell state
        C_effective = sum(beta * cand for beta, cand in zip(beta_list, candidate_list)) / seq_len
        H_effective = H_t

        latent = torch.cat([H_effective, C_effective], dim=1)
        Y = self.readout(latent)
        X_next = X[-1] + self.config.TIME_STEP * Y  # residual update
        internal_states = {
            "H_t": H_effective,
            "C_effective": C_effective,
            "alpha": torch.stack(alpha_list),
            "beta": torch.stack(beta_list)
        }
        return X_next, internal_states

    def extract_readout_weights(self):
        return self.readout.get_weights()

    def predict_dynamics(self, X_seq):
        """
        Given a chunk of states X_seq with shape (T, batch_size, input_dim),
        return predicted derivatives d_pred of shape (T-1, batch_size, input_dim).

        We do a naive approach:
          For each t in [0..T-2],
            1) Use the model forward pass on X_seq[:t+1] to get X_pred at time t+1
            2) d_pred_t = (X_pred - X_seq[t]) / dt
        """
        seq_len, batch_size, input_dim = X_seq.shape
        device = X_seq.device
        dt = self.config.TIME_STEP

        d_preds = []
        # We'll keep an internal hidden and cell state across steps
        H_t = torch.zeros(batch_size, self.config.LSTM_HIDDEN_DIM, device=device)
        C_t = torch.zeros(batch_size, self.config.LSTM_HIDDEN_DIM, device=device)

        for t in range(seq_len - 1):
            X_t = X_seq[t]
            H_t, C_t = self.lstm_cell(X_t, (H_t, C_t))

            # attention logic
            if self.use_attention:
                attn_input = torch.cat([X_t, H_t], dim=1)
                beta_t = self.attention_layer(attn_input)
            else:
                beta_t = torch.ones(batch_size, 1, device=device)

            # We'll approximate C_effective with the current C_t for the final step
            # or do a running average. For simplicity, just do current step:
            C_effective = C_t
            # read-out
            latent = torch.cat([H_t, C_effective], dim=1)
            Y = self.readout(latent)
            X_pred = X_t + dt * Y
            d_pred = (X_pred - X_t) / dt
            d_preds.append(d_pred)

        d_preds = torch.stack(d_preds, dim=0)  # shape (T-1, batch_size, input_dim)
        return d_preds


if __name__ == "__main__":
    # Example usage:
    from config.config import Config
    config = Config()
    # Simulate a dummy input sequence: (seq_len, batch_size, input_dim)
    seq_len, batch_size, input_dim = 10, 4, config.LSTM_INPUT_DIM
    dummy_input = torch.randn(seq_len, batch_size, input_dim)
    
    model = LSTMWithAttention(config)
    X_next, internal_states = model(dummy_input)
    print("Predicted next state shape:", X_next.shape)
    print("Extracted alpha shape:", internal_states["alpha"].shape)
    print("Extracted beta shape:", internal_states["beta"].shape)
    
    # Extract read-out weights (for analysis)
    W_self, W_cross = model.extract_readout_weights()
    print("W_self shape:", W_self.shape)
    print("W_cross shape:", W_cross.shape)
