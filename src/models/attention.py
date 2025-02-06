import torch
import torch.nn as nn
import numpy as np

class CrossTimescaleAttention:
    def __init__(self, input_dim, output_dim, num_heads):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.Wq = self.initialize_weights(input_dim, output_dim)
        self.Wk = self.initialize_weights(input_dim, output_dim)
        self.Wv = self.initialize_weights(input_dim, output_dim)

    def initialize_weights(self, input_dim, output_dim):
        return np.random.rand(input_dim, output_dim)

    def forward(self, query, key, value):
        query = self.linear_transform(query, self.Wq)
        key = self.linear_transform(key, self.Wk)
        value = self.linear_transform(value, self.Wv)

        attention_scores = self.calculate_attention_scores(query, key)
        attention_output = self.apply_attention(attention_scores, value)

        return attention_output

    def linear_transform(self, x, weights):
        return np.dot(x, weights)

    def calculate_attention_scores(self, query, key):
        scores = np.dot(query, key.T)
        return self.softmax(scores)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def apply_attention(self, scores, value):
        return np.dot(scores, value)
    


class CrossTimescaleAttention(nn.Module):
    def __init__(self, input_dim, num_timescales):
        super(CrossTimescaleAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.keys = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_timescales)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, current_state, timescale_states):
        query = self.query(current_state)
        attention_scores = [
            torch.dot(query, self.keys[k](timescale_states[k])) for k in range(len(timescale_states))
        ]
        attention_weights = self.softmax(torch.tensor(attention_scores))
        combined_state = sum(attention_weights[k] * timescale_states[k] for k in range(len(timescale_states)))
        return combined_state, attention_weights
