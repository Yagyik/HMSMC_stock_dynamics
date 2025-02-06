import torch
import torch.nn as nn

class LossFunctions:
    def __init__(self, delta=1.0, laplacian_weight=0.1, sparsity_weight=0.01, stability_weight=0.05, memory_weight=0.1):
        self.delta = delta
        self.laplacian_weight = laplacian_weight
        self.sparsity_weight = sparsity_weight
        self.stability_weight = stability_weight
        self.memory_weight = memory_weight

    def mse_loss(self, predictions, targets):
        return nn.MSELoss()(predictions, targets)

    def huber_loss(self, predictions, targets):
        return nn.SmoothL1Loss()(predictions, targets)

    def graph_laplacian_loss(self, phi, laplacian):
        return self.laplacian_weight * torch.trace(phi.T @ laplacian @ phi)

    def sparsity_loss(self, adjacency_matrix):
        return self.sparsity_weight * torch.norm(adjacency_matrix, p=1)

    def stability_loss(self, adjacency_t, adjacency_t_minus_1):
        return self.stability_weight * torch.norm(adjacency_t - adjacency_t_minus_1, p='fro')

    def memory_loss(self, predictions, past_states, kernel_network):
        memory_terms = [kernel_network(state) for state in past_states]
        memory_estimate = sum(memory_terms)
        return self.memory_weight * torch.norm(predictions - memory_estimate, p=2)

    def total_loss(self, predictions, targets, phi, laplacian, adjacency_t, adjacency_t_minus_1, past_states, kernel_network):
        return (
            self.mse_loss(predictions, targets) +
            self.graph_laplacian_loss(phi, laplacian) +
            self.sparsity_loss(adjacency_t) +
            self.stability_loss(adjacency_t, adjacency_t_minus_1) +
            self.memory_loss(predictions, past_states, kernel_network)
        )
