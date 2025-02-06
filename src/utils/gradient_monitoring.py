# FILE: HMSMC_stock_dynamics/src/utils/gradient_monitoring.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

class GradientMonitor:
    def __init__(self, model, clip_value=1.0):
        self.model = model
        self.clip_value = clip_value
        self.logger = logging.getLogger(__name__)

    def log_gradient_norms(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.logger.info(f'Gradient Norm: {total_norm:.4f}')

    def log_gradient_statistics(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                self.logger.info(f'{name} - Grad Mean: {grad_mean:.4f}, Grad Std: {grad_std:.4f}')

    def track_gradient_histograms(self, writer, step):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'{name}_grad', param.grad, step)

    def apply_gradient_clipping(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)

    def initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def regularize_mori_zwanzig_kernel(self, kernel_network, regularization_weight=0.01):
        regularization_loss = 0
        for param in kernel_network.parameters():
            regularization_loss += torch.norm(param, p=2)
        return regularization_weight * regularization_loss

def get_optimizer(model, optimizer_type='adam', learning_rate=0.001):
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f'Unsupported optimizer type: {optimizer_type}')

def train_model_with_monitoring(model, train_loader, criterion, optimizer, num_epochs, kernel_network, writer):
    monitor = GradientMonitor(model)
    monitor.initialize_weights()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, targets = batch['inputs'], batch['targets']
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            regularization_loss = monitor.regularize_mori_zwanzig_kernel(kernel_network)
            total_loss = loss + regularization_loss
            total_loss.backward()
            monitor.apply_gradient_clipping()
            optimizer.step()

            monitor.log_gradient_norms()
            monitor.log_gradient_statistics()
            monitor.track_gradient_histograms(writer, step)

        print(f'Epoch {epoch+1}, Loss: {total_loss.item():.4f}')