import numpy as np
import torch


def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            predictions = model(batch['inputs'])
            targets = batch['targets']
            loss = criterion(predictions, targets)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def evaluate_graph_sparsity(adjacency_matrix):
    sparsity = np.sum(adjacency_matrix == 0) / adjacency_matrix.size
    return sparsity

def evaluate_drift_consistency(drift_estimates, actual_drift):
    consistency = np.mean(np.abs(drift_estimates - actual_drift))
    return consistency

def evaluate_memory_kernel_accuracy(memory_kernel_estimates, actual_memory_kernel):
    accuracy = np.mean(np.abs(memory_kernel_estimates - actual_memory_kernel))
    return accuracy

def save_evaluation_results(results, filepath):
    with open(filepath, 'w') as f:
        f.write("Average Loss: {}\n".format(results[0]))
        f.write("Accuracy: {}\n".format(results[1]))