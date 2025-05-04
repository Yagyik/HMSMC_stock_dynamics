


# project/utils/logger.py
# -----------------------
# This file provides a simple Logger class for tracking metrics (loss, accuracy, etc.)
# during training. Metrics are stored in memory or optionally written to a CSV or JSON file.

import csv
import json
import os
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    def __init__(self,save_dir="logs"):
        self.logs = []
        self.save_dir = save_dir
        self.writer = SummaryWriter(self.save_dir)  # Initialize TensorBoard writer

    def log_metrics(self, metrics, step):
        """
        Log metrics to internal logs and TensorBoard.
        """
        self.logs.append((step, metrics))
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)  # Log to TensorBoard

    def save_logs(self, filename):
        """
        Save logs to a file.
        """
        with open(filename, "w") as f:
            for step, metrics in self.logs:
                f.write(f"Step {step}: {metrics}\n")

    def close(self):
        """
        Close the TensorBoard writer.
        """
        self.writer.close()

if __name__ == "__main__":
    logger = TrainingLogger(save_dir="logs", file_format="csv")
    logger.log_metrics({"epoch":1, "train_loss":0.5, "test_loss":0.7})
    logger.log_metrics({"epoch":2, "train_loss":0.4, "test_loss":0.6})
    logger.save_logs("my_run")
