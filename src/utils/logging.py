


# project/utils/logger.py
# -----------------------
# This file provides a simple Logger class for tracking metrics (loss, accuracy, etc.)
# during training. Metrics are stored in memory or optionally written to a CSV or JSON file.

import csv
import json
import os

class TrainingLogger:
    """
    A flexible logger for capturing metrics (e.g., loss, test_loss) at each epoch
    or iteration. You can choose to store logs in memory only or also save them to disk.
    """
    def __init__(self, save_dir=None, file_format="csv"):
        """
        Args:
            save_dir (str): Directory to save log files. If None, logs are kept in memory only.
            file_format (str): "csv" or "json" for saving logs to disk.
        """
        self.save_dir = save_dir
        self.file_format = file_format
        self.logs = []  # list of dicts, each dict is e.g. {"epoch":1, "train_loss":..., "test_loss":...}

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def log_metrics(self, metrics_dict):
        """
        Append a dictionary of metrics to the logs.
        e.g., metrics_dict = {"epoch":1, "train_loss": 0.2, "test_loss": 0.3}
        """
        self.logs.append(metrics_dict)

    def save_logs(self, filename="training_logs"):
        """
        Save the in-memory logs to a file in the specified format (csv or json).
        """
        if self.save_dir is None:
            print("save_dir is None, logs are not saved to disk.")
            return
        path = os.path.join(self.save_dir, f"{filename}.{self.file_format}")
        if self.file_format == "csv":
            keys = self.logs[0].keys() if len(self.logs) > 0 else []
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in self.logs:
                    writer.writerow(row)
        elif self.file_format == "json":
            with open(path, "w") as f:
                json.dump(self.logs, f, indent=2)
        else:
            print(f"Unsupported file format: {self.file_format}")

    def get_logs(self):
        """ Return the in-memory list of metrics dicts. """
        return self.logs

if __name__ == "__main__":
    logger = TrainingLogger(save_dir="logs", file_format="csv")
    logger.log_metrics({"epoch":1, "train_loss":0.5, "test_loss":0.7})
    logger.log_metrics({"epoch":2, "train_loss":0.4, "test_loss":0.6})
    logger.save_logs("my_run")
