import logging

# Configure the logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

def get_logger(name):
    """Create a logger with the specified name."""
    return logging.getLogger(name)