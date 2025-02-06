# FILE: HMSMC_stock_dynamics/running_examples.sh

#!/bin/bash

# Load and preprocess data
python src/data/load_data.py --source yfinance --tickers AAPL,TSLA,GOOGL
python src/data/clean_data.py

# Embed metadata
python src/data/embed_metadata.py

# Create and update graph
python src/models/graph.py --method similarity
python src/models/laplacian_operator.py

# Train model
python src/training/train.py --epochs 100 --use_lstm

# Evaluate model
python src/training/evaluate.py

# Visualize results
python src/utils/visualization.py --plot stocks