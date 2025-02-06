# FILE: setup_environment.sh

#!/bin/bash

# Create virtual environment
python -m venv finance-venv

# Activate virtual environment
source finance-venv/bin/activate

# Install requirements
pip install -r HMSMC_stock_dynamics/requirements.txt

echo "Virtual environment 'finance-venv' created and requirements installed."