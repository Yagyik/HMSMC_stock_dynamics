# **To Use:**

# 1) Set up environment in parent of HMSMC_stock_dynamics
   #!/bin/bash
   rm -rf finance-venv
   
   a) Create virtual environment
   python -m venv finance-venv
   
   b) Activate virtual environment
   source finance-venv/bin/activate
   
   c) Install requirements
   pip install -r HMSMC_stock_dynamics/requirements.txt
   
   echo "Virtual environment 'finance-venv' created and requirements installed."

# 2) Generate a prediction by running from parent of HMSMC_stock_dynamics (check paths)

   python3 HMSMC_stock_dynamics/predict.py --model_path HMSMC_stock_dynamics/logs/best_model.pth --train_data_path tiingo_downloader/dataset_extensive/NYSE_1_Jan_2016_to_1_Jan_2024_1min/ --test_data_path tiingo_downloader/dataset_extensive_test/NYSE_2_Jan_2024_to_31_Dec_2024_1min/ --seq_lens 500 --columns 1 --output_dir test_outputs/

# 3) Check the predictions vs actuals in test_output/results_<seq_len>.csv

------ background ----- 

# Stock Market Forecasting Codebase

This codebase implements a unified framework for stock market time-series forecasting using two complementary approaches:

1. **LSTM Model with Optional Attention:**  
   An LSTM-based model that, when “unrolled,” exposes its internal memory through an explicit weighted sum. This enables the extraction of self–(diagonal) and cross–(off–diagonal) interaction matrices, as well as the memory kernel (α) and, optionally, explicit attention weights (β).

2. **Coupled Mori–Zwanzig (MZ) Model:**  
   A physics-inspired model that decomposes the state update into an instantaneous operator and an explicit convolution over past states with a learnable memory kernel \(K_\phi\) and dynamic coupling \(G_\psi\), the latter of which is split into self and cross components.

Additional features include:
- **Configurable Residual Noise:** Sampled from a configurable distribution (normal, log-normal, Poisson, or heavy–tailed).
- **Optional Wavelet Decomposition:** For multiscale analysis, with an inversion module to reconstruct the original series.
- **Regularisation:** Modules to enforce smoothness of the memory kernel and sparsity in the coupling matrices.
- **Unified Hyperparameter Tuning:** Including grid search over key hyperparameters.

---

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Modules Description](#modules-description)
  - [Configuration](#configuration)
  - [Data Handling and Preprocessing](#data-handling-and-preprocessing)
  - [Wavelet Decomposition and Inversion](#wavelet-decomposition-and-inversion)
  - [Model Modules](#model-modules)
  - [Residual Noise Module](#residual-noise-module)
  - [Loss Functions and Regularisation](#loss-functions-and-regularisation)
  - [Experiments and Training](#experiments-and-training)
  - [Utilities](#utilities)
- [Installation and Dependencies](#installation-and-dependencies)
- [Usage](#usage)
- [Contribution and Extension](#contribution-and-extension)

---

## Overview

The primary objective of this codebase is to provide a robust and flexible platform for testing and comparing different forecasting approaches for multidimensional stock market time series. In our unified framework:

- **LSTM with Optional Attention:**  
  The LSTM is implemented to capture intrinsic memory via cell state updates, and by "unrolling" these updates, we expose the implicit memory kernel (α) and optional explicit attention weights (β). A modular read–out layer computes the final forecast update, and its weight matrix is decomposed into self–(diagonal) and cross–(off–diagonal) components.

- **Coupled MZ Model:**  
  This model expresses the state update as the sum of an instantaneous operator and an explicit, learnable convolution over past states:  
  \[
  X_{t+1} = X_t + \Delta t \left[ \mathcal{L}_\theta(X_t) + \sum_{j=0}^{t} K_\phi(t-j)\,\bigl( G_\psi^{\text{self}}(t,j) + G_\psi^{\text{cross}}(t,j) \bigr)\,X_j \right] + \epsilon_t.
  \]
  Here, \(K_\phi\) is the memory kernel and \(G_\psi\) is decomposed into self and cross components.

- **Residual Noise:**  
  The residual noise \(\epsilon_t\) is sampled from a configurable distribution (normal, log-normal, Poisson, or heavy–tailed) controlled via the configuration.

- **Wavelet Decomposition:**  
  Optionally, a wavelet decomposition module can map the original series into different frequency bands for multiscale analysis, with an inversion module that reconstructs \(X\) from the bands. When disabled, the inversion defaults to an identity mapping.

- **Regularisation and Losses:**  
  The framework supports standard forecasting losses (e.g., MSE) plus additional regularisers that enforce smoothness of the memory kernel and sparsity in the cross–interaction matrices.

- **Hyperparameter Tuning:**  
  A unified configuration module and grid search routine enable systematic experimentation with different hyperparameters.

---

## Directory Structure
project/
│
├── config/
│   └── config.py               # Contains all global hyperparameters and configuration settings.
│                               # Key features:
│                               #  - Data paths, model selection (LSTM or MZ), and attention options.
│                               #  - Noise distribution type and parameters.
│                               #  - Regularisation weights and grid search parameters.
│                               #  **Guidance:** Update this file before running experiments.
│
├── data/
│   ├── data_loader.py          # Loads and preprocesses the multidimensional time series data.
│   │                           # Key steps:
│   │                           #  - Read raw data (e.g., from CSV).
│   │                           #  - Clean, normalize, and extract features.
│   │                           #  **Guidance:** Extend this module for custom data formats or additional preprocessing.
│   └── wavelet_decomposition.py# Implements optional wavelet decomposition and inversion.
│                               # Key features:
│                               #  - Decomposes the time series into frequency bands using a specified wavelet.
│                               #  - Inversion function to reconstruct the original signal.
│                               #  **Guidance:** Modify the wavelet type if needed; when disabled, inversion returns identity.
│
├── models/
│   ├── lstm_model.py           # Implements the LSTM model with an optional attention mechanism.
│   │                           # Key features:
│   │                           #  - Uses PyTorch’s LSTMCell (or a custom LSTM cell) for state updates.
│   │                           #  - Extracts internal gates (e.g., forget gate product for \(\alpha\)) and optional attention weights (\(\beta\)).
│   │                           #  - Exposes internal states via hooks for later analysis.
│   │                           #  **Guidance:** Ensure attention mechanism is correctly integrated; verify hook outputs.
│   ├── mz_model.py             # Implements the coupled Mori–Zwanzig model.
│   │                           # Key features:
│   │                           #  - Defines an instantaneous operator as an MLP.
│   │                           #  - Implements a learnable memory kernel \(K_\phi\) and a dynamic coupling \(G_\psi\) split into self and cross parts.
│   │                           #  **Guidance:** Parameterize kernel and coupling functions carefully; validate matrix decompositions.
│   ├── residual_noise.py       # Provides functions to sample residual noise from configurable distributions.
│   │                           # Key features:
│   │                           #  - Supports normal, log-normal, Poisson, and heavy–tailed distributions.
│   │                           #  **Guidance:** Extend or modify distributions by editing this module.
│   └── readout_layer.py        # Defines the modular read-out layer that maps from latent representations ([H; C]) to state updates.
│                               # Key features:
│                               #  - Decomposes the weight matrix into \(W_{\text{self}}\) and \(W_{\text{cross}}\).
│                               #  - Provides extraction hooks for these matrices.
│                               #  **Guidance:** Implement masking if enforcing strict diagonal structure on \(W_{\text{self}}\) is required.
│
├── losses/
│   ├── loss_functions.py       # Contains standard forecasting loss functions (e.g., MSE).
│   │                           # **Guidance:** Add custom losses as needed.
│   └── regularizers.py         # Implements regularisers for smoothness (memory kernel) and sparsity (graph of connections).
│                               # **Guidance:** Tune regularisation weights in config; add more constraints if necessary.
│
├── experiments/
│   ├── train.py                # Main training loop that integrates all modules.
│   │                           # Key steps:
│   │                           #  - Loads and preprocesses data.
│   │                           #  - Generates sequences (sliding window approach).
│   │                           #  - Trains the model with loss computation and regularisation.
│   │                           #  - Extracts and logs internal matrices via hooks.
│   │                           #  **Guidance:** Verify the training loop and logging outputs for consistency.
│   └── hyperparameter_tuning.py# Implements grid search for hyperparameter tuning.
│                               # Key features:
│                               #  - Iterates over hyperparameter combinations.
│                               #  - Instantiates models, runs training, and logs performance.
│                               #  **Guidance:** Update the parameter grid in config as needed.
│
├── utils/
│   └── logger.py               # Provides logging utilities to record experiment progress, loss curves, and key parameters.
│                               # **Guidance:** Customize the logging format for your experiment tracking.
│
└── main.py                     # Main entry point for the codebase.
                                # Key features:
                                #  - Loads configuration.
                                #  - Instantiates the appropriate model (LSTM or MZ).
                                #  - Initiates training or hyperparameter tuning.
                                #  **Guidance:** Use this file to start experiments; ensure all configurations are set correctly.
---

## Detailed Module Descriptions

### 1. Configuration (`config/config.py`)
- **Purpose:**  
  Centralize all settings so that experiments are reproducible and easily configurable.
- **Key Steps:**  
  - Define paths, model types, attention options, noise parameters, regularisation weights, and grid search ranges.
  - **Guidance:** Update this file before starting an experiment.

### 2. Data Handling and Preprocessing (`data/data_loader.py`)
- **Purpose:**  
  Load, clean, and preprocess raw time-series data.
- **Key Steps:**  
  - Read data from CSV or other sources.
  - Normalize and standardize features.
  - Optionally, apply additional feature extraction.
- **Guidance:** Extend for custom data formats as needed.

### 3. Wavelet Decomposition and Inversion (`data/wavelet_decomposition.py`)
- **Purpose:**  
  Decompose the input series into frequency bands and reconstruct the original series.
- **Key Steps:**  
  - Apply a chosen wavelet transform.
  - Provide an inversion function that defaults to the identity mapping if wavelet decomposition is disabled.
- **Guidance:** Modify the wavelet type and parameters in config if required.

### 4. Model Modules
- **LSTM Model with Optional Attention (`models/lstm_model.py`):**  
  Implements an LSTM that:
  - Uses LSTM cell updates.
  - Extracts the implicit memory kernel (α) from the forget gates.
  - Optionally computes explicit attention weights (β) to modulate memory.
  - Uses a modular read-out layer to produce state updates.
  - **Guidance:** Verify that internal state extraction (α and β) is accurate and useful for analysis.
- **Coupled MZ Model (`models/mz_model.py`):**  
  Implements a physics-inspired model that:
  - Defines an instantaneous operator (via an MLP).
  - Parameterizes the memory kernel \(K_\phi\) and dynamic coupling \(G_\psi\).
  - Decomposes \(G_\psi\) into self and cross interaction matrices.
  - **Guidance:** Validate the parameterizations and decompositions; ensure matrix dimensions are consistent.
- **Residual Noise (`models/residual_noise.py`):**  
  Provides a flexible sampling mechanism for residual noise.
  - **Guidance:** Adjust noise parameters in the configuration as needed.
- **Read-Out Layer (`models/readout_layer.py`):**  
  Maps latent representations ([H; C]) to forecast updates and extracts self and cross weights.
  - **Guidance:** Consider implementing masking if a strict diagonal structure is required for \(W_{\text{self}}\).

### 5. Loss Functions and Regularisation
- **Loss Functions (`losses/loss_functions.py`):**  
  Implements standard forecasting losses such as MSE.
- **Regularisers (`losses/regularizers.py`):**  
  Enforces smoothness in the memory kernel and sparsity in the cross-coupling matrix.
  - **Guidance:** Tune regularisation weights via the configuration.

### 6. Experiments and Training
- **Training Loop (`experiments/train.py`):**  
  Integrates data loading, sequence generation, model forward passes, loss computation, regularisation, and optimization.
  - **Guidance:** Ensure consistency in sequence generation and logging of internal states.
- **Hyperparameter Tuning (`experiments/hyperparameter_tuning.py`):**  
  Implements grid search over a defined hyperparameter space.
  - **Guidance:** Update the grid search parameters in the configuration as required.

### 7. Utilities (`utils/logger.py`)
- **Purpose:**  
  Centralize logging for experiment tracking.
- **Key Steps:**  
  - Configure logging format and level.
  - Log key metrics, loss values, and extracted internal matrices for analysis.
- **Guidance:** Customize logging as needed for better experiment visibility.

### 8. Main Entry Point (`main.py`)
- **Purpose:**  
  Initialize configuration, select the appropriate model, and start the training or hyperparameter tuning process.
- **Key Steps:**  
  - Load configuration settings.
  - Instantiate the model (LSTM or MZ).
  - Call the training routine.
- **Guidance:** Ensure that all configuration parameters are set correctly before running.

---

## Installation and Dependencies

- **Python Version:** Python 3.7 or later  
- **Primary Libraries:**  
  - PyTorch  
  - NumPy  
  - Pandas  
  - PyWavelets (pywt)  
- **Additional Dependencies:** Listed in `requirements.txt` (if provided)

Install dependencies via:
```bash
pip install -r requirements.txt
