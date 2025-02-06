# project/data/wavelet_decomp.py
# ------------------------------
# This module provides optional wavelet decomposition on each column of a DataFrame.
# If wavelet is disabled, the identity function is used. The output DataFrame can be used
# in place of the original for further processing.

import pandas as pd
import numpy as np
import pywt

def wavelet_transform_df(df, wavelet_type="db4", level=1):
    """
    Perform wavelet decomposition on each column of df, returning
    a new DataFrame that concatenates approximation and detail coefficients.
    For multi-level decomposition, more detail columns can be appended.
    """
    transformed_dict = {}
    for col in df.columns:
        data_col = df[col].values
        coeffs = pywt.wavedec(data_col, wavelet=wavelet_type, level=level)
        # coeffs[0] is approximation, coeffs[1..] are details
        for i, c in enumerate(coeffs):
            new_col_name = f"{col}_wave_{i}"
            transformed_dict[new_col_name] = c
    # Build a new DataFrame from the dict of arrays, aligning lengths by zero-filling the shorter detail arrays
    # For simplicity, we align them to the maximum length among the decomposed arrays.
    max_len = max(len(arr) for arr in transformed_dict.values())
    for k, arr in transformed_dict.items():
        if len(arr) < max_len:
            arr_padded = np.pad(arr, (0, max_len - len(arr)), mode='constant', constant_values=np.nan)
            transformed_dict[k] = arr_padded
    
    df_out = pd.DataFrame(transformed_dict)
    # We can drop NaN rows or fill them; it depends on how you want to handle alignment.
    df_out.dropna(inplace=True)
    return df_out

def inverse_wavelet_transform_df(df_wave, original_columns, wavelet_type="db4", level=1):
    """
    Reconstruct the original columns from wavelet coefficients in df_wave.
    If wavelet is disabled, we simply return df_wave (identity).
    Otherwise, we group columns by prefix to invert them one by one.
    """
    # A full inverse operation requires carefully reversing the transform.
    # We'll assume each original column produced 'level+1' sets of coeffs.
    # For demonstration, we show a simplified approach that reverts single-level decomposition.
    reconstructed_dict = {}
    
    # We'll parse df_wave columns to group them by the original prefix. 
    # e.g. if we have col_wave_0, col_wave_1 => we group them for inverse.
    wave_groups = {}
    for c in df_wave.columns:
        # c might look like "Close_wave_0", "Close_wave_1", etc.
        prefix, _, idx_str = c.rpartition("_wave_")
        idx = int(idx_str)
        if prefix not in wave_groups:
            wave_groups[prefix] = {}
        wave_groups[prefix][idx] = df_wave[c].values
    
    # For each prefix, gather coefficients and perform inverse.
    for prefix, coeff_dict in wave_groups.items():
        # We assume keys are 0..level for level wave decomposition
        coeffs = [coeff_dict[i] for i in sorted(coeff_dict.keys())]
        data_rec = pywt.waverec(coeffs, wavelet_type)
        reconstructed_dict[prefix] = data_rec
    
    df_reconstructed = pd.DataFrame(reconstructed_dict)
    # Optionally rename columns to original_columns if we have that mapping.
    for col in original_columns:
        if col in df_reconstructed.columns:
            pass  # it's already there
    return df_reconstructed

def identity_transform_df(df):
    """
    Identity transform if wavelet decomposition is disabled.
    """
    return df.copy()

def identity_inverse_transform_df(df):
    """
    Identity inverse if wavelet is disabled.
    """
    return df.copy()
