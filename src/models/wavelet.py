def wavelet_decomposition(data, wavelet='haar', level=1):
    import pywt
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs

def inverse_wavelet_decomposition(coeffs, wavelet='haar'):
    import pywt
    reconstructed_data = pywt.waverec(coeffs, wavelet)
    return reconstructed_data

def continuous_wavelet_transform(data, wavelet='cmor'):
    import pywt
    scales = range(1, 128)
    coefficients, frequencies = pywt.cwt(data, wavelet, scales)
    return coefficients, frequencies

def discrete_wavelet_transform(data, wavelet='haar'):
    import pywt
    coeffs = pywt.dwt(data, wavelet)
    return coeffs

def wavelet_transform(data, wavelet='db1'):
    import pywt
    coeffs = pywt.wavedec(data, wavelet)
    return coeffs


# Additional utility functions for wavelet analysis can be added here.

import pywt
import numpy as np

class WaveletDecomposer:
    def __init__(self, wavelet='db4', level=3):
        self.wavelet = wavelet
        self.level = level

    def decompose(self, time_series):
        coeffs = pywt.wavedec(time_series, self.wavelet, level=self.level)
        return coeffs  # [approximation, details]

    def reconstruct(self, coeffs):
        return pywt.waverec(coeffs, self.wavelet)
