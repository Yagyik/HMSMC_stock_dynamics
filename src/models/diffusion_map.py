# FILE: HMSMC_stock_dynamics/src/models/diffusion_map.py

from sklearn.manifold import SpectralEmbedding
import numpy as np

def diffusion_map(data, n_components=2):
    embedding = SpectralEmbedding(n_components=n_components)
    return embedding.fit_transform(data)