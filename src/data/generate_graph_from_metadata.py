from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

def compute_graph_laplacian(adjacency_matrix):
    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
    laplacian = degree_matrix - adjacency_matrix
    return laplacian

def symmetrize_adjacency(adjacency):
    return (adjacency + adjacency.T) / 2

def threshold_adjacency(similarity_matrix, threshold=0.5):
    return (similarity_matrix >= threshold).astype(float)

def sparsify_adjacency(similarity_matrix, k=5):
    n = similarity_matrix.shape[0]
    adjacency = np.zeros_like(similarity_matrix)
    for i in range(n):
        top_k_indices = np.argsort(similarity_matrix[i])[-k:]  # Top k indices
        adjacency[i, top_k_indices] = similarity_matrix[i, top_k_indices]
    return adjacency



def gaussian_similarity_matrix(embeddings, sigma=1.0):
    dist_matrix = np.linalg.norm(embeddings[:, None] - embeddings[None, :], axis=-1) ** 2
    return np.exp(-dist_matrix / (2 * sigma ** 2))



def generate_adjacency_matrix(text_metadata, numerical_metadata, k=5, sigma=None):
    # Step 1: Embed text metadata
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    text_embeddings = []
    for text in text_metadata:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        text_embeddings.append(embedding)
    text_embeddings = np.vstack(text_embeddings)

    # Step 2: Normalize numerical metadata
    scaler = StandardScaler()
    normalized_numerical = scaler.fit_transform(numerical_metadata)

    # Step 3: Combine features
    combined_features = np.hstack([text_embeddings, normalized_numerical])

    # Step 4: Compute similarity matrix
    if sigma is None:
        similarity_matrix = cosine_similarity(combined_features)
    else:
        dist_matrix = np.linalg.norm(combined_features[:, None] - combined_features[None, :], axis=-1) ** 2
        similarity_matrix = np.exp(-dist_matrix / (2 * sigma ** 2))

    # Step 5: Sparsify adjacency matrix (k-NN)
    adjacency_matrix = np.zeros_like(similarity_matrix)
    for i in range(similarity_matrix.shape[0]):
        top_k_indices = np.argsort(similarity_matrix[i])[-k:]
        adjacency_matrix[i, top_k_indices] = similarity_matrix[i, top_k_indices]

    # Step 6: Symmetrize adjacency matrix
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2

    return adjacency_matrix