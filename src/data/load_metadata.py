import torch
from torch.utils.data import Dataset

class MetadataEmbeddingDataset(Dataset):
    def __init__(self, text_data, numerical_data, embedder):
        """
        Args:
            text_data (list of str): List of textual metadata (e.g., company descriptions).
            numerical_data (np.ndarray): Array of numerical metadata (e.g., market cap, P/E ratio).
            embedder (MetadataEmbedder): Pre-trained embedding model for text.
        """
        self.text_data = text_data
        self.numerical_data = numerical_data
        self.embedder = embedder

        # Precompute embeddings for efficiency
        self.text_embeddings = self.embedder.embed_text(self.text_data)
        self.numerical_embeddings = self.embedder.normalize_numerical(self.numerical_data)
        self.combined_embeddings = self.embedder.combine_features(self.text_embeddings, self.numerical_embeddings)

    def __len__(self):
        return len(self.combined_embeddings)

    def __getitem__(self, idx):
        """
        Returns:
            embedding (torch.Tensor): Combined text and numerical embedding for a single sample.
        """
        embedding = self.combined_embeddings[idx]
        return torch.tensor(embedding, dtype=torch.float32)

def get_metadata_dataloader(text_data, numerical_data, embedder, batch_size):
    dataset = MetadataEmbeddingDataset(text_data, numerical_data, embedder)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
