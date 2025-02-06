from sklearn.preprocessing import StandardScaler
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class MetadataEmbedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.scaler = StandardScaler()

    def embed_text(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
        return np.vstack(embeddings)

    def normalize_numerical(self, features):
        return self.scaler.fit_transform(features)

    def combine_features(self, text_embeddings, numerical_features):
        return np.hstack([text_embeddings, numerical_features])


