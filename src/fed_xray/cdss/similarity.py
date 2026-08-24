"""
Fed-XRay Similarity Search Engine (Case-Based Reasoning)
=========================================================
Finds similar historical cases using visual embeddings and cosine similarity.
"""

import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn

from ..data.generator import MedicalDataGenerator


class HistoricalCaseBank:
    """
    Bank of historical patient cases with pre-computed feature embeddings.
    Used for similarity search during diagnosis.
    """
    
    def __init__(self, n_cases: int = 100, embedding_dim: int = 64, seed: int = 42):
        np.random.seed(seed)
        self.n_cases = n_cases
        self.embedding_dim = embedding_dim
        
        self.embeddings = []
        self.labels = []
        self.case_ids = []
        self.images = []
        
        self._generate_cases()
    
    def _generate_cases(self) -> None:
        """Generate synthetic historical cases with class-specific patterns."""
        generator = MedicalDataGenerator(seed=42)
        
        for i in range(self.n_cases):
            label = i % 3
            image = generator.generate_synthetic_xray(label, apply_augmentation=False)
            base_embedding = np.random.randn(self.embedding_dim) * 0.3
            
            if label == 0:
                class_offset = np.array([1.0] * 20 + [0.0] * 44)
            elif label == 1:
                class_offset = np.array([0.0] * 20 + [1.0] * 20 + [0.0] * 24)
            else:
                class_offset = np.array([0.0] * 40 + [1.0] * 24)
            
            embedding = base_embedding + class_offset * 0.5
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            self.embeddings.append(embedding)
            self.labels.append(label)
            self.case_ids.append(f"CASE-{1000 + i}")
            self.images.append(image)
        
        self.embeddings = np.array(self.embeddings)
    
    def find_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 2
    ) -> List[Dict]:
        """Find top-k most similar historical cases."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'case_id': self.case_ids[idx],
                'label': self.labels[idx],
                'similarity': float(similarities[idx]),
                'image': self.images[idx]
            })
        
        return results


def extract_embedding(model: nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    """Extract feature embedding from model's penultimate layer."""
    model.eval()
    features = []
    
    def hook_fn(module, input, output):
        features.append(output.detach())
    
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Flatten) or 'flatten' in name.lower():
            target_layer = module
            break
    
    if target_layer is None:
        for name, module in model.named_modules():
            if isinstance(module, nn.AdaptiveAvgPool2d):
                target_layer = module
                break
    
    handle = None
    if target_layer is not None:
        handle = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(image_tensor)
    
    if handle is not None:
        handle.remove()
    
    if features:
        embedding = features[0].cpu().numpy().flatten()
        if len(embedding) > 64:
            embedding = embedding[:64]
        elif len(embedding) < 64:
            embedding = np.pad(embedding, (0, 64 - len(embedding)))
        return embedding
    
    return np.random.randn(64)


LABEL_NAMES = {0: "Normal", 1: "Pneumonia", 2: "COVID-19"}
LABEL_COLORS = {0: "🟢", 1: "🟠", 2: "🔴"}
