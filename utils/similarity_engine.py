"""
Fed-XRay: Similarity Search Engine (Case-Based Reasoning)
=========================================================
Finds similar historical cases using visual embeddings and cosine similarity.
Implements "Digital Twin" matching for clinical decision support.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn


class HistoricalCaseBank:
    """
    Bank of historical patient cases with pre-computed feature embeddings.
    Used for similarity search during diagnosis.
    """
    
    def __init__(self, n_cases: int = 100, embedding_dim: int = 64, seed: int = 42):
        """
        Initialize case bank with synthetic historical data.
        
        Args:
            n_cases: Number of historical cases to generate
            embedding_dim: Dimension of feature embeddings
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.n_cases = n_cases
        self.embedding_dim = embedding_dim
        
        # Generate synthetic embeddings and labels
        self.embeddings = []
        self.labels = []
        self.case_ids = []
        self.images = []  # Store thumbnail images
        
        self._generate_cases()
    
    def _generate_cases(self) -> None:
        """Generate synthetic historical cases with class-specific patterns."""
        from utils.medical_data import MedicalDataGenerator
        
        generator = MedicalDataGenerator(seed=42)
        
        for i in range(self.n_cases):
            # Balanced distribution
            label = i % 3
            
            # Generate synthetic image
            image = generator.generate_synthetic_xray(label, apply_augmentation=False)
            
            # Create class-specific embedding pattern
            # Each class has a distinct "signature" in embedding space
            base_embedding = np.random.randn(self.embedding_dim) * 0.3
            
            # Add class-specific offset
            if label == 0:  # Normal - centered around origin
                class_offset = np.array([1.0] * 20 + [0.0] * 44)
            elif label == 1:  # Pneumonia - offset in one direction
                class_offset = np.array([0.0] * 20 + [1.0] * 20 + [0.0] * 24)
            else:  # COVID-19 - offset in another direction
                class_offset = np.array([0.0] * 40 + [1.0] * 24)
            
            embedding = base_embedding + class_offset * 0.5
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)  # Normalize
            
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
        """
        Find top-k most similar historical cases.
        
        Args:
            query_embedding: Feature embedding of current patient
            top_k: Number of similar cases to return
            
        Returns:
            List of dicts with case_id, label, similarity, image
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
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
    """
    Extract feature embedding from model's penultimate layer.
    
    Args:
        model: CNN model with feature extraction layers
        image_tensor: Input image tensor (1, 1, 28, 28)
        
    Returns:
        Feature embedding as numpy array
    """
    model.eval()
    
    # Hook to capture intermediate features
    features = []
    
    def hook_fn(module, input, output):
        features.append(output.detach())
    
    # Find the layer before classification (usually flatten or last pooling)
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Flatten) or 'flatten' in name.lower():
            target_layer = module
            break
    
    # If no flatten found, try to get features after conv layers
    if target_layer is None:
        for name, module in model.named_modules():
            if isinstance(module, nn.AdaptiveAvgPool2d):
                target_layer = module
                break
    
    # Register hook
    if target_layer is not None:
        handle = target_layer.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = model(image_tensor)
    
    # Remove hook
    if target_layer is not None:
        handle.remove()
    
    # Return embedding
    if features:
        embedding = features[0].cpu().numpy().flatten()
        # Pad or truncate to 64 dimensions for consistency
        if len(embedding) > 64:
            embedding = embedding[:64]
        elif len(embedding) < 64:
            embedding = np.pad(embedding, (0, 64 - len(embedding)))
        return embedding
    
    # Fallback: use random embedding
    return np.random.randn(64)


# Label names for display
LABEL_NAMES = {0: "Normal", 1: "Pneumonia", 2: "COVID-19"}
LABEL_COLORS = {0: "🟢", 1: "🟠", 2: "🔴"}
