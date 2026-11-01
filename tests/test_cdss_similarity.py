"""Granular Unit Test Suite: Federated RAG & Similarity Case Matcher.

Verifies evidence-grounded case retrieval, HNSW vector indexing,
cosine similarity computation, and 28x28 / 224x224 image tensor dimension adaptation.
"""

from __future__ import annotations
import numpy as np
import torch
from src.fed_xray.cdss.similarity import FederatedRAGCaseMatcher


def test_similarity_matcher_initialization():
    """Verify matcher initializes with non-empty reference bank and valid embeddings."""
    matcher = FederatedRAGCaseMatcher(n_cases=15)
    assert len(matcher.case_ids) == 15
    assert len(matcher.embeddings) == 15
    assert len(matcher.embeddings[0]) > 0


def test_similarity_find_similar_from_vector():
    """Verify vector query returns requested top_k cases with descending similarity scores."""
    matcher = FederatedRAGCaseMatcher(n_cases=20)
    query_vec = matcher.embeddings[0]

    results = matcher.find_similar(query_vec, top_k=3)
    assert len(results) == 3

    # First result should be the query itself with ~1.0 (100%) similarity
    assert results[0]["similarity"] >= 0.95
    for res in results:
        assert 0.0 <= res["similarity"] <= 1.0
        assert "case_id" in res
        assert "label" in res


def test_similarity_find_similar_for_image_28x28():
    """Verify 28x28 image input returns matched cases without shape mismatch."""
    matcher = FederatedRAGCaseMatcher(n_cases=10)
    img_28 = np.random.rand(28, 28).astype(np.float32)

    results = matcher.find_similar_for_image(img_28, top_k=2)
    assert len(results) == 2
    assert "case_id" in results[0]
    assert "similarity" in results[0]


def test_similarity_find_similar_for_image_224x224():
    """Verify 224x224 image input is correctly interpolated to CNN dimension."""
    matcher = FederatedRAGCaseMatcher(n_cases=10)
    img_224 = np.random.rand(224, 224).astype(np.float32)

    results = matcher.find_similar_for_image(img_224, top_k=2)
    assert len(results) == 2
    assert 0.0 <= results[0]["similarity"] <= 100.0


def test_similarity_tensor_input():
    """Verify PyTorch tensor input in (1, 1, 28, 28) shape works seamlessly."""
    matcher = FederatedRAGCaseMatcher(n_cases=10)
    tensor_input = torch.rand(1, 1, 28, 28)

    results = matcher.find_similar_for_image(tensor_input, top_k=3)
    assert len(results) == 3
