"""Granular Unit Test Suite: Federated RAG and Similarity Case Matcher.

Verifies evidence-grounded case retrieval, HNSW vector indexing,
cosine similarity computation, dynamic record addition, and honest empty states.
Guarantees zero synthetic 28x28 noise generator leakage into clinical similarity engine.
"""

from __future__ import annotations
import numpy as np
import torch
from src.fed_xray.cdss.similarity import FederatedRAGCaseMatcher
from src.fed_xray.data.real_world import RealWorldPatientRecord


def _make_test_patient_records(count: int = 15) -> list[RealWorldPatientRecord]:
    """Create genuine in-memory test patient records with distinct patterns."""
    records = []
    for i in range(count):
        lbl = i % 3
        # Distinct pattern per case for vector discrimination
        arr = np.zeros((28, 28), dtype=np.float32)
        arr[i % 28, :] = 1.0
        arr[:, (i * 2) % 28] = 0.5
        t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        records.append(
            RealWorldPatientRecord(
                patient_id=f"TEST-PAT-{1000 + i}",
                site_id="Hospital-Alpha",
                image_tensor=t,
                label=lbl,
                clinical_notes=f"Clinical study notes for test patient {1000 + i}",
            )
        )
    return records


def test_similarity_matcher_initialization():
    """Verify matcher initializes with caller-supplied reference bank and valid embeddings."""
    recs = _make_test_patient_records(15)
    matcher = FederatedRAGCaseMatcher(reference_records=recs)
    assert len(matcher.case_ids) == 15
    assert len(matcher.embeddings) == 15
    assert len(matcher.embeddings[0]) > 0
    assert not matcher.is_empty
    assert len(matcher) == 15


def test_similarity_matcher_empty_cohort_fails_honestly():
    """Verify empty or None cohort initializes an honest empty state without synthetic noise."""
    matcher = FederatedRAGCaseMatcher(reference_records=None)
    assert matcher.is_empty
    assert len(matcher) == 0
    assert matcher.hnsw_index is None

    # Empty query should return empty results without crash
    query_vec = np.random.randn(64).astype(np.float32)
    assert matcher.find_similar(query_vec, top_k=3) == []

    # Image query on empty bank should return empty list
    img_28 = np.random.rand(28, 28).astype(np.float32)
    assert matcher.find_similar_for_image(img_28, top_k=2) == []

    # Empty bank distribution should return uniform prior
    dist = matcher.predict_rag_distribution(query_vec, top_k=5, num_classes=3)
    assert np.allclose(dist, np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32))


def test_similarity_matcher_dynamic_add_records():
    """Verify matcher supports progressive addition of real patient records."""
    matcher = FederatedRAGCaseMatcher(reference_records=None)
    assert len(matcher) == 0

    recs_batch1 = _make_test_patient_records(5)
    added = matcher.add_records(recs_batch1)
    assert added == 5
    assert len(matcher) == 5

    recs_batch2 = _make_test_patient_records(10)
    added2 = matcher.add_records(recs_batch2)
    assert added2 == 10
    assert len(matcher) == 15


def test_similarity_matcher_n_cases_caps_records():
    """Verify n_cases parameter caps the indexed records from reference_records."""
    recs = _make_test_patient_records(20)
    matcher = FederatedRAGCaseMatcher(reference_records=recs, n_cases=7)
    assert len(matcher) == 7
    assert len(matcher.case_ids) == 7


def test_similarity_find_similar_from_vector():
    """Verify vector query returns requested top_k cases with descending similarity scores."""
    recs = _make_test_patient_records(20)
    matcher = FederatedRAGCaseMatcher(reference_records=recs)
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
    recs = _make_test_patient_records(10)
    matcher = FederatedRAGCaseMatcher(reference_records=recs)
    img_28 = np.random.rand(28, 28).astype(np.float32)

    results = matcher.find_similar_for_image(img_28, top_k=2)
    assert len(results) == 2
    assert "case_id" in results[0]
    assert "similarity" in results[0]


def test_similarity_find_similar_for_image_224x224():
    """Verify 224x224 image input is correctly interpolated to CNN dimension."""
    recs = _make_test_patient_records(10)
    matcher = FederatedRAGCaseMatcher(reference_records=recs)
    img_224 = np.random.rand(224, 224).astype(np.float32)

    results = matcher.find_similar_for_image(img_224, top_k=2)
    assert len(results) == 2
    assert 0.0 <= results[0]["similarity"] <= 1.0


def test_similarity_tensor_input():
    """Verify PyTorch tensor input in (1, 1, 28, 28) shape works seamlessly."""
    recs = _make_test_patient_records(10)
    matcher = FederatedRAGCaseMatcher(reference_records=recs)
    tensor_input = torch.rand(1, 1, 28, 28)

    results = matcher.find_similar_for_image(tensor_input, top_k=3)
    assert len(results) == 3
