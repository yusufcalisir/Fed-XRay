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


def test_biopsy_reference_cohort_loading_and_matching():
    """Verify genuine Strategy E biopsy reference digital twin cohort loads and indexes into RAG."""
    from pathlib import Path
    from src.fed_xray.data.real_world import StrategyEDatasetEcosystem

    ref_csv = Path("data/reference_cohort/strategy_e_rag_reference.csv")
    assert ref_csv.exists(), "strategy_e_rag_reference.csv must exist on disk"

    ecosystem = StrategyEDatasetEcosystem(seed=42)
    records = ecosystem.load_from_disk_manifest(
        manifest_path=str(ref_csv),
        images_dir="data",
        target_size=(224, 224),
    )
    assert len(records) >= 12, f"Expected at least 12 reference twins, got {len(records)}"

    # Verify all are biopsy confirmed and have therapeutic outcomes
    for r in records:
        assert r.biopsy_confirmed is True, f"Record {r.patient_id} must be biopsy confirmed"
        assert r.therapeutic_outcome is not None and len(r.therapeutic_outcome) > 10
        assert r.anatomical_site is not None and len(r.anatomical_site) > 2
        assert r.clinical_notes is not None and "Histopathology" in r.clinical_notes

    # Verify all 3 classes are represented
    classes = {r.label for r in records}
    assert classes == {0, 1, 2}, f"All 3 diagnostic classes must be present, found {classes}"

    # Build matcher and query with one record
    matcher = FederatedRAGCaseMatcher(reference_records=records)
    assert not matcher.is_empty
    assert len(matcher) == len(records)

    query_tensor = records[0].image_tensor.unsqueeze(0)
    matches = matcher.find_similar_for_image(query_tensor, top_k=2)
    assert len(matches) == 2
    assert matches[0]["biopsy_confirmed"] is True
    assert matches[0]["therapeutic_outcome"] is not None
    assert matches[0]["anatomical_site"] is not None
    assert "diagnosis" in matches[0]


def test_dynamic_dimension_preservation_and_mismatch_guard():
    """Verify matcher dynamically adopts model dimension and rejects mismatched vectors."""
    import pytest
    from src.fed_xray.models.vit import create_medical_vit

    recs = _make_test_patient_records(8)
    model = create_medical_vit(model_type="vit_tiny", img_size=28, patch_size=4, in_channels=1, num_classes=3)
    matcher = FederatedRAGCaseMatcher(reference_records=recs, model=model)

    assert matcher.embedding_dim == 128, f"Expected 128-dim for ViT-Tiny, got {matcher.embedding_dim}"
    assert matcher.embeddings_matrix.shape == (8, 128)

    # Valid 128-dim query should succeed
    valid_query = np.random.randn(128).astype(np.float32)
    results = matcher.find_similar(valid_query, top_k=2)
    assert len(results) == 2

    # Incompatible 64-dim query must raise ValueError
    incompatible_query = np.random.randn(64).astype(np.float32)
    with pytest.raises(ValueError, match="Query embedding dimension mismatch"):
        matcher.find_similar(incompatible_query, top_k=2)


def test_cnn_extract_features_contract():
    """Verify XRayClassifier provides penultimate 128-dim feature representation."""
    from src.fed_xray.models.cnn import create_cnn_model
    from src.fed_xray.cdss.similarity import extract_embedding

    model = create_cnn_model(num_classes=3)
    assert hasattr(model, "extract_features"), "XRayClassifier must implement extract_features"

    x = torch.randn(2, 1, 28, 28)
    feats = model.extract_features(x)
    assert feats.shape == (2, 128), f"Expected (2, 128), got {feats.shape}"

    emb = extract_embedding(model, x[:1])
    assert emb.shape == (128,), f"Expected 128-dim normalized embedding, got {emb.shape}"
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-4


def test_dataset_label_mappings():
    """Verify ISIC and Chest X-ray label mappings are distinct and clinically accurate."""
    from src.fed_xray.cdss.similarity import (
        ISIC_LABEL_NAMES,
        CHEST_XRAY_LABEL_NAMES,
        get_dataset_label_map,
    )

    assert ISIC_LABEL_NAMES[0] == "Benign Dysplastic Nevus"
    assert ISIC_LABEL_NAMES[1] == "Malignant Melanoma"
    assert ISIC_LABEL_NAMES[2] == "Basal Cell Carcinoma"

    assert CHEST_XRAY_LABEL_NAMES[0] == "Normal"
    assert CHEST_XRAY_LABEL_NAMES[1] == "Bacterial Pneumonia"
    assert CHEST_XRAY_LABEL_NAMES[2] == "COVID-19"

    assert get_dataset_label_map("isic_archive_v2") == ISIC_LABEL_NAMES
    assert get_dataset_label_map("strategy_e_disk_manifest") == ISIC_LABEL_NAMES
    assert get_dataset_label_map("mimic_cxr_jpg") == CHEST_XRAY_LABEL_NAMES


def test_rag_matcher_backend_selection_and_telemetry():
    """Verify explicit selection between exact NumPy cosine and approximate HNSW backends."""
    from src.fed_xray.cdss.similarity import HNSW_AVAILABLE

    recs = _make_test_patient_records(12)

    # 1. Exact NumPy backend
    exact_matcher = FederatedRAGCaseMatcher(reference_records=recs, backend="exact_numpy")
    assert exact_matcher.active_engine == "exact_numpy_cosine"
    telemetry = exact_matcher.get_index_telemetry()
    assert telemetry["active_engine"] == "exact_numpy_cosine"
    assert telemetry["is_exact_search"] is True
    assert telemetry["total_cases"] == 12

    # 2. Runtime backend switching
    if HNSW_AVAILABLE:
        exact_matcher.set_backend("hnsw")
        assert exact_matcher.active_engine == "hnsw"
        assert exact_matcher.get_index_telemetry()["is_exact_search"] is False

        exact_matcher.set_backend("exact_numpy")
        assert exact_matcher.active_engine == "exact_numpy_cosine"
        assert exact_matcher.get_index_telemetry()["is_exact_search"] is True


def test_rag_matcher_strict_hnsw_failure_when_unavailable(monkeypatch):
    """Verify that requesting HNSW backend when hnswlib is unavailable raises honest RuntimeError."""
    import pytest
    import src.fed_xray.cdss.similarity as sim_mod

    monkeypatch.setattr(sim_mod, "HNSW_AVAILABLE", False)
    monkeypatch.setattr(sim_mod, "hnswlib", None)

    recs = _make_test_patient_records(5)
    with pytest.raises(RuntimeError) as exc_info:
        FederatedRAGCaseMatcher(reference_records=recs, backend="hnsw")

    assert "Requested HNSW backend" in str(exc_info.value)
    assert "hnswlib C++ extension is unavailable" in str(exc_info.value)


def test_rag_matcher_search_consistency_verification():
    """Verify mathematical ranking consistency between HNSW index and exact NumPy cosine."""
    recs = _make_test_patient_records(20)
    matcher = FederatedRAGCaseMatcher(reference_records=recs, backend="auto")

    query_vec = matcher.embeddings_matrix[0]
    report = matcher.verify_search_consistency(query_vec, top_k=5)

    assert report["consistency_verified"] is True
    assert report["top1_match"] is True
    assert report["rank_overlap"] >= 0.8
    assert report["max_absolute_error"] < 1e-3


def test_rag_matcher_save_and_load_index(tmp_path):
    """Verify serialization and re-loading of RAG index embeddings and metadata."""
    recs = _make_test_patient_records(10)
    original_matcher = FederatedRAGCaseMatcher(reference_records=recs, backend="exact_numpy")

    save_dir = tmp_path / "rag_index_cache"
    saved_files = original_matcher.save_index(save_dir)
    assert "embeddings" in saved_files
    assert "metadata" in saved_files
    assert (save_dir / "embeddings.npy").exists()
    assert (save_dir / "metadata.json").exists()

    # Create fresh empty matcher and load persisted state
    restored_matcher = FederatedRAGCaseMatcher(reference_records=None, backend="exact_numpy")
    loaded_count = restored_matcher.load_index(save_dir)

    assert loaded_count == 10
    assert len(restored_matcher) == 10
    assert restored_matcher.embedding_dim == original_matcher.embedding_dim

    # Verify query results on original and restored matchers are identical
    query_vec = original_matcher.embeddings_matrix[2]
    res_orig = original_matcher.find_similar(query_vec, top_k=3)
    res_rest = restored_matcher.find_similar(query_vec, top_k=3)

    assert len(res_orig) == len(res_rest) == 3
    for o, r in zip(res_orig, res_rest):
        assert o["case_id"] == r["case_id"]
        assert abs(o["similarity"] - r["similarity"]) < 1e-5


def test_fastapi_rag_status_and_backend_endpoints():
    """Verify FastAPI endpoints for RAG index status and runtime backend switching."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)

    # 1. Query RAG index status
    status_resp = client.get("/api/cdss/rag-status")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert "active_engine" in data
    assert "hnsw_available" in data
    assert "metric" in data

    # 2. Configure RAG backend
    backend_resp = client.post("/api/cdss/rag-backend?backend=exact_numpy")
    assert backend_resp.status_code == 200
    assert backend_resp.json()["configured_backend"] == "exact_numpy"



