"""Unit and integration tests for Strategy E Real-World Data Migration, Scenarios A-G, and Federated RAG."""

import pytest
import numpy as np
import torch

from src.fed_xray.data.real_world import (
    RealWorldPatientRecord,
    StrategyEDatasetEcosystem,
)
from src.fed_xray.cdss.similarity import (
    FederatedRAGCaseMatcher,
    extract_embedding,
)
from src.fed_xray.models.cnn import create_model as create_cnn_model
from src.fed_xray.models.vit import create_medical_vit


def test_real_world_cohort_generation_and_sha256():
    """Verify Strategy E multi-center patient cohort generation and SHA-256 integrity."""
    ecosystem = StrategyEDatasetEcosystem(seed=42)
    records = ecosystem.generate_synthetic_realworld_cohort(
        dataset_name="ISIC_2019",
        num_patients=50,
        samples_per_patient=2,
        num_classes=3,
    )

    assert len(records) > 0
    # Verify all records have valid SHA-256 digests
    hashes = [r.sha256 for r in records]
    assert len(hashes) == len(records)
    assert all(len(h) == 64 for h in hashes)


def test_leak_free_patient_level_isolation():
    """Verify strict patient isolation invariant: P_train intersect P_test = empty."""
    ecosystem = StrategyEDatasetEcosystem(seed=42)
    records = ecosystem.generate_synthetic_realworld_cohort(num_patients=60, samples_per_patient=3)

    train_rec, val_rec, test_rec = ecosystem.leak_free_patient_split(
        records=records,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    train_patients = {r.patient_id for r in train_rec}
    val_patients = {r.patient_id for r in val_rec}
    test_patients = {r.patient_id for r in test_rec}

    # Strict disjointness invariant
    assert len(train_patients.intersection(val_patients)) == 0, "Train and Val share patient IDs (Data Leakage!)"
    assert len(train_patients.intersection(test_patients)) == 0, "Train and Test share patient IDs (Data Leakage!)"
    assert len(val_patients.intersection(test_patients)) == 0, "Val and Test share patient IDs (Data Leakage!)"


def test_seven_imbalance_scenarios_a_to_g():
    """Verify partition generation across all 7 controlled imbalance scenarios (A through G)."""
    ecosystem = StrategyEDatasetEcosystem(seed=42)
    records = ecosystem.generate_synthetic_realworld_cohort(num_patients=80, samples_per_patient=2)

    for scenario_code in ["A", "B", "C", "D", "E", "F", "G"]:
        partitions = ecosystem.partition_into_scenarios(
            records=records,
            num_clients=4,
            scenario=scenario_code,
            num_classes=3,
        )

        assert len(partitions) == 4, f"Scenario {scenario_code} must generate 4 client partitions"
        for k in range(4):
            assert len(partitions[k]) > 0, f"Scenario {scenario_code} client {k} partition is empty"
            loader = ecosystem.records_to_dataloader(partitions[k], batch_size=8)
            batch_x, batch_y = next(iter(loader))
            assert batch_x.shape[0] > 0
            assert batch_y.shape[0] > 0


def test_federated_rag_case_matching_and_distribution():
    """Verify Federated RAG digital twin search and outcome distribution estimation."""
    rag_matcher = FederatedRAGCaseMatcher(n_cases=50, embedding_dim=64, temperature=0.1)
    query_emb = np.random.randn(64).astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb)

    results = rag_matcher.find_similar(query_emb, top_k=3)
    assert len(results) == 3
    for res in results:
        assert "case_id" in res
        assert "similarity" in res
        assert "biopsy_finding" in res
        assert 0.0 <= res["similarity"] <= 1.0

    prob_dist = rag_matcher.predict_rag_distribution(query_emb, top_k=5, num_classes=3)
    assert prob_dist.shape == (3,)
    assert abs(np.sum(prob_dist) - 1.0) < 1e-4
    assert np.all(prob_dist >= 0.0)


def test_embedding_extraction_across_architectures():
    """Verify feature embedding extraction works seamlessly across CNN and ViT models."""
    cnn = create_cnn_model()
    vit = create_medical_vit(model_type="vit_tiny", img_size=28, patch_size=4, num_classes=3)

    img_tensor = torch.randn(1, 1, 28, 28)

    emb_cnn = extract_embedding(cnn, img_tensor)
    emb_vit = extract_embedding(vit, img_tensor)

    assert emb_cnn.shape == (64,)
    assert emb_vit.shape == (64,)
    assert not np.isnan(emb_cnn).any()
    assert not np.isnan(emb_vit).any()
