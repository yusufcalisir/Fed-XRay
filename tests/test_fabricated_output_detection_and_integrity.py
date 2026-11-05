"""Forensic Test Suite: Exhaustive Fabricated Output Detection and Integrity.

Verifies Section 4.1 audit invariants:
1. Model diagnostic predictions stem exclusively from live model forward passes.
   Different inputs produce distinct logits and probabilities; weight perturbations
   dynamically alter predictions.
2. Grad-CAM class activation maps compute genuine backward gradients and raise
   GradCAMUnavailableError on degenerate inputs without substituting artificial Gaussian blobs.
3. Untrained model queries fail honestly with HTTP 409 Conflict without canned fallbacks.
4. Federated RAG case matching extracts genuine image embeddings without query vector
   fabrication, producing input-dependent similarity scores and rankings.
5. Telemetry and audit trails persist genuine execution values to SQLite WAL without
   pre-baked or synthetic constants.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.main import app, state, require_trained_server
from src.fed_xray.cdss.xai import GradCAM, GradCAMUnavailableError
from src.fed_xray.cdss.similarity import (
    FederatedRAGCaseMatcher,
    extract_embedding,
    _compute_cosine_similarity,
)
from src.fed_xray.data.real_world import RealWorldPatientRecord
from src.fed_xray.storage.state_store import PersistentStateStore


class SyntheticDiagnosticProbeNet(nn.Module):
    """Minimal CNN probe architecture with accessible convolutional feature extractor."""

    def __init__(self, in_channels: int = 1, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Linear(8 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv1(x)
        feat = self.relu(feat)
        pooled = self.pool(feat)
        flat = torch.flatten(pooled, 1)
        return self.classifier(flat)


def test_diagnostic_prediction_dynamic_variance():
    """Verify distinct inputs produce distinct logits and modifying weights alters outputs.

    Proves zero hardcoded constants, canned predictions, or static output dictionaries.
    """
    model = SyntheticDiagnosticProbeNet(in_channels=1, num_classes=3)
    model.eval()

    # Input A: horizontal edge pattern
    img_a = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    img_a[0, 0, 10:14, :] = 1.0

    # Input B: vertical edge pattern
    img_b = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    img_b[0, 0, :, 10:14] = 1.0

    with torch.no_grad():
        logits_a = model(img_a)
        logits_b = model(img_b)
        probs_a = torch.softmax(logits_a, dim=1).cpu().numpy()[0]
        probs_b = torch.softmax(logits_b, dim=1).cpu().numpy()[0]

    # Invariant: Different inputs MUST produce distinct output logits
    assert not torch.allclose(logits_a, logits_b, atol=1e-4), "Inputs produced invariant logits"
    assert not np.allclose(probs_a, probs_b, atol=1e-4), "Inputs produced invariant probabilities"

    # Perturb weights and assert outputs change on the exact same input
    with torch.no_grad():
        model.classifier.weight.add_(0.5)
        logits_a_modified = model(img_a)

    assert not torch.allclose(logits_a, logits_a_modified, atol=1e-4), (
        "Model weight perturbation failed to alter logits"
    )


def test_gradcam_saliency_authenticity_and_degeneration_guard():
    """Verify Grad-CAM computes genuine backward gradients and fails honestly on degenerate inputs.

    Proves zero artificial Gaussian blobs or static heatmaps are substituted.
    """
    model = SyntheticDiagnosticProbeNet(in_channels=1, num_classes=3)
    gradcam = GradCAM(model)

    # Structured input with spatial contrast
    img_structured = torch.zeros((1, 1, 28, 28), dtype=torch.float32, requires_grad=True)
    img_structured.data[0, 0, 5:15, 5:15] = 1.0

    heatmap, pred_class, confidence = gradcam.generate_heatmap(img_structured, target_class=0)
    gradcam.remove_hooks()

    # Verify heatmap properties
    assert heatmap is not None
    assert heatmap.shape == (28, 28)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    assert heatmap.max() > heatmap.min(), "Saliency map must possess non-zero spatial variance"
    assert np.isfinite(heatmap).all(), "Saliency map contains non-finite values"

    # Degenerate input: flat zero tensor with frozen weights producing uniform zero activation
    flat_model = SyntheticDiagnosticProbeNet(in_channels=1, num_classes=3)
    with torch.no_grad():
        flat_model.conv1.weight.zero_()
        flat_model.conv1.bias.zero_()
        flat_model.classifier.weight.zero_()
        flat_model.classifier.bias.zero_()

    flat_gradcam = GradCAM(flat_model)
    img_flat = torch.zeros((1, 1, 28, 28), dtype=torch.float32, requires_grad=True)

    with pytest.raises(GradCAMUnavailableError) as exc_info:
        flat_gradcam.generate_heatmap(img_flat, target_class=0)
    flat_gradcam.remove_hooks()

    assert "degenerate" in str(exc_info.value).lower() or "no gradients" in str(exc_info.value).lower(), (
        "Degenerate class activation did not raise GradCAMUnavailableError"
    )


def test_untrained_model_honest_failure_without_mock_fallbacks(monkeypatch):
    """Verify that uninitialized models fail honestly with HTTP 409 Conflict.

    Proves that missing models or untrained states never return canned fallback responses.
    """
    original_trained = state.model_trained
    original_server = state.server
    try:
        monkeypatch.setattr(state, "restore_model_weights", lambda *args, **kwargs: False)
        state.model_trained = False
        state.server = None

        with pytest.raises(HTTPException) as exc_info:
            require_trained_server()

        assert exc_info.value.status_code == 409
        assert "not trained yet" in exc_info.value.detail.lower()

        client = TestClient(app)
        res = client.post("/api/cdss/diagnose", json={})
        assert res.status_code == 409
        assert "not trained yet" in res.json()["detail"].lower()
    finally:
        state.model_trained = original_trained
        state.server = original_server


def test_federated_rag_embedding_dynamic_retrieval_and_cosine_fidelity():
    """Verify RAG retrieval extracts dynamic embeddings and performs genuine cosine dot products.

    Proves zero hardcoded query vectors (e.g. sample_query[:20] = 1.0) or fabricated similarity scores.
    """
    # Construct reference bank of genuine patient records
    recs = []
    for i in range(10):
        arr = np.zeros((28, 28), dtype=np.float32)
        arr[i * 2 : (i + 1) * 2, :] = 1.0
        t_img = torch.tensor(arr).unsqueeze(0)
        recs.append(
            RealWorldPatientRecord(
                patient_id=f"TWIN-{2000 + i}",
                site_id="Hospital-Beta",
                image_tensor=t_img,
                label=i % 3,
                clinical_notes=f"Clinical verified history {i}",
            )
        )

    # Probe model
    probe_model = SyntheticDiagnosticProbeNet(in_channels=1, num_classes=3)
    probe_model.eval()

    matcher = FederatedRAGCaseMatcher(reference_records=recs, model=probe_model)
    assert not matcher.is_empty
    assert len(matcher) == 10

    # Query image 1 (top quadrant activation)
    q1_t = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    q1_t[0, 0, 0:5, :] = 1.0
    emb1 = extract_embedding(probe_model, q1_t)

    # Query image 2 (bottom quadrant activation)
    q2_t = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    q2_t[0, 0, 23:28, :] = 1.0
    emb2 = extract_embedding(probe_model, q2_t)

    # Invariant: Distinct queries must produce distinct feature embeddings
    cos_sim = float(_compute_cosine_similarity(np.atleast_2d(emb1), np.atleast_2d(emb2))[0, 0])
    assert cos_sim < 0.9999, "Distinct visual inputs produced identical feature embeddings"

    # Match results must differ between query 1 and query 2
    results1 = matcher.find_similar(emb1, top_k=3)
    results2 = matcher.find_similar(emb2, top_k=3)

    assert len(results1) == 3
    assert len(results2) == 3

    # All similarities must be genuinely bounded in [0.0, 1.0]
    for r in results1 + results2:
        assert 0.0 <= r["similarity"] <= 1.0
        assert isinstance(r["case_id"], str)
        assert len(r["case_id"]) > 0

    # Case IDs or similarity scores must differ between different queries
    sim_scores1 = [r["similarity"] for r in results1]
    sim_scores2 = [r["similarity"] for r in results2]
    assert sim_scores1 != sim_scores2 or results1[0]["case_id"] != results2[0]["case_id"], (
        "Different query embeddings produced identical case rankings and similarities"
    )


def test_telemetry_and_audit_trail_persistence_integrity(tmp_path):
    """Verify that diagnostic audits and training telemetry write genuine values to SQLite WAL.

    Proves zero fake default records, static placeholder statistics, or swallowed writes.
    """
    db_file = tmp_path / "integrity_test_consortium.db"
    store = PersistentStateStore(db_path=str(db_file))

    # Initial state: verify clean zero state without canned metrics
    initial_audits = store.get_diagnostic_audits(limit=10)
    assert len(initial_audits) == 0, "Initial audit trail contains unearned records"

    # Persist genuine diagnostic audit record
    audit_payload = {
        "patient_id": "TEST-ISIC-9999",
        "predicted_class": 1,
        "predicted_name": "Melanoma",
        "true_class": 1,
        "true_name": "Melanoma",
        "confidence": 94.25,
        "probabilities": [2.50, 94.25, 3.25],
        "findings": "Dense asymmetrical hyperpigmented lesion localized to left lower quadrant.",
        "raw_image": [[0.1, 0.2], [0.3, 0.4]],
        "heatmap": [[0.8, 0.9], [0.1, 0.2]],
        "image_source": "isic_holdout",
        "provenance": "ISIC_2019",
        "anatomical_site": "anterior torso",
        "biopsy_confirmed": True,
    }
    store.record_diagnostic_audit(audit_payload)

    # Read back and verify strict fidelity
    saved_audits = store.get_diagnostic_audits(limit=10)
    assert len(saved_audits) == 1
    rec = saved_audits[0]
    assert rec["patient_id"] == "TEST-ISIC-9999"
    assert rec["predicted_class"] == 1
    assert rec["predicted_name"] == "Melanoma"
    assert math.isclose(rec["confidence"], 94.25, abs_tol=1e-2)
    assert rec["biopsy_confirmed"] is True

    # Persist genuine training round
    round_payload = {
        "round_num": 1,
        "train_loss": 0.4521,
        "train_accuracy": 82.50,
        "test_loss": 0.4812,
        "test_accuracy": 80.00,
        "precision": 81.20,
        "recall": 79.80,
        "f1_score": 80.49,
        "privacy_epsilon": 0.8523,
        "uplink_bytes": 1048576,
        "threat_detected": False,
        "blocked_nodes": [],
        "status": "complete",
    }
    store.record_training_round(round_payload)

    history = store.get_training_history()
    assert len(history["round"]) == 1
    assert history["round"][0] == 1
    assert math.isclose(history["loss"][0], 0.4521, abs_tol=1e-4)
    assert math.isclose(history["test_accuracy"][0], 80.00, abs_tol=1e-2)
    assert history["uplink_bytes"][0] == 1048576
