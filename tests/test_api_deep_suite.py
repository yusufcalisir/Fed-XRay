"""Exhaustive Deep Integration & Stress Testing Suite for FastAPI Endpoints."""

import json
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_index_detailed():
    """Verify root index status, service metadata, and architecture reflection."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "active_architecture" in data
    assert "active_peft" in data
    assert "version" in data


def test_health_check_endpoint():
    """Verify health check endpoint returns 200 and device information."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "cuda_available" in data


def test_state_endpoint_consistency():
    """Verify /api/state tracks application state."""
    res = client.get("/api/state")
    assert res.status_code == 200
    data = res.json()
    assert "model_trained" in data
    assert "hospitals_loaded" in data
    assert "has_test_set" in data


@pytest.mark.parametrize("scenario", ["A", "B", "C", "D", "E", "F", "G"])
def test_cohort_generation_all_scenarios(scenario: str):
    """Verify Strategy E cohort generation across all 7 non-IID scenarios."""
    payload = {
        "num_hospitals": 4,
        "samples_per_hospital": 80,
        "scenario": scenario,
        "dataset_name": "ISIC_2019",
    }
    res = client.post("/api/cohorts/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["num_hospitals"] == 4
    assert len(data["hospitals"]) == 4
    for h in data["hospitals"]:
        assert h["num_samples"] > 0
        assert len(h["distribution"]) == 3
        assert sum(h["distribution"]) > 0.99  # Normalizes to 1.0


def test_cohort_generation_validation_errors():
    """Verify parameter validation rejection on invalid requests."""
    # Hospital count out of bounds (<2 or >8)
    res = client.post("/api/cohorts/generate", json={"num_hospitals": 1, "samples_per_hospital": 50})
    assert res.status_code == 422

    res2 = client.post("/api/cohorts/generate", json={"num_hospitals": 10, "samples_per_hospital": 50})
    assert res2.status_code == 422


def test_sse_streaming_training_events():
    """Verify /api/fl/train-stream Server-Sent Events stream yields valid JSON for each round."""
    res = client.get("/api/fl/train-stream?num_rounds=3&local_epochs=1&learning_rate=0.001&algorithm=FedDyn&loss_fn=DAFL&model_type=vit_tiny&peft_mode=ffa_lora")
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]

    lines = res.text.strip().split("\n\n")
    assert len(lines) >= 3  # At least 3 rounds

    received_rounds = []
    for line in lines:
        if line.startswith("data:"):
            json_str = line.replace("data:", "").strip()
            data = json.loads(json_str)
            assert "round_num" in data
            assert "train_loss" in data
            assert "test_accuracy" in data
            assert "threat_detected" in data
            assert "status" in data
            received_rounds.append(data["round_num"])

    assert received_rounds == [1, 2, 3]


def test_sse_streaming_validation_bounds():
    """Verify input validation bounds on train-stream parameters."""
    # num_rounds > 20 should fail with 422
    res = client.get("/api/fl/train-stream?num_rounds=25")
    assert res.status_code == 422

    # num_rounds < 1 should fail with 422
    res2 = client.get("/api/fl/train-stream?num_rounds=0")
    assert res2.status_code == 422


@pytest.mark.parametrize("class_idx", [0, 1, 2, None])
def test_cdss_diagnose_class_variations(class_idx):
    """Verify radiological diagnosis and Grad-CAM generation across classes."""
    payload = {"class_index": class_idx, "opacity": 0.6, "colormap": "Jet"}
    res = client.post("/api/cdss/diagnose", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] in (0, 1, 2)
    assert 0.0 <= data["confidence"] <= 100.0
    assert len(data["probabilities"]) == 3
    assert len(data["heatmap"]) == 28
    assert len(data["heatmap"][0]) == 28
    assert len(data["raw_image"]) == 28


@pytest.mark.parametrize("query_class", [0, 1, 2, 99])
def test_rag_similar_cases_retrieval(query_class: int):
    """Verify Federated RAG digital twin retrieval with fallback handling."""
    res = client.get(f"/api/cdss/rag-similar?query_class={query_class}")
    assert res.status_code == 200
    data = res.json()
    assert "matched_cases" in data
    assert len(data["matched_cases"]) == 2
    for c in data["matched_cases"]:
        assert 0.0 <= c["similarity"] <= 100.0
        assert len(c["history"]) > 0


def test_pdf_report_generation_custom_findings():
    """Verify single-page medical PDF generation with custom parameters and findings."""
    params = {
        "patient_id": "PX-CLINICAL-884",
        "predicted_class": 2,
        "confidence": 98.2,
        "findings": "Bilateral multifocal ground-glass opacities with peripheral lung distribution.",
        "lang": "en",
    }
    res = client.get("/api/cdss/report-pdf", params=params)
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    # PDF magic bytes verification
    assert res.content.startswith(b"%PDF-")
    assert len(res.content) > 1000
