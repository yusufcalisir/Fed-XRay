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
    assert data["data_source"] == "synthetic_phantom"
    for h in data["hospitals"]:
        assert h["name"].startswith("Synthetic Node")
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
    cohort = client.post(
        "/api/cohorts/generate",
        json={"num_hospitals": 3, "samples_per_hospital": 75, "scenario": "A", "dataset_name": "ISIC_2019"},
    )
    assert cohort.status_code == 200
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
            assert data["full_model_bytes"] > 0
            assert 0 < data["uplink_bytes_per_client"] <= data["full_model_bytes"]
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
def test_cdss_diagnose_class_variations(trained_api, class_idx):
    """Verify diagnosis on held-out test samples and Grad-CAM generation across classes."""
    payload = {"class_index": class_idx, "opacity": 0.6, "colormap": "Jet"}
    res = trained_api.post("/api/cdss/diagnose", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] in (0, 1, 2)
    assert 0.0 <= data["confidence"] <= 100.0
    assert len(data["probabilities"]) == 3
    assert abs(sum(data["probabilities"]) - 100.0) < 0.1
    assert data["image_source"].startswith("holdout_test_set:")
    assert data["true_class"] in (0, 1, 2)
    if class_idx is not None:
        assert data["true_class"] == class_idx
    assert data["heatmap_available"] is True, f"Heatmap failed: {data.get('heatmap_error')}"
    assert len(data["heatmap"]) == 28
    assert len(data["heatmap"][0]) == 28
    assert len(data["raw_image"]) == 28
    assert "not a clinical diagnosis" in data["findings"]


def test_rag_similar_cases_retrieval(diagnosed_api):
    """Verify retrieval embeds the last diagnosed scan and returns reference cohort cases."""
    res = diagnosed_api.get("/api/cdss/rag-similar?top_k=3")
    assert res.status_code == 200
    data = res.json()
    assert data["query_class"] in (0, 1, 2)
    assert len(data["matched_cases"]) == 3
    sims = [c["similarity"] for c in data["matched_cases"]]
    assert sims == sorted(sims, reverse=True)
    for c in data["matched_cases"]:
        assert 0.0 <= c["similarity"] <= 100.0
        assert c["case_id"].startswith("PAT_")
        assert len(c["history"]) > 0


def test_cdss_endpoints_fail_honestly_without_trained_model(monkeypatch):
    """Without a trained model the CDSS must return 409 instead of fabricating predictions."""
    from api.main import state

    monkeypatch.setattr(state, "model_trained", False)
    assert client.post("/api/cdss/diagnose", json={}).status_code == 409
    assert client.get("/api/cdss/rag-similar").status_code == 409


def test_training_requires_loaded_cohort(monkeypatch):
    """Training must not silently create a synthetic cohort when none is loaded."""
    from api.main import state

    monkeypatch.setattr(state, "dataloaders", None)
    res = client.get("/api/fl/train-stream?num_rounds=1")
    assert res.status_code == 409
    assert client.post("/api/fl/train-start?num_rounds=1").status_code == 409


def test_pdf_requires_explicit_parameters_and_scan(monkeypatch):
    """The report endpoint has no default patient, confidence or findings, and needs a diagnosed scan."""
    from api.main import state

    assert client.get("/api/cdss/report-pdf").status_code == 422
    monkeypatch.setattr(state, "last_scan", None)
    res = client.get(
        "/api/cdss/report-pdf",
        params={"patient_id": "PX-1", "predicted_class": 1, "confidence": 50.0, "findings": "text"},
    )
    assert res.status_code == 409


def test_manifest_requires_configured_data_dir(monkeypatch):
    """Disk manifests are only readable inside FED_XRAY_DATA_DIR; unset configuration is an error."""
    monkeypatch.delenv("FED_XRAY_DATA_DIR", raising=False)
    res = client.post(
        "/api/cohorts/generate",
        json={"num_hospitals": 2, "samples_per_hospital": 50, "manifest_name": "manifest.csv"},
    )
    assert res.status_code == 400


def test_manifest_path_traversal_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("FED_XRAY_DATA_DIR", str(tmp_path))
    res = client.post(
        "/api/cohorts/generate",
        json={"num_hospitals": 2, "samples_per_hospital": 50, "manifest_name": "../outside.csv"},
    )
    assert res.status_code == 400


def test_pdf_report_generation_custom_findings(diagnosed_api):
    """Verify single-page medical PDF generation with custom parameters and findings."""
    params = {
        "patient_id": "PX-CLINICAL-884",
        "predicted_class": 2,
        "confidence": 98.2,
        "findings": "Bilateral multifocal ground-glass opacities with peripheral lung distribution.",
        "lang": "en",
    }
    res = diagnosed_api.get("/api/cdss/report-pdf", params=params)
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    # PDF magic bytes verification
    assert res.content.startswith(b"%PDF-")
    assert len(res.content) > 1000


def test_cdss_diagnose_with_matrix_data(trained_api):
    """Verify diagnosis endpoint accepting raw 2D image matrix."""
    gradient = [[(r * 28 + c) / 783.0 for c in range(28)] for r in range(28)]
    payload = {"image_data": gradient, "opacity": 0.5, "colormap": "Hot"}
    res = trained_api.post("/api/cdss/diagnose", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] in (0, 1, 2)
    assert data["true_class"] == -1
    assert data["image_source"] == "uploaded"
    assert len(data["raw_image"]) == 28


def test_cdss_diagnose_with_base64_image(trained_api):
    """Verify diagnosis endpoint accepting base64-encoded radiological scan."""
    import io
    import base64
    from PIL import Image

    img = Image.new("L", (32, 32), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    payload = {"image_base64": b64_str, "opacity": 0.6, "colormap": "Jet"}
    res = trained_api.post("/api/cdss/diagnose", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] in (0, 1, 2)
    assert len(data["raw_image"]) == 28
    # A constant image has no class activation structure: the API must say so instead of faking a map.
    if data["heatmap_available"]:
        assert len(data["heatmap"]) == 28
    else:
        assert data["heatmap"] is None
        assert data["heatmap_error"]


def test_cdss_diagnose_invalid_inputs(trained_api):
    """Verify rejection of malformed image inputs."""
    res = trained_api.post("/api/cdss/diagnose", json={"image_data": [1, 2, 3]})
    assert res.status_code == 422

    res2 = trained_api.post("/api/cdss/diagnose", json={"image_base64": "not-valid-base64-content!!!"})
    assert res2.status_code == 400
