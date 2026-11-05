"""Tests for Section 4.3: Failure Masking and Error-to-Success Conversion Elimination.

Verifies that failures, invalid inputs, uninitialized states, and runtime exceptions
propagate authentic HTTP status codes (400, 404, 409, 422, 500, 503, 504) and structured
error payloads rather than converting errors into fake successes or default predictions.
"""

from __future__ import annotations

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from api.main import app, require_cohort, state
from src.fed_xray.core.client import HospitalClient


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


def test_untrained_model_diagnose_throws_409(client, monkeypatch):
    """Verify inference request without trained model raises HTTP 409, never fake predictions."""
    original_trained = state.model_trained
    original_server = state.server
    try:
        monkeypatch.setattr(state, "restore_model_weights", lambda checkpoint_dir="checkpoints": False)
        state.model_trained = False
        state.server = None

        payload = {"class_index": 0}
        response = client.post("/api/cdss/diagnose", json=payload)

        assert response.status_code == 409, (
            f"Expected HTTP 409 Conflict for untrained model, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert "not trained" in data["detail"].lower(), "Error detail must state model is not trained"
    finally:
        state.model_trained = original_trained
        state.server = original_server


def test_invalid_base64_payload_throws_400(client):
    """Verify corrupted image base64 input returns HTTP 400 Bad Request, never fallback inference."""
    original_trained = state.model_trained
    original_server = state.server
    try:
        state.model_trained = True

        payload = {"image_base64": "invalid_base64_garbage_!!!_###"}
        response = client.post("/api/cdss/diagnose", json=payload)

        assert response.status_code == 400, (
            f"Expected HTTP 400 Bad Request for corrupted base64, got {response.status_code}: {response.text}"
        )
        assert "invalid base64" in response.json()["detail"].lower()
    finally:
        state.model_trained = original_trained
        state.server = original_server


def test_invalid_image_matrix_dimensions_throws_error(client):
    """Verify malformed image matrix input returns client rejection error, never fake inference."""
    original_trained = state.model_trained
    original_server = state.server
    try:
        state.model_trained = True

        # Provide malformed input: 1D array instead of 2D image matrix
        payload = {"image_data": [1.0, 2.0, 3.0, 4.0]}
        response = client.post("/api/cdss/diagnose", json=payload)

        assert response.status_code in (400, 422), (
            f"Expected HTTP 400 or 422 for malformed matrix, got {response.status_code}: {response.text}"
        )
    finally:
        state.model_trained = original_trained
        state.server = original_server


def test_uninitialized_cohort_require_throws_409():
    """Verify require_cohort raises HTTP 409 when cohorts have not been loaded."""
    original_dataloaders = state.dataloaders
    original_test_set = state.global_test_set
    try:
        state.dataloaders = None
        state.global_test_set = None

        with pytest.raises(HTTPException) as exc_info:
            require_cohort()

        assert exc_info.value.status_code == 409
        assert "no cohort is loaded" in exc_info.value.detail.lower()
    finally:
        state.dataloaders = original_dataloaders
        state.global_test_set = original_test_set


def test_background_training_failure_preserves_error_state(client):
    """Verify task failure records task_error and guarantees is_complete remains False."""
    original_running = state.task_running
    original_complete = state.task_complete
    original_error = state.task_error
    try:
        state.task_running = False
        state.task_complete = True
        state.task_error = "Simulated distributed out-of-memory error"

        response = client.get("/api/fl/train-status")
        assert response.status_code == 200
        data = response.json()

        assert data["error"] == "Simulated distributed out-of-memory error"
        assert data["is_complete"] is False, (
            "When task_error is present, is_complete must be False to eliminate failure masking"
        )
    finally:
        state.task_running = original_running
        state.task_complete = original_complete
        state.task_error = original_error


def test_client_image_size_fallback_to_standard_medical_resolution():
    """Verify HospitalClient defaults to standard medical resolution 224 if dataloader cannot be read."""
    # Empty dataset triggers DataLoader inspection exception
    empty_dataset = TensorDataset(torch.empty(0, 1, 224, 224), torch.empty(0, dtype=torch.long))
    empty_loader = DataLoader(empty_dataset, batch_size=1)

    client_node = HospitalClient(
        client_id=1,
        dataloader=empty_loader,
        model_type="vit_tiny",
        num_classes=3,
        peft_mode="ffa_lora",
        img_size=None,  # triggers inference try-block
    )

    assert client_node.img_size == 224, (
        f"Expected standard medical ViT resolution 224, but got {client_node.img_size}"
    )


def test_pdf_report_without_diagnosed_scan_throws_409(client):
    """Verify PDF report download without prior diagnostic scan raises HTTP 409."""
    original_scan = state.last_scan
    try:
        state.last_scan = None

        response = client.get(
            "/api/cdss/report-pdf",
            params={
                "patient_id": "PT-TEST-001",
                "predicted_class": 0,
                "confidence": 95.0,
                "findings": "Test findings text",
                "lang": "en",
            },
        )

        assert response.status_code == 409, (
            f"Expected HTTP 409 Conflict when no scan is available, got {response.status_code}"
        )
        assert "no diagnosed scan is available" in response.json()["detail"].lower()
    finally:
        state.last_scan = original_scan

