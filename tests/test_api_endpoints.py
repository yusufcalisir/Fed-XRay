"""Integration tests for Fed-XRay FastAPI Async Endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_and_health_endpoints():
    """Verify root index and health endpoints return 200 OK."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "healthy"

    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"


def test_cohort_generation_endpoint():
    """Verify multi-hospital cohort generation endpoint adhering to Strategy E."""
    payload = {
        "num_hospitals": 4,
        "samples_per_hospital": 100,
        "scenario": "A",
        "dataset_name": "ISIC_2019",
    }
    res = client.post("/api/cohorts/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["num_hospitals"] == 4
    assert len(data["hospitals"]) == 4


def test_cdss_diagnose_endpoint():
    """Verify diagnostic inference and Grad-CAM heatmap generation."""
    payload = {"class_index": 1, "opacity": 0.5, "colormap": "Hot"}
    res = client.post("/api/cdss/diagnose", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "heatmap" in data
    assert len(data["heatmap"]) == 28


def test_cdss_rag_similar_endpoint():
    """Verify evidence-grounded Federated RAG case matching."""
    res = client.get("/api/cdss/rag-similar?query_class=1")
    assert res.status_code == 200
    data = res.json()
    assert "matched_cases" in data
    assert len(data["matched_cases"]) == 2
    assert "case_id" in data["matched_cases"][0]


def test_pdf_report_endpoint():
    """Verify medical PDF report generation and binary streaming."""
    res = client.get("/api/cdss/report-pdf?patient_id=PX-TEST&predicted_class=1&confidence=95.0&lang=en")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert len(res.content) > 1000
