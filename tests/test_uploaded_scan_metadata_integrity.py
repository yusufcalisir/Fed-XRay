"""Granular Unit Test Suite: Uploaded Scan Metadata Normalization & SQLite Persistence Parity.

Verifies:
1. Custom uploaded scans (base64 and 2D matrix) receive modality-appropriate metadata normalization
   (e.g. 'Cutaneous Lesion (Uploaded)', biopsy_confirmed=False) rather than null or hardcoded Torso fallbacks.
2. User-specified metadata in DiagnoseRequest (anatomical_site, biopsy_confirmed, provenance) is respected.
3. 100% strict contract equivalence between API response and SQLite audit trail persistence.
4. Database state store records SQL NULL when anatomical_site is None, eliminating fake 'Torso' or 'None' defaults.
"""

from __future__ import annotations
import base64
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import state
from src.fed_xray.storage.state_store import PersistentStateStore


@pytest.fixture
def temp_store(tmp_path):
    """Provide isolated SQLite state store instance for testing."""
    db_file = tmp_path / "test_metadata_consortium.db"
    return PersistentStateStore(db_path=str(db_file))


def _make_dummy_base64_image(width: int = 32, height: int = 32) -> str:
    """Generate a valid base64 grayscale test image."""
    img = Image.new("L", (width, height), color=120)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def test_sqlite_state_store_null_site_and_unconfirmed_biopsy(temp_store: PersistentStateStore):
    """Verify state store stores and retrieves SQL NULL without substituting fake Torso or string None."""
    audit_entry = {
        "patient_id": "PAT_UPLOAD_TEST_01",
        "predicted_class": 0,
        "predicted_name": "Benign Nevus",
        "true_class": -1,
        "true_name": "Uploaded scan (no reference label)",
        "confidence": 92.5,
        "probabilities": [92.5, 5.0, 2.5],
        "findings": "Typical symmetric pigmented network.",
        "image_source": "uploaded",
        "provenance": "uploaded_custom_scan",
        "anatomical_site": None,
        "biopsy_confirmed": False,
    }

    row_id = temp_store.record_diagnostic_audit(audit_entry)
    assert row_id > 0

    last = temp_store.get_last_diagnosis()
    assert last is not None
    assert last["patient_id"] == "PAT_UPLOAD_TEST_01"
    assert last["anatomical_site"] is None, f"Expected None but got {last['anatomical_site']!r}"
    assert last["biopsy_confirmed"] is False
    assert last["provenance"] == "uploaded_custom_scan"

    history = temp_store.get_diagnostic_audits(limit=5)
    assert len(history) == 1
    assert history[0]["anatomical_site"] is None
    assert history[0]["biopsy_confirmed"] is False


def test_sqlite_state_store_preserves_custom_site_and_confirmed_biopsy(temp_store: PersistentStateStore):
    """Verify state store faithfully preserves explicit anatomical site and biopsy verification."""
    audit_entry = {
        "patient_id": "PAT_UPLOAD_TEST_02",
        "predicted_class": 1,
        "predicted_name": "Melanoma",
        "true_class": -1,
        "true_name": "Uploaded scan (no reference label)",
        "confidence": 96.0,
        "probabilities": [2.0, 96.0, 2.0],
        "findings": "Atypical pigment network with irregular borders.",
        "image_source": "uploaded",
        "provenance": "clinical_registry_v3",
        "anatomical_site": "Left Upper Arm",
        "biopsy_confirmed": True,
    }

    temp_store.record_diagnostic_audit(audit_entry)
    last = temp_store.get_last_diagnosis()
    assert last is not None
    assert last["patient_id"] == "PAT_UPLOAD_TEST_02"
    assert last["anatomical_site"] == "Left Upper Arm"
    assert last["biopsy_confirmed"] is True
    assert last["provenance"] == "clinical_registry_v3"


def test_api_diagnose_uploaded_scan_metadata_normalization_and_sqlite_parity(trained_api: TestClient):
    """Verify /api/cdss/diagnose normalizes uploaded scan metadata and maintains 100% SQLite parity."""
    b64_image = _make_dummy_base64_image()
    payload = {
        "image_base64": b64_image,
        "opacity": 0.5,
        "colormap": "Hot",
        "language": "en",
    }

    res = trained_api.post("/api/cdss/diagnose", json=payload)
    assert res.status_code == 200, f"Diagnose request failed: {res.text}"
    data = res.json()

    # Uploaded image must be normalized
    assert data["image_source"] == "uploaded"
    assert data["patient_id"].startswith("PAT_UPLOAD_")
    assert data["anatomical_site"] is not None
    assert "Uploaded" in data["anatomical_site"]
    assert data["biopsy_confirmed"] is False
    assert data["provenance"] == "uploaded_custom_scan"
    assert data["modality"] in ("Dermoscopy", "Histopathology", "Chest X-Ray")

    # Verify SQLite audit trail has exact parity with API response
    last_db = state.store.get_last_diagnosis()
    assert last_db is not None
    assert last_db["patient_id"] == data["patient_id"]
    assert last_db["anatomical_site"] == data["anatomical_site"]
    assert last_db["biopsy_confirmed"] == data["biopsy_confirmed"]
    assert last_db["provenance"] == data["provenance"]

    # Verify /api/diagnose/audit-trail returns identical metadata
    audit_res = trained_api.get("/api/diagnose/audit-trail?limit=1")
    assert audit_res.status_code == 200
    audits = audit_res.json()["audits"]
    assert len(audits) >= 1
    assert audits[0]["patient_id"] == data["patient_id"]
    assert audits[0]["anatomical_site"] == data["anatomical_site"]
    assert audits[0]["biopsy_confirmed"] == data["biopsy_confirmed"]


def test_api_diagnose_uploaded_scan_with_user_specified_metadata(trained_api: TestClient):
    """Verify /api/cdss/diagnose accepts and preserves user-specified clinical metadata."""
    gradient = [[(r * 28 + c) / 783.0 for c in range(28)] for r in range(28)]
    payload = {
        "image_data": gradient,
        "opacity": 0.5,
        "colormap": "Hot",
        "language": "en",
        "anatomical_site": "Right Forearm / Dorsal",
        "biopsy_confirmed": True,
        "provenance": "prospective_clinical_cohort",
    }

    res = trained_api.post("/api/cdss/diagnose", json=payload)
    assert res.status_code == 200, f"Diagnose request failed: {res.text}"
    data = res.json()

    assert data["anatomical_site"] == "Right Forearm / Dorsal"
    assert data["biopsy_confirmed"] is True
    assert data["provenance"] == "prospective_clinical_cohort"

    # SQLite parity verification
    last_db = state.store.get_last_diagnosis()
    assert last_db is not None
    assert last_db["patient_id"] == data["patient_id"]
    assert last_db["anatomical_site"] == "Right Forearm / Dorsal"
    assert last_db["biopsy_confirmed"] is True
    assert last_db["provenance"] == "prospective_clinical_cohort"
