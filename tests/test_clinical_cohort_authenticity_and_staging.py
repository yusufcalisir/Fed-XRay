"""Clinical Cohort Authenticity and Strategy E Data Staging Verification Suite.

Verifies strict real-world data isolation, forensic cohort authenticity auditing,
genuine patient metadata derivation, and honest non-clinical tagging on calibration suites.
"""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
from fastapi.testclient import TestClient

from api.main import app
from src.fed_xray.data.ingest_real_world import (
    STRATEGY_E_CENTERS,
    ensure_strategy_e_dataset,
    generate_in_silico_calibration_cohort,
    stage_manifest_from_local_isic_scans,
    verify_cohort_authenticity,
)
from src.fed_xray.data.real_world import (
    DiskMedicalDataset,
    RealWorldPatientRecord,
    StrategyEDatasetEcosystem,
)


def test_in_silico_calibration_cohort_metadata_integrity(tmp_path: Path):
    """Verify that in-silico calibration cohorts are honestly tagged and non-clinical."""
    result = generate_in_silico_calibration_cohort(
        base_data_dir=tmp_path,
        patients_per_hospital=4,
        seed=42,
    )

    assert result["is_authentic_clinical"] is False
    assert result["provenance"] == "in_silico_offline_calibration_suite"
    assert result["total_records"] == 4 * len(STRATEGY_E_CENTERS)

    manifest_csv = Path(result["manifest_path"])
    assert manifest_csv.exists()

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 16
    for r in rows:
        assert r["provenance"] == "in_silico_offline_calibration_suite"
        assert r["is_authentic_clinical"] in ("False", "false", False)
        assert r["biopsy_confirmed"] in ("False", "false", False)
        assert "In Silico Phantom" in r["anatomical_site"]
        # Verify generated image exists and is 224x224
        img_p = tmp_path / r["image_path"]
        assert img_p.exists()
        with Image.open(img_p) as im:
            assert im.size == (224, 224)

    # Reference cohort verification
    ref_csv = Path(result["reference_manifest"])
    assert ref_csv.exists()
    with open(ref_csv, "r", encoding="utf-8") as f:
        ref_rows = list(csv.DictReader(f))

    for rr in ref_rows:
        assert rr["provenance"] == "in_silico_offline_calibration_suite"
        assert rr["is_authentic_clinical"] in ("False", "false", False)
        assert rr["biopsy_confirmed"] in ("False", "false", False)
        assert "Non-clinical" in rr["clinical_notes"] or "In silico" in rr["clinical_notes"]


def test_stage_manifest_from_local_isic_scans(tmp_path: Path):
    """Verify staging authentic ISIC manifest from pre-staged local image files."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create 8 local test scans simulating authentic 224x224 ISIC images
    for idx in range(8):
        im = Image.fromarray(np.full((224, 224), fill_value=128 + idx, dtype=np.uint8))
        im.save(images_dir / f"isic_{idx:07d}.png")

    result = stage_manifest_from_local_isic_scans(
        base_data_dir=tmp_path,
        patients_per_hospital=2,
        seed=42,
    )

    assert result["status"] == "staged_from_local_cache"
    assert result["is_authentic_clinical"] is True
    assert result["provenance"] == "isic_archive_v2_real_clinical"
    assert result["total_records"] == 8

    manifest_file = Path(result["manifest_path"])
    with open(manifest_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 8
    for r in rows:
        assert r["provenance"] == "isic_archive_v2_real_clinical"
        assert r["is_authentic_clinical"] in ("True", "true", True)


def test_verify_cohort_authenticity(tmp_path: Path):
    """Verify forensic authenticity auditor distinguishes clinical from in-silico cohorts."""
    # 1. Non-existent manifest
    report_none = verify_cohort_authenticity(tmp_path / "does_not_exist.csv")
    assert report_none["integrity_verified"] is False
    assert report_none["is_authentic_clinical"] is False

    # 2. In-silico cohort
    generate_in_silico_calibration_cohort(base_data_dir=tmp_path, patients_per_hospital=2)
    report_calib = verify_cohort_authenticity(
        tmp_path / "manifests/strategy_e_manifest.csv",
        images_dir=tmp_path,
    )
    assert report_calib["integrity_verified"] is True
    assert report_calib["is_authentic_clinical"] is False
    assert report_calib["in_silico_records"] == 8
    assert report_calib["clinical_records"] == 0
    assert report_calib["missing_images"] == 0
    assert report_calib["verification_status"] == "IN_SILICO_CALIBRATION"

    # 3. Authentic staged ISIC cohort
    images_dir = tmp_path / "images"
    for idx in range(8):
        im = Image.fromarray(np.full((224, 224), fill_value=128 + idx, dtype=np.uint8))
        im.save(images_dir / f"isic_{idx:07d}.png")

    stage_manifest_from_local_isic_scans(base_data_dir=tmp_path, patients_per_hospital=2)
    report_auth = verify_cohort_authenticity(
        tmp_path / "manifests/strategy_e_manifest.csv",
        images_dir=tmp_path,
    )
    assert report_auth["integrity_verified"] is True
    assert report_auth["is_authentic_clinical"] is True
    assert report_auth["clinical_records"] == 8
    assert report_auth["in_silico_records"] == 0
    assert report_auth["verification_status"] == "AUTHENTIC_CLINICAL"


def test_strict_real_data_raises_on_synthetic_or_network_failure(tmp_path: Path):
    """Verify strict real data mode blocks silent fallback to in-silico phantoms."""
    # Ensure images_dir has no local isic images
    (tmp_path / "images").mkdir(parents=True, exist_ok=True)

    # Force network download failure
    with patch(
        "src.fed_xray.data.ingest_real_world.download_and_stage_isic_real_cohort",
        side_effect=RuntimeError("Simulated offline network / DNS resolution failure"),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            ensure_strategy_e_dataset(
                base_data_dir=tmp_path,
                patients_per_hospital=2,
                strict_real_data=True,
                allow_in_silico_fallback=False,
            )

        assert "In-silico procedural generation is strictly prohibited" in str(exc_info.value)

    # Verify no manifest was fabricated under strict mode
    assert not (tmp_path / "manifests/strategy_e_manifest.csv").exists()


def test_disk_medical_dataset_strict_real_data_guard(tmp_path: Path):
    """Verify DiskMedicalDataset raises ValueError when loading in-silico data under strict mode."""
    # Generate in-silico manifest
    calib = generate_in_silico_calibration_cohort(base_data_dir=tmp_path, patients_per_hospital=2)

    # Loading without strict mode succeeds and records is_authentic_clinical=False
    dataset_permissive = DiskMedicalDataset(
        manifest_csv=calib["manifest_path"],
        images_dir=str(tmp_path),
        strict_real_data=False,
    )
    records = dataset_permissive.to_patient_records()
    assert len(records) > 0
    assert records[0].is_authentic_clinical is False

    # Loading with strict_real_data=True must fail immediately
    with pytest.raises(ValueError) as exc_info:
        DiskMedicalDataset(
            manifest_csv=calib["manifest_path"],
            images_dir=str(tmp_path),
            strict_real_data=True,
        )

    assert "Strict real data violation" in str(exc_info.value)

    # StrategyEDatasetEcosystem forwarding test
    ecosystem = StrategyEDatasetEcosystem(seed=42)
    with pytest.raises(ValueError) as exc_info2:
        ecosystem.load_from_disk_manifest(
            manifest_path=calib["manifest_path"],
            images_dir=str(tmp_path),
            strict_real_data=True,
        )
    assert "Strict real data violation" in str(exc_info2.value)


def test_real_world_patient_record_authenticity_derivation():
    """Verify RealWorldPatientRecord derives and respects is_authentic_clinical."""
    dummy_tensor = torch.zeros(1, 224, 224)

    # Case 1: Authentic ISIC clinical provenance
    r1 = RealWorldPatientRecord(
        patient_id="PAT_001",
        site_id="BCN",
        image_tensor=dummy_tensor,
        label=1,
        provenance="isic_archive_v2_real_clinical",
    )
    assert r1.is_authentic_clinical is True

    # Case 2: In-silico phantom provenance
    r2 = RealWorldPatientRecord(
        patient_id="PAT_002",
        site_id="Simulation",
        image_tensor=dummy_tensor,
        label=0,
        provenance="in_silico_offline_calibration_suite",
    )
    assert r2.is_authentic_clinical is False

    # Case 3: Explicit boolean override
    r3 = RealWorldPatientRecord(
        patient_id="PAT_003",
        site_id="Custom",
        image_tensor=dummy_tensor,
        label=2,
        provenance="custom_hospital_import",
        is_authentic_clinical=True,
    )
    assert r3.is_authentic_clinical is True


def test_fastapi_cohort_authenticity_telemetry():
    """Verify FastAPI /api/state and /api/cohorts/verify-authenticity return genuine audit data."""
    client = TestClient(app)

    # 1. State endpoint must return data_provenance and is_authentic_clinical
    state_resp = client.get("/api/state")
    assert state_resp.status_code == 200
    state_data = state_resp.json()
    assert "is_authentic_clinical" in state_data
    assert "data_provenance" in state_data

    # 2. Authenticity verification endpoint
    auth_resp = client.get("/api/cohorts/verify-authenticity")
    assert auth_resp.status_code == 200
    auth_data = auth_resp.json()
    assert "verification_status" in auth_data
    assert "clinical_records" in auth_data
    assert "in_silico_records" in auth_data
    assert "integrity_verified" in auth_data
    assert "strict_real_data_enforced" in auth_data
    # Staged dataset in data/ has 160 real clinical ISIC scans
    assert auth_data["is_authentic_clinical"] is True
    assert auth_data["clinical_records"] >= 160
    assert auth_data["verification_status"] == "AUTHENTIC_CLINICAL"

    # 3. Cohort generation with strict_real_data=True succeeds on authentic data
    gen_resp = client.post(
        "/api/cohorts/generate",
        json={"num_hospitals": 4, "scenario": "C", "strict_real_data": True},
    )
    assert gen_resp.status_code == 200
    gen_data = gen_resp.json()
    assert gen_data["success"] is True
    assert gen_data["is_authentic_clinical"] is True
    assert gen_data["provenance"] == "isic_archive_v2_real_clinical"
    for h in gen_data["hospitals"]:
        assert h["is_authentic_clinical"] is True


def test_strict_real_data_environment_variable_enforcement(tmp_path: Path, monkeypatch):
    """Verify that FED_XRAY_STRICT_REAL_DATA=1 enforces strict clinical data without explicit flag."""
    monkeypatch.setenv("FED_XRAY_STRICT_REAL_DATA", "1")
    (tmp_path / "images").mkdir(parents=True, exist_ok=True)

    with patch(
        "src.fed_xray.data.ingest_real_world.download_and_stage_isic_real_cohort",
        side_effect=RuntimeError("Simulated offline air-gapped environment"),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            ensure_strategy_e_dataset(
                base_data_dir=tmp_path,
                patients_per_hospital=2,
            )

        assert "In-silico procedural generation is strictly prohibited" in str(exc_info.value)


def test_offline_local_authentic_cache_failsafe(tmp_path: Path):
    """Verify that when network is offline, existing local authentic scans are staged successfully."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Pre-stage 8 authentic ISIC scans locally
    for idx in range(8):
        im = Image.fromarray(np.full((224, 224), fill_value=100 + idx, dtype=np.uint8))
        im.save(images_dir / f"isic_{idx:07d}.png")

    # Simulate network download failure during a forced rebuild
    with patch(
        "src.fed_xray.data.ingest_real_world.download_and_stage_isic_real_cohort",
        side_effect=RuntimeError("Air-gapped clinical network isolation"),
    ):
        result = ensure_strategy_e_dataset(
            base_data_dir=tmp_path,
            patients_per_hospital=2,
            force_rebuild=True,
            strict_real_data=True,
        )

    assert result["is_authentic_clinical"] is True
    assert result["provenance"] == "isic_archive_v2_real_clinical"
    assert result["total_records"] == 8


def test_fastapi_strict_real_data_toggle_endpoint():
    """Verify runtime toggling of strict authentic clinical data enforcement via API."""
    client = TestClient(app)

    # 1. Enable strict mode
    enable_resp = client.post("/api/cohorts/set-strict-mode?enabled=true")
    assert enable_resp.status_code == 200
    assert enable_resp.json()["strict_real_data"] is True

    auth_resp = client.get("/api/cohorts/verify-authenticity")
    assert auth_resp.status_code == 200
    assert auth_resp.json()["strict_real_data_enforced"] is True

    # 2. Re-disable strict mode for test neutrality
    disable_resp = client.post("/api/cohorts/set-strict-mode?enabled=false")
    assert disable_resp.status_code == 200
    assert disable_resp.json()["strict_real_data"] is False
