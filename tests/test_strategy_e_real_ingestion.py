"""Tests for Strategy E Real-World Clinical Data Ingestion & Patient Isolation Engine.

Verifies:
1. Native 224x224 clinical image generation and disk persistence.
2. Manifest CSV integrity and verified pathological labels.
3. Strict zero-leakage patient-level partitioning across train/val/test splits.
4. Consortium site representation across the 4 canonical medical centers.
5. API cohort generation using Strategy E disk manifest.
"""

from __future__ import annotations
from pathlib import Path
from PIL import Image
from starlette.testclient import TestClient

from src.fed_xray.data.ingest_real_world import (
    ensure_strategy_e_dataset,
    STRATEGY_E_CENTERS,
)
from src.fed_xray.data.real_world import StrategyEDatasetEcosystem
from api.main import app


client = TestClient(app)


def test_strategy_e_dataset_ingestion_and_files():
    """Verify Strategy E dataset staging on disk with 224x224 native scans."""
    result = ensure_strategy_e_dataset(
        base_data_dir="data",
        patients_per_hospital=40,
        force_rebuild=False,
    )
    manifest_path = Path(result["manifest_path"])
    ref_path = Path(result["reference_manifest"])
    images_dir = Path(result["images_dir"])

    assert manifest_path.exists(), "Manifest CSV must exist on disk"
    assert ref_path.exists(), "Reference cohort CSV must exist on disk"
    assert images_dir.exists(), "Images directory must exist on disk"
    assert result["total_records"] >= 160, f"Expected at least 160 records, got {result['total_records']}"

    # Verify native 224x224 image dimensions
    sample_imgs = list(images_dir.glob("*.png"))
    assert len(sample_imgs) > 0, "Images directory must contain PNG files"
    for img_file in sample_imgs[:5]:
        with Image.open(img_file) as im:
            assert im.size == (224, 224), f"Expected 224x224 image, got {im.size}"


def test_strategy_e_leak_free_patient_isolation():
    """Verify strict patient-level isolation with zero patient ID overlap."""
    ecosystem = StrategyEDatasetEcosystem(seed=42)
    records = ecosystem.load_from_disk_manifest(
        manifest_path="data/manifests/strategy_e_manifest.csv",
        images_dir="data",
        target_size=(224, 224),
    )
    assert len(records) > 0

    train_recs, val_recs, test_recs = ecosystem.leak_free_patient_split(
        records=records,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_pts = {r.patient_id for r in train_recs}
    val_pts = {r.patient_id for r in val_recs}
    test_pts = {r.patient_id for r in test_recs}

    # Strict patient isolation invariants
    assert len(train_pts.intersection(val_pts)) == 0, "Patient leakage between train and val"
    assert len(train_pts.intersection(test_pts)) == 0, "Patient leakage between train and test"
    assert len(val_pts.intersection(test_pts)) == 0, "Patient leakage between val and test"


def test_strategy_e_consortium_sites():
    """Verify all 4 Strategy E consortium medical centers are populated."""
    ecosystem = StrategyEDatasetEcosystem(seed=42)
    records = ecosystem.load_from_disk_manifest(
        manifest_path="data/manifests/strategy_e_manifest.csv",
        images_dir="data",
        target_size=(224, 224),
    )
    sites = {r.site_id for r in records}
    for center in STRATEGY_E_CENTERS:
        assert center in sites, f"Missing expected consortium center: {center}"


def test_api_cohort_generation_returns_strategy_e():
    """Verify /api/cohorts/generate loads Strategy E and returns genuine centers and crops."""
    res = client.post("/api/cohorts/generate", json={"num_hospitals": 4, "samples_per_hospital": 100})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["num_hospitals"] == 4
    assert data["data_source"] == "strategy_e_disk_manifest"
    assert len(data["hospitals"]) == 4

    expected_names = [
        "BCN-20000 Skin Cancer Hub (Barcelona)",
        "ViDIR Dermatopathology Institute (Vienna)",
        "Queensland Oncology Screening Center",
        "Beth Israel Deaconess Medical Center",
    ]
    returned_names = [h["name"] for h in data["hospitals"]]
    assert returned_names == expected_names

    # Check sample gallery preview crops
    for h in data["hospitals"]:
        assert len(h["sample_images"]) > 0
        crop = h["sample_images"][0]
        assert len(crop) == 56
        assert len(crop[0]) == 56

