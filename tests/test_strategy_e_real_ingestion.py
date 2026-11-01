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
    ingest_isic_dataset,
    ingest_mimic_cxr_dataset,
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


def test_strategy_e_modality_coherence():
    """Verify clinical notes strictly match imaging modality without cross-contamination."""
    import csv
    manifest_path = Path("data/manifests/strategy_e_manifest.csv")
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            modality = row["modality"]
            notes = row["clinical_notes"].lower()

            if modality == "Dermoscopy":
                # Must not contain pulmonary or chest terms
                assert "sars-cov-2" not in notes
                assert "air bronchograms" not in notes
                assert "pleural effusion" not in notes
                assert "pulmonary" not in notes
            elif modality == "Chest Radiography":
                # Must not contain dermatological pigment network terms
                assert "pigment network" not in notes
                assert "melanocytic" not in notes
                assert "dermatoscopy" not in notes


def test_ingest_isic_dataset_pipeline(tmp_path: Path):
    """Verify genuine ISIC dataset ingestion pipeline with deduplication and mapping."""
    import csv
    images_dir = tmp_path / "isic_images"
    images_dir.mkdir()
    meta_csv = tmp_path / "isic_meta.csv"
    out_manifest = tmp_path / "strategy_e_isic_out.csv"

    # Create dummy image files
    im1 = Image.new("RGB", (224, 224), color=(200, 150, 100))
    im1.save(images_dir / "ISIC_0000001.jpg")
    im2 = Image.new("RGB", (224, 224), color=(50, 50, 50))
    im2.save(images_dir / "ISIC_0000002.jpg")

    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "MEL", "NV", "BCC", "patient_id", "lesion_id"])
        writer.writeheader()
        writer.writerow({"image": "ISIC_0000001", "MEL": "0.0", "NV": "1.0", "BCC": "0.0", "patient_id": "IP_101", "lesion_id": "IL_101"})
        writer.writerow({"image": "ISIC_0000002", "MEL": "1.0", "NV": "0.0", "BCC": "0.0", "patient_id": "IP_102", "lesion_id": "IL_102"})

    res = ingest_isic_dataset(
        metadata_csv_path=meta_csv,
        images_dir_path=images_dir,
        output_manifest_path=out_manifest,
    )
    assert res["status"] == "ingested"
    assert res["total_records"] == 2
    assert out_manifest.exists()

    with open(out_manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["label"] == "0"
        assert rows[1]["label"] == "1"
        assert rows[0]["modality"] == "Dermoscopy"
        assert rows[1]["modality"] == "Dermoscopy"


def test_ingest_mimic_cxr_dataset_pipeline(tmp_path: Path):
    """Verify genuine MIMIC-CXR dataset ingestion pipeline with mapping and metadata."""
    import csv
    images_dir = tmp_path / "mimic_images"
    images_dir.mkdir()
    meta_csv = tmp_path / "mimic_meta.csv"
    out_manifest = tmp_path / "strategy_e_mimic_out.csv"

    im1 = Image.new("L", (224, 224), color=120)
    im1.save(images_dir / "CXR_001.jpg")
    im2 = Image.new("L", (224, 224), color=80)
    im2.save(images_dir / "CXR_002.jpg")

    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dicom_id", "Pneumonia", "Consolidation", "subject_id", "study_id"])
        writer.writeheader()
        writer.writerow({"dicom_id": "CXR_001", "Pneumonia": "1.0", "Consolidation": "0.0", "subject_id": "SUB_01", "study_id": "STD_01"})
        writer.writerow({"dicom_id": "CXR_002", "Pneumonia": "0.0", "Consolidation": "0.0", "subject_id": "SUB_02", "study_id": "STD_02"})

    res = ingest_mimic_cxr_dataset(
        metadata_csv_path=meta_csv,
        images_dir_path=images_dir,
        output_manifest_path=out_manifest,
    )
    assert res["status"] == "ingested"
    assert res["total_records"] == 2
    assert out_manifest.exists()

    with open(out_manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["label"] == "1"
        assert rows[1]["label"] == "0"
        assert rows[0]["modality"] == "Chest Radiography"
        assert rows[1]["modality"] == "Chest Radiography"

