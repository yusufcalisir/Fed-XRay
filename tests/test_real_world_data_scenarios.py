"""Unit and integration tests for Strategy E Real-World Data Migration, Scenarios A-G, and Federated RAG."""

import numpy as np
import torch

from src.fed_xray.data.real_world import (
    DiskMedicalDataset,
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


def test_disk_medical_dataset_manifest_and_preprocessing(tmp_path):
    """Verify DiskMedicalDataset reads real image files from disk with manifest and transforms."""
    import csv
    from PIL import Image

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create dummy image files (grayscale and RGB)
    img1_path = img_dir / "case_001.png"
    img2_path = img_dir / "case_002.png"
    Image.new("L", (100, 100), color=128).save(img1_path)
    Image.new("L", (120, 80), color=200).save(img2_path)

    # Create manifest CSV
    manifest_file = tmp_path / "clinical_manifest.csv"
    with open(manifest_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "patient_id", "site_id", "lesion_id", "label", "diagnosis", "clinical_notes"],
        )
        writer.writeheader()
        writer.writerow({
            "image_path": str(img1_path),
            "patient_id": "PAT_001",
            "site_id": "BCN_20000",
            "lesion_id": "LES_001",
            "label": "0",
            "diagnosis": "Normal",
            "clinical_notes": "Clear field.",
        })
        writer.writerow({
            "image_path": str(img2_path),
            "patient_id": "PAT_002",
            "site_id": "ViDIR_Vienna",
            "lesion_id": "LES_002",
            "label": "1",
            "diagnosis": "Pneumonia",
            "clinical_notes": "Right consolidation.",
        })

    dataset = DiskMedicalDataset(
        manifest_csv=str(manifest_file),
        target_size=(224, 224),
        in_channels=1,
        normalize=True,
    )

    assert len(dataset) == 2
    tensor1, label1 = dataset[0]
    tensor2, label2 = dataset[1]

    assert tensor1.shape == (1, 224, 224)
    assert tensor2.shape == (1, 224, 224)
    assert label1 == 0
    assert label2 == 1
    assert not torch.isnan(tensor1).any()

    # Verify patient records conversion
    records = dataset.to_patient_records()
    assert len(records) == 2
    assert records[0].patient_id == "PAT_001"
    assert records[0].site_id == "BCN_20000"
    assert len(records[0].sha256) == 64


def test_strategy_e_manifest_template_generation(tmp_path):
    """Verify Strategy E manifest template generation writes valid clinical CSV structure."""
    out_csv = tmp_path / "isic_template.csv"
    generated_path = StrategyEDatasetEcosystem.create_manifest_template(
        output_csv_path=str(out_csv),
        dataset_name="ISIC_2019",
        num_samples=6,
    )

    assert str(out_csv) == generated_path
    assert out_csv.exists()

    import csv
    with open(out_csv, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 6
        first = reader[0]
        assert "image_path" in first
        assert "patient_id" in first
        assert "site_id" in first
        assert "label" in first
        assert "diagnosis" in first
        assert first["site_id"] in ["BCN_20000_Barcelona", "ViDIR_Vienna", "Univ_Queensland_Brisbane"]


def test_load_from_disk_manifest_e2e(tmp_path):
    """Verify end-to-end load from disk manifest into Strategy E leak-free partitioning."""
    import csv
    from PIL import Image

    img_dir = tmp_path / "cohort_images"
    img_dir.mkdir()

    # Generate 12 sample images for 4 patients (3 images per patient)
    rows = []
    for p_idx in range(4):
        pat_id = f"PAT_DISK_{p_idx:03d}"
        for s_idx in range(3):
            file_name = f"{pat_id}_sample_{s_idx}.png"
            file_path = img_dir / file_name
            color_val = 50 + p_idx * 40 + s_idx * 5
            Image.new("L", (64, 64), color=color_val).save(file_path)
            rows.append({
                "image_path": str(file_path),
                "patient_id": pat_id,
                "site_id": f"Hospital_{p_idx % 2}",
                "lesion_id": f"LES_{pat_id}",
                "label": p_idx % 3,
                "diagnosis": f"Class_{p_idx % 3}",
                "clinical_notes": "Clinical validation case.",
            })

    manifest_file = tmp_path / "cohort_manifest.csv"
    with open(manifest_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ecosystem = StrategyEDatasetEcosystem(seed=42)
    records = ecosystem.load_from_disk_manifest(
        manifest_path=str(manifest_file),
        target_size=(28, 28),
        in_channels=1,
    )

    assert len(records) == 12

    # Verify leak-free partitioning on real disk records
    train_r, val_r, test_r = ecosystem.leak_free_patient_split(
        records=records,
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
    )

    train_pts = {r.patient_id for r in train_r}
    val_pts = {r.patient_id for r in val_r}
    test_pts = {r.patient_id for r in test_r}

    assert len(train_pts.intersection(val_pts)) == 0
    assert len(train_pts.intersection(test_pts)) == 0
    assert len(val_pts.intersection(test_pts)) == 0
