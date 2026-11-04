"""Test suite verifying multi-class oncology modality support and range enforcements.

Validates:
1. /api/cdss/report-pdf accepts classes >= 3 (e.g. classes 3 through 7 in ISIC 2019) without HTTP 422.
2. get_dataset_label_map provides complete class ontologies for ISIC 2019 (8 classes) and NCT-CRC-HE (9 classes).
3. /api/cdss/diagnose honors class_index >= 3 for multi-class holdout evaluation.
4. Cohort generation formats authentic modality counts (nevus, melanoma, carcinoma) for dermoscopy.
"""

from __future__ import annotations
import numpy as np
import torch
from fastapi.testclient import TestClient

from api.main import app, state
from src.fed_xray.cdss.similarity import (
    get_dataset_label_map,
    ISIC_LABEL_NAMES,
    NCT_CRC_HE_LABEL_NAMES,
    CHEST_XRAY_LABEL_NAMES,
)


client = TestClient(app)


def test_dataset_label_maps_cover_all_classes():
    """Verify Strategy E ontologies are complete and accurately mapped."""
    isic_map = get_dataset_label_map("ISIC_2019")
    assert isic_map == ISIC_LABEL_NAMES
    assert len(isic_map) == 8
    assert isic_map[0] == "Benign Dysplastic Nevus"
    assert isic_map[1] == "Malignant Melanoma"
    assert isic_map[2] == "Basal Cell Carcinoma"
    assert isic_map[3] == "Actinic Keratosis"
    assert isic_map[7] == "Squamous Cell Carcinoma"

    crc_map = get_dataset_label_map("NCT_CRC_HE")
    assert crc_map == NCT_CRC_HE_LABEL_NAMES
    assert len(crc_map) == 9
    assert crc_map[0] == "Adipose Tissue"
    assert crc_map[8] == "Colorectal Adenocarcinoma Epithelium"

    cxr_map = get_dataset_label_map("MIMIC_CXR")
    assert cxr_map == CHEST_XRAY_LABEL_NAMES
    assert len(cxr_map) == 3
    assert cxr_map[0] == "Normal"
    assert cxr_map[1] == "Bacterial Pneumonia"
    assert cxr_map[2] == "COVID-19"


def test_pdf_report_accepts_classes_above_two():
    """Verify /api/cdss/report-pdf accepts multi-class predictions without HTTP 422 error."""
    # Ensure state has a valid scan and probabilities
    state.last_scan = np.random.rand(224, 224).astype(np.float32)
    state.last_probabilities = [12.5] * 8
    state.last_patient_id = "PAT_ISIC_MULTI_003"
    state.data_source = "isic_archive_v2_real_clinical"

    # Test class 3 (Actinic Keratosis)
    res_3 = client.get(
        "/api/cdss/report-pdf",
        params={
            "patient_id": "PAT_ISIC_MULTI_003",
            "predicted_class": 3,
            "confidence": 88.5,
            "findings": "Test multi-class dermoscopy evaluation findings.",
            "lang": "en",
        },
    )
    assert res_3.status_code == 200, f"Failed with {res_3.status_code}: {res_3.text}"
    assert res_3.headers["content-type"] == "application/pdf"
    assert len(res_3.content) > 1000

    # Test class 7 (Squamous Cell Carcinoma)
    res_7 = client.get(
        "/api/cdss/report-pdf",
        params={
            "patient_id": "PAT_ISIC_MULTI_007",
            "predicted_class": 7,
            "confidence": 92.1,
            "findings": "Test SCC dermoscopy evaluation findings.",
            "lang": "tr",
        },
    )
    assert res_7.status_code == 200, f"Failed with {res_7.status_code}: {res_7.text}"
    assert res_7.headers["content-type"] == "application/pdf"
    assert len(res_7.content) > 1000


def test_diagnose_honors_class_index_above_two(trained_api: TestClient):
    """Verify /api/cdss/diagnose respects class_index >= 3 when filtering test samples."""
    # Seed state global_test_set with multi-class samples matching model image size
    img_dim = state.image_size or 224
    test_imgs = torch.randn(10, 1, img_dim, img_dim)
    test_lbls = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 3], dtype=torch.long)
    state.global_test_set = (test_imgs, test_lbls)
    state.data_source = "isic_archive_v2_real_clinical"

    res = trained_api.post(
        "/api/cdss/diagnose",
        json={"class_index": 3, "language": "en"},
    )
    assert res.status_code == 200, f"Failed with {res.status_code}: {res.text}"
    data = res.json()
    assert data["true_class"] == 3
    assert data["true_name"] == "Actinic Keratosis"


def test_cohort_generation_contains_authentic_dermoscopy_keys():
    """Verify /api/cohorts/generate formats authentic dermoscopy counts rather than solely pulmonary keys."""
    res = client.post(
        "/api/cohorts/generate",
        json={"num_hospitals": 2, "samples_per_hospital": 50, "scenario": "A", "dataset_name": "ISIC_2019"},
    )
    assert res.status_code == 200, f"Failed with {res.status_code}: {res.text}"
    data = res.json()
    assert "hospitals" in data and len(data["hospitals"]) > 0
    h1 = data["hospitals"][0]
    counts = h1.get("counts", {})
    # Verify authentic dermoscopy keys are populated
    assert "nevus" in counts or "benign_dysplastic_nevus" in counts
    assert "melanoma" in counts or "malignant_melanoma" in counts
    assert "carcinoma" in counts or "basal_cell_carcinoma" in counts


def test_central_server_dynamic_confusion_matrix_8class_isic():
    """Verify DEFECT-12 fix: evaluate_on_test_set dynamically dimensions confusion matrix to (8, 8)."""
    from src.fed_xray.core.server import CentralServer

    server = CentralServer(num_classes=8, model_type="cnn", img_size=28)
    test_imgs = torch.randn(16, 1, 28, 28)
    test_lbls = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)

    metrics = server.evaluate_on_test_set(test_imgs, test_lbls)

    # Confusion matrix must be 8x8, not hardcoded 3x3
    assert metrics.confusion_matrix.shape == (8, 8), f"Expected (8, 8), got {metrics.confusion_matrix.shape}"
    # All 16 samples must be accounted for across all 8 classes without truncation
    assert np.sum(metrics.confusion_matrix) == 16
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0


def test_central_server_dynamic_confusion_matrix_9class_nct_crc():
    """Verify DEFECT-12 fix: evaluate_on_test_set dynamically dimensions confusion matrix to (9, 9)."""
    from src.fed_xray.core.server import CentralServer

    server = CentralServer(num_classes=9, model_type="vit_tiny", img_size=28)
    test_imgs = torch.randn(18, 1, 28, 28)
    test_lbls = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)

    metrics = server.evaluate_on_test_set(test_imgs, test_lbls)

    assert metrics.confusion_matrix.shape == (9, 9), f"Expected (9, 9), got {metrics.confusion_matrix.shape}"
    assert np.sum(metrics.confusion_matrix) == 18
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0


def test_central_server_dynamic_matrix_expands_to_observed_labels():
    """Verify that if num_classes is default (3), evaluate_on_test_set expands dynamically to avoid dropping samples."""
    from src.fed_xray.core.server import CentralServer

    server = CentralServer(num_classes=3, model_type="cnn", img_size=28)
    test_imgs = torch.randn(6, 1, 28, 28)
    # Include classes 5 and 7 which exceed default 3 classes
    test_lbls = torch.tensor([0, 1, 2, 5, 7, 7], dtype=torch.long)

    metrics = server.evaluate_on_test_set(test_imgs, test_lbls)

    # Matrix must dynamically expand to at least 8x8 to accommodate class 7
    assert metrics.confusion_matrix.shape[0] >= 8
    assert metrics.confusion_matrix.shape[1] >= 8
    # All 6 samples must be counted
    assert np.sum(metrics.confusion_matrix) == 6


def test_empty_test_set_preserves_dynamic_matrix_dimension():
    """Verify evaluate_on_test_set handles empty test batches while preserving expected class dimensions."""
    from src.fed_xray.core.server import CentralServer

    server = CentralServer(num_classes=8, model_type="cnn", img_size=28)
    empty_imgs = torch.empty(0, 1, 28, 28)
    empty_lbls = torch.empty(0, dtype=torch.long)

    metrics = server.evaluate_on_test_set(empty_imgs, empty_lbls, num_classes=8)

    assert metrics.confusion_matrix.shape == (8, 8)
    assert np.sum(metrics.confusion_matrix) == 0
    assert metrics.accuracy == 0.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0


def test_run_federated_round_multi_class_evaluation():
    """Verify run_federated_round propagates multi-class evaluation metrics cleanly."""
    from torch.utils.data import DataLoader, TensorDataset
    from src.fed_xray.core.server import CentralServer, run_federated_round
    from src.fed_xray.core.client import HospitalClient

    server = CentralServer(num_classes=8, model_type="cnn", img_size=28)
    train_x = torch.randn(20, 1, 28, 28)
    train_y = torch.randint(0, 8, (20,))
    ds = TensorDataset(train_x, train_y)
    loader = DataLoader(ds, batch_size=10)

    clients = [
        HospitalClient(client_id=0, dataloader=loader, model_type="cnn", num_classes=8),
        HospitalClient(client_id=1, dataloader=loader, model_type="cnn", num_classes=8),
    ]

    test_imgs = torch.randn(16, 1, 28, 28)
    test_lbls = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)

    _, _, test_metrics, _ = run_federated_round(
        server=server,
        clients=clients,
        round_num=1,
        total_rounds=1,
        test_images=test_imgs,
        test_labels=test_lbls,
        num_classes=8,
    )

    assert test_metrics is not None
    assert test_metrics.confusion_matrix.shape == (8, 8)
    assert np.sum(test_metrics.confusion_matrix) == 16

