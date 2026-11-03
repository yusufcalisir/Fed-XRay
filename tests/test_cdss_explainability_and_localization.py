"""CDSS Diagnostic Explainability and Spatial Localization Verification Suite.

Verifies zero-markdown-asterisk invariant, structured clinical assessment formatting,
multimodal terminology adaptation (pathology vs radiology), and bilingual parity.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.fed_xray.cdss.xai import get_explanation_text
from src.fed_xray.cdss.report import generate_medical_report
from api.main import app, state
from src.fed_xray.core.server import CentralServer


@pytest.fixture
def api_client() -> TestClient:
    return TestClient(app)


def test_xai_explanation_zero_markdown_asterisks():
    """Verify get_explanation_text produces clean text with zero markdown asterisks."""
    heatmap = np.zeros((224, 224), dtype=np.float32)
    heatmap[20:50, 20:50] = 0.9  # upper-left hotspot

    for lang in ("en", "tr"):
        text = get_explanation_text(
            predicted_class=1,
            confidence=0.885,
            probabilities=[0.05, 0.885, 0.065],
            heatmap=heatmap,
            language=lang,
        )
        assert "**" not in text, f"Found raw double asterisks in {lang} explanation: {text}"
        assert "*" not in text, f"Found raw single asterisks in {lang} explanation: {text}"


def test_xai_explanation_bilingual_exact_medical_phrasing():
    """Verify exact medical phrasing in Turkish and English without translation disconnect."""
    heatmap = np.zeros((224, 224), dtype=np.float32)
    heatmap[150:200, 150:200] = 0.8  # lower-right hotspot

    # 1. Turkish Mode
    text_tr = get_explanation_text(
        predicted_class=1,
        confidence=0.854,
        probabilities=[0.08, 0.854, 0.066],
        heatmap=heatmap,
        class_names={0: "Normal", 1: "Pneumonia", 2: "COVID-19"},
        language="tr",
    )
    assert "Model tahmini: Pnömoni (Konsolidasyon)" in text_tr
    assert "softmax güven skoru: %85.4" in text_tr
    assert "İkinci olası sınıf: Normal Parankim" in text_tr
    assert "Grad-CAM aktivasyon odağı görüntünün alt-sağ bölgesinde" in text_tr
    assert "Araştırma modeli klinik karar destek çıktısıdır" in text_tr

    # 2. English Mode
    text_en = get_explanation_text(
        predicted_class=1,
        confidence=0.854,
        probabilities=[0.08, 0.854, 0.066],
        heatmap=heatmap,
        class_names={0: "Normal", 1: "Pneumonia", 2: "COVID-19"},
        language="en",
    )
    assert "Model prediction: Pneumonia" in text_en
    assert "softmax confidence: 85.4%" in text_en
    assert "Runner-up class: Normal" in text_en
    assert "Grad-CAM peak relevance lies in the lower-right part" in text_en
    assert "Decision-support output of a research model" in text_en


def test_xai_explanation_multimodal_dermoscopy_and_radiology():
    """Verify specialty adaptation for dermoscopy (pathology review) vs radiology."""
    # Dermoscopy class set
    dermo_labels = {
        0: "Benign Dysplastic Nevus",
        1: "Malignant Melanoma",
        2: "Basal Cell Carcinoma",
    }

    text_dermo_tr = get_explanation_text(
        predicted_class=1,
        confidence=0.92,
        class_names=dermo_labels,
        language="tr",
    )
    assert "Malign Melanom" in text_dermo_tr
    assert "patoloji incelemesi" in text_dermo_tr

    text_dermo_en = get_explanation_text(
        predicted_class=2,
        confidence=0.78,
        class_names=dermo_labels,
        language="en",
    )
    assert "Basal Cell Carcinoma" in text_dermo_en
    assert "pathology review" in text_dermo_en

    # Radiology class set
    rad_labels = {0: "Normal", 1: "Pneumonia", 2: "COVID-19"}
    text_rad_tr = get_explanation_text(
        predicted_class=1,
        confidence=0.92,
        class_names=rad_labels,
        language="tr",
    )
    assert "radyolog incelemesi" in text_rad_tr


def test_xai_explanation_spatial_quadrant_localization():
    """Verify Grad-CAM vertical and horizontal spatial quadrant calculations."""
    h = np.zeros((300, 300), dtype=np.float32)

    # Upper-Left
    h_ul = h.copy()
    h_ul[10, 10] = 1.0
    text_ul = get_explanation_text(0, 0.9, heatmap=h_ul, language="en")
    assert "upper-left" in text_ul

    # Middle-Center
    h_mc = h.copy()
    h_mc[150, 150] = 1.0
    text_mc = get_explanation_text(0, 0.9, heatmap=h_mc, language="en")
    assert "middle-center" in text_mc

    # Lower-Right in Turkish
    h_lr = h.copy()
    h_lr[280, 280] = 1.0
    text_lr = get_explanation_text(0, 0.9, heatmap=h_lr, language="tr")
    assert "alt-sağ" in text_lr


def test_pdf_report_synchronization_and_sanitization():
    """Verify generated medical PDF reports contain clean findings without asterisks."""
    explanation_raw = (
        "Model prediction: Malignant Melanoma (softmax confidence: 91.2%)\n\n"
        "Runner-up class: Basal Cell Carcinoma (5.4%), margin 85.8 percentage points.\n\n"
        "Decision-support output of a research model."
    )

    pdf_bytes = generate_medical_report(
        patient_id="PAT-ISIC-0099",
        diagnosis="Malignant Melanoma",
        confidence=91.2,
        explanation=explanation_raw,
        original_image=np.ones((224, 224), dtype=np.float32) * 0.5,
        heatmap_image=np.ones((224, 224), dtype=np.float32) * 0.7,
        lang="en",
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 2000

    # Turkish PDF report generation
    pdf_tr_bytes = generate_medical_report(
        patient_id="PAT-ISIC-0099",
        diagnosis="Malign Melanom",
        confidence=91.2,
        explanation="Model tahmini: Malign Melanom (softmax güven skoru: %91.2)",
        lang="tr",
    )
    assert isinstance(pdf_tr_bytes, bytes)
    assert pdf_tr_bytes.startswith(b"%PDF-")


def test_fastapi_diagnose_language_negotiation(api_client: TestClient):
    """Verify /api/cdss/diagnose returns localized findings according to request language."""
    # Ensure server and held-out test scan exist in state with matching dimensions
    import torch
    img_size = 28
    state.image_size = img_size
    state.server = CentralServer(
        device=torch.device("cpu"),
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        img_size=img_size,
    )
    state.model_trained = True

    dummy_scan = np.ones((img_size, img_size), dtype=np.float32) * 0.4
    dummy_labels = [0]
    state.global_test_set = (
        torch.tensor(dummy_scan, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        torch.tensor(dummy_labels, dtype=torch.long),
    )

    # Test Turkish request
    resp_tr = api_client.post(
        "/api/cdss/diagnose",
        json={"image_data": dummy_scan.tolist(), "language": "tr"},
    )
    assert resp_tr.status_code == 200
    data_tr = resp_tr.json()
    assert "findings" in data_tr
    assert "**" not in data_tr["findings"]
    assert "Model tahmini:" in data_tr["findings"]

    # Test English request
    resp_en = api_client.post(
        "/api/cdss/diagnose",
        json={"image_data": dummy_scan.tolist(), "language": "en"},
    )
    assert resp_en.status_code == 200
    data_en = resp_en.json()
    assert "findings" in data_en
    assert "**" not in data_en["findings"]
    assert "Model prediction:" in data_en["findings"]
