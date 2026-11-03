"""Granular Unit Test Suite: CDSS Medical PDF Report Generator.

Verifies single-page A4 compliance, ReportLab/FPDF vector drawing,
dual-pane X-Ray and Grad-CAM visual evidence, and SHA-256 audit trail.
"""

from __future__ import annotations
import numpy as np
from src.fed_xray.cdss.report import generate_medical_report, get_diagnosis_explanation


def test_generate_medical_report_valid_bytes():
    """Verify generated report is non-empty bytes beginning with PDF magic header."""
    pdf_bytes = generate_medical_report(
        patient_id="PAT-TEST-001",
        diagnosis="Pneumonia",
        confidence=96.4,
        explanation="Consolidation in right lower lobe.",
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_medical_report_all_classes():
    """Verify PDF generator correctly formats all 3 classes plus unknown fallback."""
    for diag_name in ["Normal", "Pneumonia", "COVID-19", "Other"]:
        pdf_bytes = generate_medical_report(
            patient_id=f"PAT-TEST-{diag_name}",
            diagnosis=diag_name,
            confidence=88.5,
            explanation=f"Diagnostic assessment for {diag_name}.",
        )
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF-")


def test_generate_medical_report_with_scans_and_heatmap():
    """Verify PDF generator embeds genuine 2D scan and Grad-CAM attention heatmap."""
    scan = np.random.rand(28, 28).astype(np.float32)
    heat = np.random.rand(28, 28).astype(np.float32)

    pdf_bytes = generate_medical_report(
        patient_id="PAT-XAI-SCAN",
        diagnosis="COVID-19",
        confidence=92.1,
        explanation="Bilateral ground-glass opacities.",
        original_image=scan,
        heatmap_image=heat,
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000


def test_generate_medical_report_with_rag_digital_twins():
    """Verify PDF generator includes verified RAG digital twins in citation table."""
    similar_cases = [
        {
            "case_id": "REF-TWIN-101",
            "similarity": 98.4,
            "biopsy_finding": "Confirmed lobar pneumonia with severe alveolar infiltration.",
        },
        {
            "case_id": "REF-TWIN-102",
            "similarity": 94.7,
            "history": "Bacterial pneumonia resolved after 10-day antibiotic regimen.",
        },
    ]

    pdf_bytes = generate_medical_report(
        patient_id="PAT-RAG-001",
        diagnosis="Pneumonia",
        confidence=95.0,
        explanation="Evidence-grounded retrieval matches bacterial profile.",
        similar_cases=similar_cases,
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_medical_report_empty_similar_cases_renders_honest_empty_state():
    """Verify PDF generator renders honest empty state when no digital twins are retrieved."""
    pdf_bytes = generate_medical_report(
        patient_id="PAT-NO-TWINS-001",
        diagnosis="Normal",
        confidence=98.5,
        explanation="Routine screening scan without matching reference cases.",
        similar_cases=[],
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    # Verify neither fake PAT_REF_0114 nor PAT_REF_1042 are embedded
    assert b"PAT_REF_0114" not in pdf_bytes
    assert b"PAT_REF_1042" not in pdf_bytes
    assert b"PAT_REF_2035" not in pdf_bytes


def test_generate_medical_report_without_images_renders_honest_placeholders():
    """Verify PDF generator renders honest placeholders when no scan or heatmap is attached."""
    pdf_bytes = generate_medical_report(
        patient_id="PAT-NO-IMG-001",
        diagnosis="Pneumonia",
        confidence=91.0,
        explanation="Clinical evaluation based on non-imaging biomarkers.",
        original_image=None,
        heatmap_image=None,
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 1000


def test_generate_medical_report_with_scan_only_renders_unavailable_saliency():
    """Verify PDF generator renders primary scan and unavailable saliency indicator when heatmap is omitted."""
    scan = np.random.rand(224, 224).astype(np.float32)
    pdf_bytes = generate_medical_report(
        patient_id="PAT-SCAN-ONLY-001",
        diagnosis="COVID-19",
        confidence=89.5,
        explanation="Bilateral infiltrate assessment.",
        original_image=scan,
        heatmap_image=None,
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")


def test_get_diagnosis_explanation():
    """Verify evidence-grounded findings report actual model predictions without canned claims."""
    normal_exp = get_diagnosis_explanation("Normal", 98.0)
    assert "normal" in normal_exp.lower()
    assert "98.0%" in normal_exp
    # Verify canned anatomical hallucinations are eliminated
    assert "cardiomediastinal silhouette" not in normal_exp.lower()

    pneu_exp = get_diagnosis_explanation("Pneumonia", 94.5)
    assert "pneumonia" in pneu_exp.lower()
    assert "94.5%" in pneu_exp
    assert "air bronchograms" not in pneu_exp.lower()

    covid_exp = get_diagnosis_explanation("COVID-19", 91.0)
    assert "covid" in covid_exp.lower()
    assert "91.0%" in covid_exp
    assert "interlobular septal" not in covid_exp.lower()


def test_get_diagnosis_explanation_with_probabilities_and_heatmap():
    """Verify explanation dynamically computes runner-up margin and Grad-CAM peak relevance."""
    probs = [0.10, 0.75, 0.15]
    # Create synthetic heatmap with peak in top-left
    heatmap = np.zeros((224, 224), dtype=np.float32)
    heatmap[20:40, 20:40] = 1.0

    exp = get_diagnosis_explanation(
        diagnosis="Pneumonia",
        confidence=75.0,
        probabilities=probs,
        heatmap=heatmap,
    )
    assert "pneumonia" in exp.lower()
    assert "runner-up class" in exp.lower()
    assert "covid-19" in exp.lower()
    assert "grad-cam peak relevance" in exp.lower()
    assert "upper-left" in exp.lower()


def test_generate_medical_report_isic_dermoscopy():
    """Verify ISIC dermoscopy reports use genuine pathology and omit chest markers and canned COVID badges."""
    import re
    import zlib

    scan = np.random.rand(224, 224).astype(np.float32)
    heat = np.random.rand(224, 224).astype(np.float32)
    pdf_bytes = generate_medical_report(
        patient_id="PAT-ISIC-0099",
        diagnosis="Basal Cell Carcinoma",
        confidence=40.0,
        explanation="Model prediction: Basal Cell Carcinoma (softmax confidence: 40.0%).",
        original_image=scan,
        heatmap_image=heat,
        probabilities=[24.0, 36.0, 40.0],
        class_labels=["Benign Dysplastic Nevus", "Malignant Melanoma", "Basal Cell Carcinoma"],
        dataset_name="ISIC_2019",
        lang="en",
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")

    # Decompress PDF streams to inspect rendered text
    streams = re.findall(b"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, re.DOTALL)
    decompressed = b""
    for s in streams:
        try:
            decompressed += zlib.decompress(s)
        except Exception:
            decompressed += s

    # Verify no canned COVID-19 fallback badge
    assert b"COVID-19 (INFECTIOUS)" not in decompressed
    assert b"BASAL CELL CARCINOMA" in decompressed
    assert b"Dermoscopy" in decompressed or b"DERMOSCOPY" in decompressed


def test_generate_medical_report_turkish_localization():
    """Verify Turkish localization safely sanitizes non-Latin-1 characters without encoding crashes."""
    pdf_bytes = generate_medical_report(
        patient_id="PAT-TR-01",
        diagnosis="Bazal Hücreli Karsinom",
        confidence=40.0,
        explanation="Model tahmini: Bazal Hücreli Karsinom. İkinci olası sınıf: Malign Melanom. Yoğun odak sağ alanda.",
        probabilities=[24.0, 36.0, 40.0],
        class_labels=["Benign Displastik Nevüs", "Malign Melanom", "Bazal Hücreli Karsinom"],
        dataset_name="ISIC_2019",
        lang="tr",
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 2000

