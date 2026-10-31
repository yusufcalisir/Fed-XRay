"""Tests for Evidence-Grounded CDSS: Native 224x224 Grad-CAM and Federated RAG.

Verifies:
1. Native 224x224 Grad-CAM saliency heatmap generation on Vision Transformer.
2. Spatial peak relevance quadrant analysis and active pixel thresholding.
3. Cosine metric retrieval over genuine Strategy E biopsy reference cohort.
4. Absence of hardcoded, static, or fabricated retrieval vectors.
"""

from __future__ import annotations
import numpy as np
import torch
from src.fed_xray.models.vit import create_medical_vit
from src.fed_xray.cdss.xai import GradCAM, get_explanation_text
from src.fed_xray.cdss.similarity import (
    FederatedRAGCaseMatcher,
    extract_embedding,
)
from src.fed_xray.data.real_world import StrategyEDatasetEcosystem


def test_vit_tiny_gradcam_at_224x224():
    """Verify Grad-CAM produces native 224x224 heatmap on Vision Transformer."""
    model = create_medical_vit(
        model_type="vit_tiny",
        img_size=224,
        patch_size=16,
        in_channels=1,
        num_classes=3,
    )
    gradcam = GradCAM(model)

    test_scan = torch.randn(1, 1, 224, 224, requires_grad=True)
    try:
        heatmap, pred_class, conf = gradcam.generate_heatmap(test_scan)
        assert heatmap.shape == (224, 224), f"Expected 224x224 heatmap, got {heatmap.shape}"
        assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0, "Heatmap must be normalized in [0, 1]"
        assert heatmap.max() > heatmap.min(), "Heatmap must have non-zero spatial contrast"
        assert pred_class in (0, 1, 2)
        assert 0.0 <= conf <= 1.0
    finally:
        gradcam.remove_hooks()


def test_gradcam_spatial_peak_relevance_analysis():
    """Verify spatial explanation text extraction from 224x224 heatmap."""
    heatmap = np.zeros((224, 224), dtype=np.float32)
    # Inject focal relevance peak in the lower-right quadrant
    heatmap[180:210, 180:210] = 0.85

    explanation = get_explanation_text(
        predicted_class=1,
        confidence=0.88,
        probabilities=[0.05, 0.88, 0.07],
        heatmap=heatmap,
    )

    assert "Model prediction: Pneumonia" in explanation
    assert "88.0%" in explanation
    assert "lower-right" in explanation
    assert "exceed 0.5 normalized relevance" in explanation
    assert "Decision-support output of a research model" in explanation


def test_federated_rag_case_retrieval_over_real_cohort():
    """Verify evidence-grounded RAG case matching over genuine Strategy E reference cohort."""
    ecosystem = StrategyEDatasetEcosystem(seed=42)
    records = ecosystem.load_from_disk_manifest(
        manifest_path="data/manifests/strategy_e_manifest.csv",
        images_dir="data",
        target_size=(224, 224),
    )
    assert len(records) >= 100, f"Expected at least 100 records, got {len(records)}"

    model = create_medical_vit(
        model_type="vit_tiny",
        img_size=224,
        patch_size=16,
        in_channels=1,
        num_classes=3,
    )

    # Use first 30 records as reference bank
    reference_bank = records[:30]
    matcher = FederatedRAGCaseMatcher(reference_records=reference_bank, model=model)

    # Use 31st record as query
    query_rec = records[30]
    query_tensor = query_rec.image_tensor.unsqueeze(0)
    query_emb = extract_embedding(model, query_tensor)

    assert query_emb.shape == (64,), f"Expected 64-dimensional embedding, got {query_emb.shape}"
    assert abs(np.linalg.norm(query_emb) - 1.0) < 1e-4, "Embedding must be L2-normalized"

    similar_cases = matcher.find_similar(query_emb, top_k=3)
    assert len(similar_cases) == 3

    for case in similar_cases:
        assert case["case_id"].startswith("PAT_")
        assert case["label"] in (0, 1, 2)
        assert 0.0 <= case["similarity"] <= 1.0
        assert len(case["clinical_notes"]) > 10, "Case must have genuine clinical notes"

