"""Granular Unit Test Suite: CDSS Explainable AI (Grad-CAM).

Verifies forward/backward activation hook registration,
gradient backpropagation, saliency heatmap normalization, and shape fidelity.
"""

from __future__ import annotations
import numpy as np
import torch
from src.fed_xray.cdss.xai import GradCAM
from src.fed_xray.models.cnn import XRayClassifier


def test_gradcam_hook_lifecycle():
    """Verify forward and backward hooks register and clean up properly."""
    model = XRayClassifier(num_classes=3)
    cam = GradCAM(model)

    assert len(cam.hook_handles) == 2
    cam.remove_hooks()
    assert len(cam.hook_handles) == 0


def test_gradcam_heatmap_generation_cnn():
    """Verify Grad-CAM produces normalized 2D saliency heatmap for CNN."""
    model = XRayClassifier(num_classes=3)
    cam = GradCAM(model)

    # 1 channel 28x28 chest X-Ray tensor
    input_tensor = torch.rand(1, 1, 28, 28, requires_grad=True)

    heatmap, pred_class, confidence = cam.generate_heatmap(input_tensor)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (28, 28)
    assert 0 <= pred_class < 3
    assert 0.0 <= confidence <= 1.0

    # Normalization bounds: values must strictly lie in [0.0, 1.0]
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    cam.remove_hooks()


def test_gradcam_specific_target_class():
    """Verify Grad-CAM backpropagates with respect to an explicit target class."""
    model = XRayClassifier(num_classes=3)
    cam = GradCAM(model)

    input_tensor = torch.rand(1, 1, 28, 28)

    heatmap_c0, pred_cls, conf = cam.generate_heatmap(input_tensor, target_class=0)
    heatmap_c2, _, _ = cam.generate_heatmap(input_tensor, target_class=2)

    assert heatmap_c0.shape == (28, 28)
    assert heatmap_c2.shape == (28, 28)
    # Different target classes should produce different gradient attention distributions
    assert not np.allclose(heatmap_c0, heatmap_c2, atol=1e-5)
    cam.remove_hooks()


def test_gradcam_heatmap_generation_vision_transformer():
    """Verify Grad-CAM hooks onto last encoder block of VisionTransformer and produces organic 2D heatmap."""
    from src.fed_xray.models.vit import VisionTransformer

    model = VisionTransformer(img_size=224, patch_size=16, in_channels=1, num_classes=3, depth=4, num_heads=4, embed_dim=128)
    cam = GradCAM(model)

    assert "blocks." in cam.target_layer_name
    assert not cam.target_layer_name.startswith("patch_embed")

    input_tensor = torch.rand(1, 1, 224, 224, requires_grad=True)
    heatmap, pred_class, confidence = cam.generate_heatmap(input_tensor)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert 0 <= pred_class < 3
    assert 0.0 <= confidence <= 1.0
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0

    # Ensure heatmap has rich spatial variation rather than a single degenerate patch
    nonzero_ratio = (heatmap > 0.05).mean()
    assert nonzero_ratio > 0.15, f"Heatmap attention is overly collapsed: {nonzero_ratio:.4f}"

    cam.remove_hooks()
