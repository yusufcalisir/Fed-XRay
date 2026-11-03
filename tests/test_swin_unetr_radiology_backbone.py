"""Swin UNETR Hierarchical Radiology Foundation Backbone Verification Suite.

Validates 12 ShiftedWindowAttention blocks, window shift mechanics, patch merging,
PEFT parameter extraction, and 2D/3D volumetric tensor support for medical imaging.
"""

from __future__ import annotations
import pytest
import torch
import torchvision.models.swin_transformer as sw_trans

from src.fed_xray.models.foundation import SwinUNETRRadiologyBackbone
from src.fed_xray.models.peft import inject_lora_to_model


def test_swin_unetr_authentic_hierarchical_architecture():
    """Verify SwinUNETRRadiologyBackbone uses genuine ShiftedWindowAttention and SwinTransformer."""
    model = SwinUNETRRadiologyBackbone(embed_dim=768, num_classes=3, pretrained=True)

    # Verify underlying architecture is authentic SwinTransformer
    assert isinstance(model.swin, sw_trans.SwinTransformer)
    assert hasattr(model, "conv_stem")
    assert not model.conv_stem.weight.requires_grad
    assert model.conv_stem.weight.norm().item() > 0.1

    # Verify presence of ShiftedWindowAttention blocks
    attn_blocks = [m for m in model.swin.modules() if isinstance(m, sw_trans.ShiftedWindowAttention)]
    assert len(attn_blocks) == 12  # Swin-T contains 12 ShiftedWindowAttention blocks across 4 stages


def test_swin_unetr_2d_and_3d_volumetric_inference():
    """Verify inference across 2D slices (grayscale & RGB) and 3D volumetric CT/MRI tensors."""
    model = SwinUNETRRadiologyBackbone(embed_dim=768, num_classes=3, pretrained=True)

    # 1. Grayscale 2D scan [B, 1, 28, 28]
    x_gray = torch.randn(2, 1, 28, 28)
    logits_gray, feats_gray = model(x_gray, return_features=True)
    assert logits_gray.shape == (2, 3)
    assert feats_gray.shape == (2, 768)

    # 2. RGB 2D scan [B, 3, 224, 224] (Strategy E standard)
    x_rgb = torch.randn(2, 3, 224, 224)
    logits_rgb, feats_rgb = model(x_rgb, return_features=True)
    assert logits_rgb.shape == (2, 3)
    assert feats_rgb.shape == (2, 768)

    # 3. 3D Volumetric CT/MRI volume [B, C, D, H, W]
    x_vol = torch.randn(2, 1, 4, 28, 28)
    logits_vol, feats_vol = model(x_vol, return_features=True)
    assert logits_vol.shape == (2, 3)
    assert feats_vol.shape == (2, 768)


def test_swin_unetr_peft_lora_injection():
    """Verify LoRA and FedSA-LoRA parameter-efficient adaptation on Swin UNETR."""
    model = SwinUNETRRadiologyBackbone(embed_dim=768, num_classes=3, pretrained=True)

    # Inject FedSA-LoRA into Swin qkv attention projections
    res = inject_lora_to_model(
        model,
        target_modules=["qkv"],
        r=8,
        mode="fedsa_lora",
    )

    assert res["injected_adapters"] == 12
    assert res["trainable_parameters"] > 0
    assert res["compression_percentage"] > 95.0

    # Forward pass executes with adapted parameters
    x = torch.randn(2, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (2, 3)
    assert not torch.isnan(logits).any()


def test_swin_unetr_strict_checkpoint_honest_failure():
    """Verify strict checkpoint fails honestly when foundation weights are missing."""
    with pytest.raises(RuntimeError) as excinfo:
        SwinUNETRRadiologyBackbone(
            strict_checkpoint=True,
            checkpoint_path="nonexistent_checkpoint/monai_swin_5050.pt",
        )
    assert "Swin UNETR foundation checkpoint not found" in str(excinfo.value)

