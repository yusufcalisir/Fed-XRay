"""Foundation Model Checkpoint Integrity and Weight Provenance Verification Suite.

Verifies zero-silent-pass policy, honest exception raising on missing or corrupted weights,
SHA-256 weight fingerprinting, and runtime provenance telemetry exposure.
"""

from __future__ import annotations

import pytest
import torch

from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer
from src.fed_xray.models.foundation import SwinUNETRRadiologyBackbone, UNIFoundationBackbone
from src.fed_xray.models.vit import create_medical_vit, load_pretrained_vit_weights


def test_uni_explicit_checkpoint_missing_raises_error(tmp_path):
    """Passing a non-existent explicit checkpoint path must fail honestly."""
    fake_path = tmp_path / "nonexistent_uni_checkpoint.pt"
    with pytest.raises((RuntimeError, FileNotFoundError)) as exc_info:
        UNIFoundationBackbone(
            embed_dim=512,
            num_classes=3,
            pretrained=True,
            checkpoint_path=str(fake_path),
            strict_checkpoint=False,
        )
    assert "UNI foundation checkpoint not found" in str(exc_info.value)


def test_uni_strict_checkpoint_failure():
    """In strict mode, if no valid foundation weights are found, it must raise RuntimeError."""
    with pytest.raises(RuntimeError) as exc_info:
        UNIFoundationBackbone(
            embed_dim=512,
            num_classes=3,
            pretrained=True,
            checkpoint_path=None,
            strict_checkpoint=True,
        )
    assert "Strict checkpoint loading failed" in str(exc_info.value) or "UNI foundation" in str(exc_info.value)


def test_uni_provenance_scratch_fallback():
    """When strict_checkpoint=False and no weights exist, model runs with scratch weights and reports provenance."""
    model = UNIFoundationBackbone(
        embed_dim=512,
        num_classes=3,
        pretrained=False,
        strict_checkpoint=False,
    )
    assert model.is_pretrained is False
    assert model.checkpoint_provenance == "uninitialized_scratch_weights"
    assert model.checkpoint_hash is None


def test_uni_valid_checkpoint_loading(tmp_path):
    """A valid checkpoint must load properly, set is_pretrained=True, and compute a hash."""
    ck_path = tmp_path / "valid_uni.pt"
    m_temp = UNIFoundationBackbone(embed_dim=512, num_classes=3, pretrained=False)
    torch.save(m_temp.state_dict(), ck_path)

    loaded = UNIFoundationBackbone(
        embed_dim=512,
        num_classes=3,
        pretrained=True,
        checkpoint_path=str(ck_path),
        strict_checkpoint=True,
    )
    assert loaded.is_pretrained is True
    assert "local_checkpoint" in loaded.checkpoint_provenance
    assert loaded.checkpoint_hash is not None
    assert len(loaded.checkpoint_hash) == 16


def test_swin_unetr_explicit_checkpoint_missing_raises_error(tmp_path):
    """Non-existent Swin UNETR checkpoint must fail honestly."""
    fake_path = tmp_path / "nonexistent_swin_checkpoint.pt"
    with pytest.raises((RuntimeError, FileNotFoundError)) as exc_info:
        SwinUNETRRadiologyBackbone(
            embed_dim=384,
            num_classes=3,
            pretrained=True,
            checkpoint_path=str(fake_path),
            strict_checkpoint=False,
        )
    assert "Swin UNETR foundation checkpoint not found" in str(exc_info.value)


def test_swin_unetr_strict_checkpoint_failure():
    """In strict mode, Swin UNETR must raise RuntimeError if weights cannot be authenticated."""
    with pytest.raises(RuntimeError) as exc_info:
        SwinUNETRRadiologyBackbone(
            embed_dim=384,
            num_classes=3,
            pretrained=True,
            checkpoint_path=None,
            strict_checkpoint=True,
        )
    assert "Strict checkpoint loading failed" in str(exc_info.value) or "Swin UNETR" in str(exc_info.value)


def test_swin_unetr_valid_checkpoint_loading(tmp_path):
    """Valid Swin UNETR checkpoint must set is_pretrained=True, provenance, and hash."""
    ck_path = tmp_path / "valid_swin.pt"
    m_temp = SwinUNETRRadiologyBackbone(embed_dim=384, num_classes=3, pretrained=False)
    torch.save(m_temp.state_dict(), ck_path)

    loaded = SwinUNETRRadiologyBackbone(
        embed_dim=384,
        num_classes=3,
        pretrained=True,
        checkpoint_path=str(ck_path),
        strict_checkpoint=True,
    )
    assert loaded.is_pretrained is True
    assert "local_checkpoint" in loaded.checkpoint_provenance
    assert loaded.checkpoint_hash is not None
    assert len(loaded.checkpoint_hash) == 16


def test_vit_explicit_checkpoint_missing_strict():
    """create_medical_vit with non-existent checkpoint and strict_checkpoint=True must raise RuntimeError."""
    with pytest.raises(RuntimeError) as exc_info:
        create_medical_vit(
            model_type="vit_tiny",
            pretrained=True,
            checkpoint_path="nonexistent_vit_weights.pt",
            strict_checkpoint=True,
        )
    assert "VisionTransformer checkpoint not found" in str(exc_info.value) or "Failed to initialize" in str(exc_info.value)


def test_vit_valid_checkpoint_loading(tmp_path):
    """A valid ViT checkpoint updates provenance and hash."""
    m_temp = create_medical_vit(model_type="vit_tiny", pretrained=False)
    ck_path = tmp_path / "valid_vit.pt"
    torch.save(m_temp.state_dict(), ck_path)

    m = create_medical_vit(
        model_type="vit_tiny",
        pretrained=False,
    )
    success = load_pretrained_vit_weights(m, checkpoint_path=str(ck_path), strict=True)
    assert success is True
    assert m.is_pretrained is True
    assert "local_checkpoint" in m.checkpoint_provenance
    assert m.checkpoint_hash is not None
    assert len(m.checkpoint_hash) == 16


def test_central_server_and_client_provenance_telemetry():
    """CentralServer and HospitalClient must expose authentic get_model_provenance metadata."""
    server = CentralServer(
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        pretrained=False,
        strict_checkpoint=False,
    )
    srv_prov = server.get_model_provenance()
    assert srv_prov["model_type"] == "vit_tiny"
    assert srv_prov["is_pretrained"] is False
    assert srv_prov["checkpoint_provenance"] == "uninitialized_scratch_weights"
    assert srv_prov["strict_checkpoint"] is False
    assert srv_prov["peft_mode"] == "ffa_lora"

    client = HospitalClient(
        client_id=0,
        dataloader=None,
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        pretrained=False,
        strict_checkpoint=False,
    )
    clt_prov = client.get_model_provenance()
    assert clt_prov["client_id"] == 0
    assert clt_prov["model_type"] == "vit_tiny"
    assert clt_prov["is_pretrained"] is False
    assert clt_prov["checkpoint_provenance"] == "uninitialized_scratch_weights"
    assert clt_prov["strict_checkpoint"] is False
    assert clt_prov["peft_mode"] == "ffa_lora"


def test_foundation_provenance_method_contracts():
    """Verify UNIFoundationBackbone and SwinUNETRRadiologyBackbone provide get_model_provenance telemetry."""
    uni = UNIFoundationBackbone(embed_dim=512, num_classes=3, pretrained=False)
    uni_prov = uni.get_model_provenance()
    assert uni_prov["model_type"] == "uni"
    assert uni_prov["is_pretrained"] is False
    assert uni_prov["checkpoint_provenance"] == "uninitialized_scratch_weights"
    assert uni_prov["strict_checkpoint"] is False
    assert uni_prov["is_fallback_distilled"] is False
    assert "last_load_error" in uni_prov

    swin = SwinUNETRRadiologyBackbone(embed_dim=384, num_classes=3, pretrained=False)
    swin_prov = swin.get_model_provenance()
    assert swin_prov["model_type"] == "swin_unetr"
    assert swin_prov["is_pretrained"] is False
    assert swin_prov["checkpoint_provenance"] == "uninitialized_scratch_weights"
    assert swin_prov["strict_checkpoint"] is False
    assert swin_prov["is_fallback_distilled"] is False
    assert "last_load_error" in swin_prov


def test_ensure_foundation_checkpoints_staging(tmp_path):
    """Verify ensure_foundation_checkpoints stages valid weights and computes digests."""
    from src.fed_xray.models.foundation import ensure_foundation_checkpoints

    staging_dir = tmp_path / "stage_ck"
    result = ensure_foundation_checkpoints(target_dir=staging_dir)

    assert result["integrity_verified"] is True
    assert (staging_dir / "uni_foundation_weights.pt").exists()
    assert (staging_dir / "swin_unetr_weights.pt").exists()
    assert len(result["uni"]["sha256"]) == 64
    assert len(result["swin_unetr"]["sha256"]) == 64
    assert result["uni"]["size_bytes"] > 0
    assert result["swin_unetr"]["size_bytes"] > 0

    # Ensure a backbone can load directly from the staged file
    uni_loaded = UNIFoundationBackbone(
        embed_dim=512,
        num_classes=3,
        pretrained=True,
        checkpoint_path=result["uni"]["path"],
        strict_checkpoint=True,
    )
    assert uni_loaded.is_pretrained is True
    assert "local_checkpoint" in uni_loaded.checkpoint_provenance

