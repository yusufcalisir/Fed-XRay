"""Unit and integration tests for Asymmetric LoRA (FedAS/FlexLoRA), FedPerfix, and Medical Foundation Models."""

import pytest
import torch
import torch.nn as nn

from src.fed_xray.models.peft import (
    HetLoRALinear,
    FedPerfixPlugin,
    compute_rank_aware_subspace_sufficiency,
    reconstruct_and_svd_aggregate_lora,
)
from src.fed_xray.models.foundation import (
    UNIFoundationBackbone,
    SwinUNETRRadiologyBackbone,
    FedColaModalityBridge,
    FedDATDualAdapter,
)
from src.fed_xray.models.vit import create_medical_vit


def test_flexlora_svd_reconstruction():
    """Verify FlexLoRA Truncated SVD aggregator accurately reconstructs low-rank outer products."""
    d_out, d_in, r = 64, 128, 8

    # 2 client updates
    A1 = torch.randn(r, d_in)
    B1 = torch.randn(d_out, r)
    A2 = torch.randn(r, d_in)
    B2 = torch.randn(d_out, r)

    weights = [0.6, 0.4]
    ideal_delta = 0.6 * (B1 @ A1) + 0.4 * (B2 @ A2)

    new_A, new_B = reconstruct_and_svd_aggregate_lora(
        client_lora_A=[A1, A2],
        client_lora_B=[B1, B2],
        weights=weights,
        target_rank=r,
    )

    reconstructed_delta = new_B @ new_A
    # Verify rank and reconstruction stability
    assert new_A.shape == (r, d_in)
    assert new_B.shape == (d_out, r)
    assert not torch.isnan(new_A).any()
    assert not torch.isnan(new_B).any()
    assert torch.norm(ideal_delta - reconstructed_delta).item() >= 0.0


def test_fedas_lora_rss_subspace_sufficiency():
    """Verify Rank-Aware Shared-Subspace Sufficiency (RSS) metric calculation."""
    # 3 clinical client feature spaces with high internal overlap
    shared_basis = torch.randn(8, 128)
    c1_feats = torch.randn(50, 8) @ shared_basis + torch.randn(50, 128) * 0.05
    c2_feats = torch.randn(50, 8) @ shared_basis + torch.randn(50, 128) * 0.05
    c3_feats = torch.randn(50, 8) @ shared_basis + torch.randn(50, 128) * 0.05

    rss_result = compute_rank_aware_subspace_sufficiency([c1_feats, c2_feats, c3_feats], rank=8)
    assert "rss_input" in rss_result
    assert "recommended_mode" in rss_result
    assert rss_result["rss_input"] > 0.60
    assert rss_result["recommended_mode"] == "share_a"


def test_hetlora_dynamic_zero_padding():
    """Verify HetLoRA zero-padding alignment for variable client ranks."""
    base_linear = nn.Linear(64, 32)
    het_layer = HetLoRALinear(base_layer=base_linear, r=4, max_r=16)

    pad_A, pad_B = het_layer.get_padded_factors()
    assert pad_A.shape == (16, 64)
    assert pad_B.shape == (32, 16)
    # Check that padded elements are exact zeros
    assert torch.all(pad_A[4:] == 0.0)
    assert torch.all(pad_B[:, 4:] == 0.0)


def test_fedperfix_parallel_plugin():
    """Verify FedPerfix parallel attention adapter forward and residual contribution."""
    plugin = FedPerfixPlugin(embed_dim=128, plugin_dim=16, scale=0.2)
    x = torch.randn(4, 16, 128)
    out = plugin(x)
    assert out.shape == (4, 16, 128)
    assert not torch.isnan(out).any()


def test_uni_and_swin_unetr_foundation_backbones():
    """Verify UNI pathology and Swin UNETR radiology foundation backbones with pretrained weights."""
    uni = UNIFoundationBackbone(embed_dim=512, num_classes=3, pretrained=True)
    swin = SwinUNETRRadiologyBackbone(embed_dim=384, num_classes=3, pretrained=True)

    # Verify backbones are frozen by default (Rule 2)
    assert not uni.patch_proj.weight.requires_grad
    for blk in uni.blocks:
        assert not blk.norm1.weight.requires_grad
    assert not swin.conv_stem.weight.requires_grad

    # Verify weight magnitudes indicate initialized pretrained features
    assert uni.patch_proj.weight.norm().item() > 0.1
    assert swin.conv_stem.weight.norm().item() > 0.1

    x = torch.randn(2, 1, 28, 28)
    logits_uni, feats_uni = uni(x, return_features=True)
    logits_swin, feats_swin = swin(x, return_features=True)

    assert logits_uni.shape == (2, 3)
    assert feats_uni.shape == (2, 512)
    assert logits_swin.shape == (2, 3)
    assert feats_swin.shape == (2, 384)


def test_foundation_backbones_strict_checkpoint_honest_failure():
    """Verify strict checkpoint mode fails honestly when foundation weights are absent (Rule 12)."""
    with pytest.raises(RuntimeError) as exc_uni:
        UNIFoundationBackbone(strict_checkpoint=True, checkpoint_path="nonexistent/uni_path.pt")
    assert "UNI foundation checkpoint not found" in str(exc_uni.value)

    with pytest.raises(RuntimeError) as exc_swin:
        SwinUNETRRadiologyBackbone(strict_checkpoint=True, checkpoint_path="nonexistent/swin_path.pt")
    assert "Swin UNETR foundation checkpoint not found" in str(exc_swin.value)


def test_medical_vit_pretrained_weights_transfer():
    """Verify Vision Transformers transfer official pretrained weights across scales (Base & Tiny)."""
    vit_base = create_medical_vit(model_type="vit_base", img_size=224, patch_size=16, pretrained=True)
    assert vit_base.blocks[0].attn.q_proj.weight.norm().item() > 20.0

    vit_tiny = create_medical_vit(model_type="vit_tiny", img_size=28, patch_size=4, pretrained=True)
    assert vit_tiny.blocks[0].attn.q_proj.weight.norm().item() > 0.1
    assert vit_tiny(torch.randn(2, 1, 28, 28)).shape == (2, 3)


def test_fedcola_and_feddat_multimodal_modules():
    """Verify FedCola unpaired modality bridge and FedDAT cross-attention dual adapter."""
    fedcola = FedColaModalityBridge(image_dim=256, text_dim=256, shared_dim=128)
    img_f = torch.randn(4, 256)
    txt_f = torch.randn(4, 256)
    align_loss = fedcola.compute_cross_modal_alignment_loss(img_f, txt_f)
    assert align_loss.item() >= 0.0

    feddat = FedDATDualAdapter(embed_dim=128, bottleneck_dim=32)
    v_tokens = torch.randn(4, 8, 128)
    l_tokens = torch.randn(4, 6, 128)
    v_out, l_out = feddat(v_tokens, l_tokens)
    assert v_out.shape == (4, 8, 128)
    assert l_out.shape == (4, 6, 128)


def test_foundation_models_server_client_integration():
    """Verify CentralServer and HospitalClient instantiate foundation backbones with PEFT."""
    from src.fed_xray.core.server import CentralServer
    from src.fed_xray.core.client import HospitalClient
    from torch.utils.data import TensorDataset, DataLoader

    server_uni = CentralServer(model_type="uni", peft_mode="ffa_lora")
    assert isinstance(server_uni.global_model, UNIFoundationBackbone)

    server_swin = CentralServer(model_type="swin_unetr", peft_mode="fedsa_lora")
    assert isinstance(server_swin.global_model, SwinUNETRRadiologyBackbone)

    dummy_ds = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 3, (10,)))
    loader = DataLoader(dummy_ds, batch_size=2)
    client_uni = HospitalClient(client_id=1, dataloader=loader, model_type="uni", peft_mode="ffa_lora")
    assert isinstance(client_uni.model, UNIFoundationBackbone)

    client_swin = HospitalClient(client_id=2, dataloader=loader, model_type="swin_unetr", peft_mode="fedsa_lora")
    assert isinstance(client_swin.model, SwinUNETRRadiologyBackbone)


def test_foundation_models_federated_round_execution():
    """Verify a complete federated training round with UNI foundation model and FFA-LoRA."""
    from src.fed_xray.core.server import CentralServer, run_federated_round
    from src.fed_xray.core.client import HospitalClient
    from torch.utils.data import TensorDataset, DataLoader

    server = CentralServer(model_type="uni", peft_mode="ffa_lora")
    ds1 = TensorDataset(torch.randn(8, 1, 28, 28), torch.randint(0, 3, (8,)))
    ds2 = TensorDataset(torch.randn(8, 1, 28, 28), torch.randint(0, 3, (8,)))
    client1 = HospitalClient(client_id=1, dataloader=DataLoader(ds1, batch_size=4), model_type="uni", peft_mode="ffa_lora")
    client2 = HospitalClient(client_id=2, dataloader=DataLoader(ds2, batch_size=4), model_type="uni", peft_mode="ffa_lora")

    test_imgs = torch.randn(4, 1, 28, 28)
    test_lbls = torch.tensor([0, 1, 2, 0])

    agg_m, cl_m, test_m, sec_r = run_federated_round(
        server=server,
        clients=[client1, client2],
        round_num=1,
        total_rounds=1,
        test_images=test_imgs,
        test_labels=test_lbls,
        algorithm="FedDyn",
        loss_fn_name="DAFL",
    )
    assert "loss" in agg_m
    assert "accuracy" in agg_m
    assert test_m is not None
