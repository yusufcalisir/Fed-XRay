"""Test MOD-03: Advanced Multimodal Foundation and PEFT Adapter Integration.

Verifies that MedSAMClientAdapter, FedMedCLIPModule, FedColaModalityBridge,
and FedDATDualAdapter are actively integrated into CentralServer, HospitalClient,
peft.py, and federated training routines with zero dormant or disconnected code.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round
from src.fed_xray.models.foundation import (
    FedColaModalityBridge,
    FedDATDualAdapter,
    FedMedCLIPModel,
    FedMedCLIPModule,
    MedSAMClientAdapter,
)
from src.fed_xray.models.peft import (
    FedDATAdapterLinear,
    MedSAMAdapterLinear,
    extract_peft_state_dict,
    inject_lora_to_model,
    load_peft_state_dict,
)
from src.fed_xray.models.vit import create_medical_vit


def test_fedmedclip_model_end_to_end_lifecycle():
    """Verify FedMedCLIPModel couples vision backbone with multimodal alignment."""
    model = FedMedCLIPModel(
        backbone_type="vit_tiny",
        img_size=28,
        patch_size=4,
        num_classes=3,
        pretrained=False,
    )
    assert isinstance(model.medclip, FedMedCLIPModule)
    assert model.embed_dim == 128
    assert model.is_pretrained is False
    assert "fedmedclip:" in model.checkpoint_provenance

    # Forward pass: log-probabilities
    x = torch.randn(4, 1, 28, 28)
    log_probs = model(x)
    assert log_probs.shape == (4, 3)

    # Multimodal forward pass
    y_ens, p_fam, p_mlp = model.forward_multimodal(x)
    assert y_ens.shape == (4, 3)
    assert p_fam.shape == (4, 3)
    assert p_mlp.shape == (4, 3)

    # Extract features for RAG / prototype matching
    feats = model.extract_features(x)
    assert feats.shape == (4, 128)

    # Compute multimodal mutual distillation loss
    targets = torch.tensor([0, 1, 2, 0])
    loss, y_pred = model.compute_local_loss(x, targets, beta=0.5)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    assert y_pred.shape == (4, 3)


def test_medsam_client_adapter_injection_and_peft_extraction():
    """Verify MedSAMClientAdapter wraps linear layers and participates in PEFT aggregation."""
    base_vit = create_medical_vit(
        model_type="vit_tiny",
        img_size=28,
        patch_size=4,
        num_classes=3,
        pretrained=False,
    )
    report = inject_lora_to_model(base_vit, mode="medsam", r=16)
    assert report["injected_adapters"] > 0
    assert report["mode"] == "medsam"

    # Confirm MedSAMAdapterLinear is injected
    has_medsam = False
    for _, m in base_vit.named_modules():
        if isinstance(m, MedSAMAdapterLinear):
            has_medsam = True
            assert isinstance(m.adapter, MedSAMClientAdapter)
    assert has_medsam is True

    # Forward pass through adapted model
    x = torch.randn(2, 1, 28, 28)
    out = base_vit(x)
    assert out.shape == (2, 3)

    # Extract adapter state dict
    peft_state = extract_peft_state_dict(base_vit, mode="medsam")
    assert len(peft_state) > 0
    # Must contain MedSAM bottleneck adapter parameters
    assert any("adapter.down_proj.weight" in k for k in peft_state)
    assert any("adapter.up_proj.weight" in k for k in peft_state)

    # Reload weights into a fresh instance
    new_vit = create_medical_vit(
        model_type="vit_tiny",
        img_size=28,
        patch_size=4,
        num_classes=3,
        pretrained=False,
    )
    inject_lora_to_model(new_vit, mode="medsam", r=16)
    load_peft_state_dict(new_vit, peft_state, mode="medsam")


def test_feddat_dual_adapter_injection_and_peft_extraction():
    """Verify FedDATDualAdapter cross-attention wraps layers and participates in PEFT aggregation."""
    base_vit = create_medical_vit(
        model_type="vit_tiny",
        img_size=28,
        patch_size=4,
        num_classes=3,
        pretrained=False,
    )
    report = inject_lora_to_model(base_vit, mode="feddat", r=16)
    assert report["injected_adapters"] > 0
    assert report["mode"] == "feddat"

    # Confirm FedDATAdapterLinear is injected
    has_feddat = False
    for _, m in base_vit.named_modules():
        if isinstance(m, FedDATAdapterLinear):
            has_feddat = True
            assert isinstance(m.adapter, FedDATDualAdapter)
    assert has_feddat is True

    # Forward pass
    x = torch.randn(2, 1, 28, 28)
    out = base_vit(x)
    assert out.shape == (2, 3)

    # Extract adapter state dict
    peft_state = extract_peft_state_dict(base_vit, mode="feddat")
    assert len(peft_state) > 0
    assert any("adapter.vision_adapter" in k for k in peft_state)
    assert any("adapter.cross_attn" in k for k in peft_state)

    # Reload
    new_vit = create_medical_vit(
        model_type="vit_tiny",
        img_size=28,
        patch_size=4,
        num_classes=3,
        pretrained=False,
    )
    inject_lora_to_model(new_vit, mode="feddat", r=16)
    load_peft_state_dict(new_vit, peft_state, mode="feddat")


def test_fedcola_modality_bridge_client_and_server_lifecycle():
    """Verify FedColaModalityBridge aligns features during training and aggregates globally."""
    # Create synthetic dataset
    imgs = torch.randn(12, 1, 28, 28)
    lbls = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    loader = DataLoader(TensorDataset(imgs, lbls), batch_size=4)

    client = HospitalClient(
        client_id=0,
        dataloader=loader,
        model_type="vit_tiny",
        img_size=28,
        patch_size=4,
        pretrained=False,
        peft_mode="ffa_lora",
    )

    initial_weights = client.get_weights()
    updated_weights, metrics, _ = client.train(
        global_weights=initial_weights,
        algorithm="FedCola",
        loss_fn_name="CE",
        current_round=1,
        total_rounds=1,
    )

    assert client.cola_bridge is not None
    assert isinstance(client.cola_bridge, FedColaModalityBridge)
    # Check that bridge weights are included in updated_weights
    assert any("cola_bridge.img_to_shared" in k for k in updated_weights)
    assert any("cola_bridge.txt_to_shared" in k for k in updated_weights)
    assert metrics.samples_trained == 12


def test_server_client_federated_round_with_advanced_modules():
    """Verify complete federated round execution using FedMedCLIP model and FedCola algorithm."""
    imgs1 = torch.randn(8, 1, 28, 28)
    lbls1 = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    loader1 = DataLoader(TensorDataset(imgs1, lbls1), batch_size=4)

    imgs2 = torch.randn(8, 1, 28, 28)
    lbls2 = torch.tensor([1, 2, 0, 1, 2, 0, 1, 2])
    loader2 = DataLoader(TensorDataset(imgs2, lbls2), batch_size=4)

    test_imgs = torch.randn(6, 1, 28, 28)
    test_lbls = torch.tensor([0, 1, 2, 0, 1, 2])

    server = CentralServer(
        model_type="fedmedclip",
        img_size=28,
        patch_size=4,
        peft_mode=None,
        pretrained=False,
        strict_checkpoint=False,
    )
    assert isinstance(server.global_model, FedMedCLIPModel)

    clients = [
        HospitalClient(
            client_id=0,
            dataloader=loader1,
            model_type="fedmedclip",
            img_size=28,
            patch_size=4,
            pretrained=False,
            strict_checkpoint=False,
        ),
        HospitalClient(
            client_id=1,
            dataloader=loader2,
            model_type="fedmedclip",
            img_size=28,
            patch_size=4,
            pretrained=False,
            strict_checkpoint=False,
        ),
    ]

    metrics, client_metrics, test_metrics, sec_report = run_federated_round(
        server=server,
        clients=clients,
        round_num=1,
        total_rounds=1,
        test_images=test_imgs,
        test_labels=test_lbls,
        algorithm="FedCola",
        loss_fn_name="CE",
    )

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert len(client_metrics) == 2
    assert test_metrics is not None
    assert test_metrics.accuracy >= 0.0

    # Ensure server successfully received and aggregated cola_bridge parameters
    assert server.cola_bridge is not None
    srv_weights = server.get_global_weights()
    assert any("cola_bridge.img_to_shared" in k for k in srv_weights)
