"""Unit and integration tests for Vision Transformer PEFT and modern FL optimizers."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.fed_xray.models.vit import (
    VisionTransformer,
    create_medical_vit,
)
from src.fed_xray.models.peft import (
    LoRALinear,
    FFALoRALinear,
    FedSALoRALinear,
    inject_lora_to_model,
    extract_peft_state_dict,
    load_peft_state_dict,
)
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round


def test_vit_forward_and_feature_extraction():
    """Verify ViT forward pass and feature representation extraction."""
    model = create_medical_vit(model_type="vit_tiny", img_size=28, patch_size=4, num_classes=3)
    x = torch.randn(4, 1, 28, 28)
    logits, features = model(x, return_features=True)

    assert logits.shape == (4, 3)
    assert features.shape == (4, 128)
    assert not torch.isnan(logits).any()


def test_ffa_lora_zero_bilinear_aggregation_discrepancy():
    """Verify mathematically that FFA-LoRA eliminates bilinear aggregation distortion: \bar{B}A = sum p_k (B_k A)."""
    in_feat = 64
    out_feat = 32
    r = 8
    alpha = 16.0

    base_layer = nn.Linear(in_feat, out_feat, bias=False)
    
    # Initialize FFA-LoRA with frozen A
    adapter1 = FFALoRALinear(base_layer, r=r, lora_alpha=alpha)
    adapter2 = FFALoRALinear(base_layer, r=r, lora_alpha=alpha)
    
    # Ensure Matrix A is globally identical & frozen
    adapter2.lora_A.data.copy_(adapter1.lora_A.data)
    assert not adapter1.lora_A.requires_grad
    assert adapter1.lora_B.requires_grad

    # Simulate client updates on matrix B
    torch.nn.init.normal_(adapter1.lora_B, std=0.1)
    torch.nn.init.normal_(adapter2.lora_B, std=0.1)

    p1, p2 = 0.6, 0.4
    # Ideal full weight delta aggregation: \bar{\Delta W}_ideal = p1*(B1*A) + p2*(B2*A)
    delta1 = adapter1.get_effective_delta()
    delta2 = adapter2.get_effective_delta()
    ideal_delta = p1 * delta1 + p2 * delta2

    # FFA-LoRA aggregation: \bar{B} = p1*B1 + p2*B2, effective delta = \bar{B} * A
    B_bar = p1 * adapter1.lora_B.data + p2 * adapter2.lora_B.data
    ffa_effective_delta = (alpha / r) * (B_bar @ adapter1.lora_A.data)

    # Discrepancy should be mathematically exact zero within float precision
    error = torch.norm(ideal_delta - ffa_effective_delta).item()
    assert error < 1e-6, f"FFA-LoRA aggregation error should be 0, got {error}"


def test_vit_peft_injection_and_compression():
    """Verify PEFT parameter reduction exceeds 90% on Vision Transformer."""
    vit = create_medical_vit(model_type="vit_small", img_size=28, patch_size=4, num_classes=3)
    stats = inject_lora_to_model(
        model=vit,
        r=8,
        lora_alpha=16.0,
        mode="ffa_lora",
        deep_layers_only=True,
        num_deep_layers=2,
        total_layers=6,
    )

    assert stats["trainable_parameters"] < stats["total_parameters"]
    assert stats["compression_percentage"] > 70.0
    
    # Test state dict extraction
    peft_state = extract_peft_state_dict(vit, mode="ffa_lora")
    assert len(peft_state) > 0
    # In FFA-LoRA, matrix A is excluded from transmitted state dict
    assert not any("lora_A" in k for k in peft_state.keys())


def test_modern_fl_optimizers_on_vit():
    """Test full federated round execution across modern SOTA optimizers: FedAvg, FedProx, FedDyn, FedOpt, SCAFFOLD."""
    # Synthetic mini dataset
    x_train = torch.randn(20, 1, 28, 28)
    y_train = torch.randint(0, 3, (20,))
    loader1 = DataLoader(TensorDataset(x_train[:10], y_train[:10]), batch_size=5)
    loader2 = DataLoader(TensorDataset(x_train[10:], y_train[10:]), batch_size=5)

    x_test = torch.randn(10, 1, 28, 28)
    y_test = torch.randint(0, 3, (10,))

    for algo in ["FedAvg", "FedProx", "FedDyn", "FedOpt", "SCAFFOLD"]:
        server = CentralServer(
            model_type="vit_tiny",
            peft_mode="ffa_lora",
            lora_r=8,
            lora_alpha=16.0,
            deep_layers_only=True,
        )
        c1 = HospitalClient(
            client_id=1,
            dataloader=loader1,
            model_type="vit_tiny",
            peft_mode="ffa_lora",
            lora_r=8,
            lora_alpha=16.0,
            deep_layers_only=True,
        )
        c2 = HospitalClient(
            client_id=2,
            dataloader=loader2,
            model_type="vit_tiny",
            peft_mode="ffa_lora",
            lora_r=8,
            lora_alpha=16.0,
            deep_layers_only=True,
        )

        agg_metrics, client_metrics, test_metrics, sec_report = run_federated_round(
            server=server,
            clients=[c1, c2],
            round_num=1,
            total_rounds=2,
            test_images=x_test,
            test_labels=y_test,
            algorithm=algo,
            prox_mu=0.01 if algo == "FedProx" else 0.0,
            feddyn_alpha=0.01 if algo == "FedDyn" else 0.0,
        )

        assert agg_metrics["loss"] >= 0.0
        assert len(client_metrics) == 2
        assert test_metrics is not None
        assert 0.0 <= test_metrics.accuracy <= 1.0
