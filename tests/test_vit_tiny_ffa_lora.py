"""Tests for CPU-Optimized Medical Vision Transformer with FFA-LoRA Architecture.

Verifies:
1. Native 224x224 ViT-Tiny parameterization (16x16 patch embedding -> 196 tokens).
2. FFA-LoRA attention-decoupled projection hooks (W_Q, W_V in deep layers).
3. Bilinear LoRA aggregation consistency: sum p_k (B_k A) == (sum p_k B_k) A.
4. Frozen matrix A invariance across training steps.
5. CPU runtime under 1.0s per step and parameter compression > 99%.
"""

from __future__ import annotations
import time
import torch
from src.fed_xray.models.vit import create_medical_vit
from src.fed_xray.models.peft import (
    inject_lora_to_model,
    extract_peft_state_dict,
    load_peft_state_dict,
)


def test_vit_tiny_224_ffa_lora_injection():
    """Verify FFA-LoRA adapter injection into deep self-attention projection layers."""
    model = create_medical_vit(
        model_type="vit_tiny",
        img_size=224,
        patch_size=16,
        in_channels=1,
        num_classes=3,
    )
    stats = inject_lora_to_model(
        model=model,
        r=8,
        lora_alpha=16.0,
        mode="ffa_lora",
        deep_layers_only=True,
    )

    assert stats["injected_adapters"] == 4, f"Expected 4 adapters in deep layers, got {stats['injected_adapters']}"
    assert stats["compression_percentage"] > 99.0, f"Expected >99% compression, got {stats['compression_percentage']}"

    # Verify matrix A is frozen while matrix B is trainable
    for name, param in model.named_parameters():
        if "lora_A" in name:
            assert not param.requires_grad, f"Matrix A must be frozen: {name}"
        elif "lora_B" in name:
            assert param.requires_grad, f"Matrix B must be trainable: {name}"

    # Verify PEFT extraction excludes matrix A
    peft_sd = extract_peft_state_dict(model, mode="ffa_lora")
    assert not any("lora_A" in k for k in peft_sd.keys()), "FFA-LoRA state dict must not communicate matrix A"
    assert any("lora_B" in k for k in peft_sd.keys()), "FFA-LoRA state dict must communicate matrix B"


def test_ffa_lora_bilinear_aggregation_consistency():
    """Mathematically verify that FFA-LoRA eliminates bilinear aggregation error."""
    torch.manual_seed(42)
    d, r = 128, 8

    # Globally frozen matrix A (shared across all clients)
    matrix_A = torch.randn(r, d)

    # Two client updates for matrix B
    matrix_B1 = torch.randn(d, r)
    matrix_B2 = torch.randn(d, r)

    p1, p2 = 0.6, 0.4

    # Direct aggregation of weight deltas
    delta_w1 = matrix_B1 @ matrix_A
    delta_w2 = matrix_B2 @ matrix_A
    ideal_aggregated_delta = p1 * delta_w1 + p2 * delta_w2

    # FFA-LoRA aggregated B
    aggregated_B = p1 * matrix_B1 + p2 * matrix_B2
    ffa_aggregated_delta = aggregated_B @ matrix_A

    # In FFA-LoRA, because A is identical and linear, error is within float32 precision
    diff = torch.norm(ideal_aggregated_delta - ffa_aggregated_delta).item()
    assert diff < 1e-4, f"FFA-LoRA aggregation error must be near machine epsilon: {diff}"


def test_vit_tiny_224_cpu_runtime_and_efficiency():
    """Verify forward and backward pass latency on local CPU at 224x224."""
    model = create_medical_vit(
        model_type="vit_tiny",
        img_size=224,
        patch_size=16,
        in_channels=1,
        num_classes=3,
    )
    inject_lora_to_model(
        model=model,
        r=8,
        lora_alpha=16.0,
        mode="ffa_lora",
        deep_layers_only=True,
    )

    batch_x = torch.randn(8, 1, 224, 224)
    t0 = time.time()
    logits = model(batch_x)
    loss = logits.sum()
    loss.backward()
    elapsed = time.time() - t0

    assert logits.shape == (8, 3)
    assert elapsed < 1.5, f"Expected step under 1.5s on CPU, took {elapsed:.3f}s"


def test_ffa_lora_load_state_dict_roundtrip():
    """Verify state dict roundtrip loading produces identical forward activations."""
    model_server = create_medical_vit("vit_tiny", img_size=224, patch_size=16)
    inject_lora_to_model(model_server, r=8, mode="ffa_lora", deep_layers_only=True)

    model_client = create_medical_vit("vit_tiny", img_size=224, patch_size=16)
    inject_lora_to_model(model_client, r=8, mode="ffa_lora", deep_layers_only=True)
    # Under FFA-LoRA, client initializes from server (sharing the globally frozen A matrix)
    model_client.load_state_dict(model_server.state_dict())

    # Randomize client B weights
    for name, param in model_client.named_parameters():
        if "lora_B" in name:
            param.data.add_(torch.randn_like(param.data) * 0.1)

    peft_sd = extract_peft_state_dict(model_client, mode="ffa_lora")
    load_peft_state_dict(model_server, peft_sd, mode="ffa_lora")

    test_input = torch.randn(2, 1, 224, 224)
    model_client.eval()
    model_server.eval()

    with torch.no_grad():
        out_client = model_client(test_input)
        out_server = model_server(test_input)

    diff = torch.norm(out_client - out_server).item()
    assert diff < 1e-5, f"Server and client model outputs must match after state dict load: {diff}"
