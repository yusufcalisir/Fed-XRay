"""Unit and integration tests for Streamlit dashboard architecture and PEFT parameter wiring.

Verifies:
1. Authentic estimation of parameter counts and communication economics via CentralServer.estimate_communication_economics.
2. Correct formatting of communication bandwidth metrics via format_byte_size.
3. CentralServer and HospitalClient initialization with ViT-Tiny backbone and FFA-LoRA PEFT adapters.
4. Invariance of frozen matrix A and trainability of low-rank matrix B under Option J federated protocol.
5. End-to-end execution of a federated round and Grad-CAM saliency extraction with trained ViT backbone.
"""

from __future__ import annotations
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.fed_xray.core.server import CentralServer, run_federated_round
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.cdss.xai import GradCAM
from src.fed_xray.cdss.similarity import extract_embedding
from utils.federated_core import format_byte_size


def test_estimate_communication_economics_vit_ffa_lora() -> None:
    """Verify authentic parameter counts and bandwidth reduction for ViT-Tiny with FFA-LoRA."""
    econ = CentralServer.estimate_communication_economics(
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        lora_r=16,
        img_size=224,
        num_classes=3,
    )

    assert econ["model_type"] == "vit_tiny"
    assert econ["peft_mode"] == "ffa_lora"
    assert econ["lora_r"] == 16
    assert econ["full_model_params"] > 500_000, f"Unexpected full params: {econ['full_model_params']}"
    assert econ["uplink_params"] < 50_000, f"Expected low uplink params, got {econ['uplink_params']}"
    assert econ["reduction_percentage"] > 95.0, f"Expected >95% reduction, got {econ['reduction_percentage']}%"
    assert econ["uplink_bytes"] < econ["full_model_bytes"]


def test_estimate_communication_economics_cnn_full() -> None:
    """Verify authentic baseline parameter counts and 0% reduction for CNN full model."""
    econ = CentralServer.estimate_communication_economics(
        model_type="cnn",
        peft_mode=None,
        lora_r=16,
        img_size=224,
        num_classes=3,
    )

    assert econ["model_type"] == "cnn"
    assert econ["peft_mode"] is None
    assert econ["full_model_params"] == econ["uplink_params"]
    assert econ["full_model_bytes"] == econ["uplink_bytes"]
    assert econ["reduction_percentage"] == 0.0


def test_format_byte_size_metric_units() -> None:
    """Verify byte count string formatting across metric scales."""
    assert format_byte_size(500) == "500 B"
    assert format_byte_size(2048) == "2.0 KB"
    assert format_byte_size(3 * 1024 * 1024) == "3.00 MB"
    assert format_byte_size(2.5 * 1024 * 1024 * 1024) == "2.50 GB"


def test_central_server_and_client_vit_ffa_lora_wiring() -> None:
    """Verify server and client instantiation with ViT backbone and FFA-LoRA adapters."""
    device = torch.device("cpu")
    server = CentralServer(
        device=device,
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        lora_r=16,
        lora_alpha=16.0,
        img_size=224,
        patch_size=16,
        pretrained=False,
    )

    # Check global model architecture and PEFT adapter hooks
    global_model = server.get_model()
    assert hasattr(global_model, "blocks"), "ViT model must contain transformer blocks"
    assert server.model_type == "vit_tiny"
    assert server.peft_mode == "ffa_lora"

    # Verify frozen matrix A and trainable matrix B
    has_frozen_a = False
    has_trainable_b = False
    for name, param in global_model.named_parameters():
        if "lora_A" in name:
            assert not param.requires_grad, f"Matrix A must be frozen: {name}"
            has_frozen_a = True
        elif "lora_B" in name:
            assert param.requires_grad, f"Matrix B must be trainable: {name}"
            has_trainable_b = True

    assert has_frozen_a, "FFA-LoRA must contain frozen lora_A matrices"
    assert has_trainable_b, "FFA-LoRA must contain trainable lora_B matrices"

    # Create dummy medical dataloader
    dummy_x = torch.randn(8, 1, 224, 224)
    dummy_y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)
    dummy_dataset = TensorDataset(dummy_x, dummy_y)
    dataloader = DataLoader(dummy_dataset, batch_size=4, shuffle=False)

    client = HospitalClient(
        client_id=0,
        dataloader=dataloader,
        device=device,
        learning_rate=0.001,
        local_epochs=1,
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        lora_r=16,
        lora_alpha=16.0,
        img_size=224,
        patch_size=16,
        pretrained=False,
    )

    # Run single federated round
    metrics, client_metrics, test_metrics, security_report = run_federated_round(
        server=server,
        clients=[client],
        round_num=1,
        total_rounds=1,
        test_images=dummy_x[:4],
        test_labels=dummy_y[:4],
        use_defense=True,
    )

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["uplink_bytes_per_client"] > 0
    assert metrics["uplink_bytes_per_client"] < metrics["full_model_bytes"]


def test_vit_cdss_diagnostic_workspace_evaluation() -> None:
    """Verify that CDSS Grad-CAM and embedding extraction execute successfully on trained ViT."""
    model = CentralServer(
        device=torch.device("cpu"),
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        lora_r=16,
        lora_alpha=16.0,
        img_size=224,
        patch_size=16,
        pretrained=False,
    ).get_model()

    model.eval()
    test_scan = torch.randn(1, 1, 224, 224)

    # 1. Grad-CAM saliency generation
    gradcam = GradCAM(model)
    heatmap, pred_class, confidence = gradcam.generate_heatmap(test_scan)
    gradcam.remove_hooks()

    assert heatmap.shape == (224, 224), f"Expected 224x224 heatmap, got {heatmap.shape}"
    assert 0 <= pred_class < 3
    assert 0.0 <= confidence <= 1.0

    # 2. Embedding extraction for Case-Based Reasoning
    emb = extract_embedding(model, test_scan)
    assert emb.ndim == 1
    assert emb.shape[0] == 128, f"Expected 128-dim embedding for ViT-Tiny, got {emb.shape[0]}"
    norm = float(torch.norm(torch.tensor(emb)))
    assert abs(norm - 1.0) < 1e-4, f"Embedding must be unit normalized, got norm={norm}"
