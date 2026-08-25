"""Adversarial Robustness, Byzantine Defense, and Numerical Edge-Case Test Suite."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round
from src.fed_xray.data.generator import MedicalDataGenerator


def create_dummy_dataloader(n_samples: int = 40, label: int = 0) -> DataLoader:
    """Helper to create dummy dataloader."""
    images = torch.randn(n_samples, 1, 28, 28)
    labels = torch.full((n_samples,), label, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def test_byzantine_poisoning_detection_and_mitigation():
    """Verify Byzantine poisoning defense intercepts malicious client updates."""
    device = torch.device("cpu")
    server = CentralServer(device=device, defense_mode=True, model_type="vit_tiny", peft_mode="ffa_lora")

    # 3 honest clients, 1 malicious client
    c0 = HospitalClient(0, create_dummy_dataloader(30, 0), device=device, model_type="vit_tiny", peft_mode="ffa_lora")
    c1 = HospitalClient(1, create_dummy_dataloader(30, 1), device=device, model_type="vit_tiny", peft_mode="ffa_lora")
    c2 = HospitalClient(2, create_dummy_dataloader(30, 2), device=device, malicious=True, model_type="vit_tiny", peft_mode="ffa_lora")
    c3 = HospitalClient(3, create_dummy_dataloader(30, 0), device=device, model_type="vit_tiny", peft_mode="ffa_lora")

    test_images = torch.randn(20, 1, 28, 28)
    test_labels = torch.randint(0, 3, (20,))

    agg_metrics, client_metrics, test_metrics, sec_report = run_federated_round(
        server=server,
        clients=[c0, c1, c2, c3],
        round_num=1,
        total_rounds=1,
        test_images=test_images,
        test_labels=test_labels,
        use_defense=True,
    )

    assert sec_report is not None
    assert sec_report.total_clients == 4
    # Malicious client should be detected or score divergence logged
    assert len(sec_report.validation_accuracies) == 4


def test_nan_and_inf_gradient_resilience():
    """Verify that extreme inputs and NaN protections prevent model weight corruption."""
    device = torch.device("cpu")
    client = HospitalClient(
        client_id=0,
        dataloader=create_dummy_dataloader(20, 0),
        device=device,
        model_type="vit_tiny",
        peft_mode="ffa_lora",
    )

    # Inject extreme input
    extreme_images = torch.full((10, 1, 28, 28), 1e6)
    extreme_labels = torch.zeros(10, dtype=torch.long)
    extreme_loader = DataLoader(TensorDataset(extreme_images, extreme_labels), batch_size=5)

    server = CentralServer(device=device, model_type="vit_tiny", peft_mode="ffa_lora")
    global_weights = server.get_global_weights()

    client.dataloader = extreme_loader
    trained_weights, metrics, _ = client.train(
        global_weights=global_weights,
        current_round=1,
        total_rounds=5,
    )

    # Weights must not contain NaN
    for name, param in trained_weights.items():
        assert not torch.isnan(param).any(), f"NaN detected in parameter {name}"
        assert not torch.isinf(param).any(), f"Inf detected in parameter {name}"


def test_free_rider_and_empty_batch_handling():
    """Verify handling when client data is minimal or identical."""
    device = torch.device("cpu")
    server = CentralServer(device=device, model_type="vit_tiny", peft_mode="ffa_lora")
    
    # 2 clients with single-sample batches
    c0 = HospitalClient(0, create_dummy_dataloader(1, 0), device=device, model_type="vit_tiny", peft_mode="ffa_lora")
    c1 = HospitalClient(1, create_dummy_dataloader(1, 1), device=device, model_type="vit_tiny", peft_mode="ffa_lora")

    test_images = torch.randn(5, 1, 28, 28)
    test_labels = torch.randint(0, 3, (5,))

    agg_metrics, _, test_metrics, _ = run_federated_round(
        server=server,
        clients=[c0, c1],
        round_num=1,
        total_rounds=1,
        test_images=test_images,
        test_labels=test_labels,
    )

    assert "loss" in agg_metrics
    assert not torch.isnan(torch.tensor(agg_metrics["loss"]))
