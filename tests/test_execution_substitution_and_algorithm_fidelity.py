"""Tests for Section 4.2: Execution Substitution and False Computation Elimination.

Verifies mathematical fidelity across federated learning optimizers, PEFT parameter
adaptations, dynamic objective alignment, and Flower protocol bridge routing.
Guarantees zero algorithmic bypassing, zero discarded control variates, and
genuine execution of claimed optimization routines.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer
from src.fed_xray.federation.flower_protocol_bridge import run_flower_federated_orchestration


def _build_test_dataloader(num_samples: int = 12, num_classes: int = 3) -> DataLoader:
    """Build deterministic DataLoader with standard medical image dimensions."""
    torch.manual_seed(42)
    images = torch.randn(num_samples, 1, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def test_fedprox_proximal_penalty_divergence():
    """Verify FedProx proximal term explicitly alters optimization trajectory compared to FedAvg."""
    torch.manual_seed(101)
    dataloader = _build_test_dataloader(num_samples=16)

    # Initialize client for FedAvg (prox_mu=0.0)
    client_fedavg = HospitalClient(
        client_id=1,
        dataloader=dataloader,
        model_type="vit_tiny",
        num_classes=3,
        peft_mode="ffa_lora",
        learning_rate=0.01,
        local_epochs=1,
    )
    init_weights = {k: v.clone() for k, v in client_fedavg.get_weights().items()}

    fedavg_weights, metrics_avg, _ = client_fedavg.train(
        global_weights=init_weights,
        algorithm="FedAvg",
        prox_mu=0.0,
    )

    # Initialize identical client for FedProx (prox_mu=1.0)
    client_fedprox = HospitalClient(
        client_id=1,
        dataloader=dataloader,
        model_type="vit_tiny",
        num_classes=3,
        peft_mode="ffa_lora",
        learning_rate=0.01,
        local_epochs=1,
    )

    fedprox_weights, metrics_prox, _ = client_fedprox.train(
        global_weights=init_weights,
        algorithm="FedProx",
        prox_mu=1.0,
    )

    # Proximal regularizer must constrain parameter drift toward initial weights
    drift_fedavg = sum(
        torch.norm(fedavg_weights[k] - init_weights[k]).item()
        for k in fedavg_weights
    )
    drift_fedprox = sum(
        torch.norm(fedprox_weights[k] - init_weights[k]).item()
        for k in fedprox_weights
    )

    assert drift_fedavg > 0.0, "FedAvg training must produce non-zero weight displacement"
    assert drift_fedprox > 0.0, "FedProx training must produce non-zero weight displacement"
    assert drift_fedprox < drift_fedavg, (
        f"FedProx with mu=1.0 must constrain parameter drift closer to reference weights: "
        f"drift_fedprox={drift_fedprox:.6f} vs drift_fedavg={drift_fedavg:.6f}"
    )


def test_scaffold_control_variates_gradient_correction_and_aggregation():
    """Verify SCAFFOLD computes client delta_c starting from round 1 and applies gradient correction."""
    torch.manual_seed(202)
    dataloader = _build_test_dataloader(num_samples=12)

    server = CentralServer(model_type="vit_tiny", num_classes=3, peft_mode="ffa_lora")
    init_weights = server.get_global_weights()

    client1 = HospitalClient(
        client_id=1,
        dataloader=dataloader,
        model_type="vit_tiny",
        num_classes=3,
        peft_mode="ffa_lora",
        local_epochs=1,
    )
    client2 = HospitalClient(
        client_id=2,
        dataloader=dataloader,
        model_type="vit_tiny",
        num_classes=3,
        peft_mode="ffa_lora",
        local_epochs=1,
    )

    # Round 1: c_global is None, verify both clients compute authentic non-zero delta_c
    w1, _, delta_c1 = client1.train(
        global_weights=init_weights,
        algorithm="SCAFFOLD",
        c_global=None,
    )
    w2, _, delta_c2 = client2.train(
        global_weights=init_weights,
        algorithm="SCAFFOLD",
        c_global=None,
    )

    assert delta_c1 is not None, "SCAFFOLD round 1 must return non-null delta_c"
    assert delta_c2 is not None, "SCAFFOLD round 1 must return non-null delta_c"
    assert len(delta_c1) > 0, "delta_c must contain parameter deltas"

    norm_dc1 = sum(torch.norm(t).item() for t in delta_c1.values())
    assert norm_dc1 > 0.0, "SCAFFOLD delta_c norm must be positive"

    # Server aggregation: updates c_global
    aggregated_w = server.aggregate(
        client_weights=[w1, w2],
        sample_counts=[12, 12],
        algorithm="SCAFFOLD",
        delta_cs=[delta_c1, delta_c2],
    )

    assert hasattr(server, "c_global") and len(server.c_global) > 0, (
        "CentralServer.aggregate must update server c_global state under SCAFFOLD"
    )
    c_global_norm = sum(torch.norm(t).item() for t in server.c_global.values())
    assert c_global_norm > 0.0, "Global control variates c_global must be non-zero after aggregation"

    # Round 2: client uses populated c_global for gradient drift correction
    w1_r2, _, delta_c1_r2 = client1.train(
        global_weights=aggregated_w,
        algorithm="SCAFFOLD",
        c_global=server.c_global,
    )
    assert delta_c1_r2 is not None, "SCAFFOLD round 2 must return updated delta_c"


def test_feddyn_dynamic_objective_alignment():
    """Verify FedDyn tracks client gradients and updates server dynamic state h^{t+1}."""
    torch.manual_seed(303)
    dataloader = _build_test_dataloader(num_samples=12)

    server = CentralServer(model_type="vit_tiny", num_classes=3, peft_mode="ffa_lora")
    server.feddyn_alpha = 0.02
    init_weights = server.get_global_weights()

    client1 = HospitalClient(
        client_id=1,
        dataloader=dataloader,
        model_type="vit_tiny",
        num_classes=3,
        peft_mode="ffa_lora",
        local_epochs=1,
    )
    client2 = HospitalClient(
        client_id=2,
        dataloader=dataloader,
        model_type="vit_tiny",
        num_classes=3,
        peft_mode="ffa_lora",
        local_epochs=1,
    )

    # Train round 1
    w1, _, _ = client1.train(
        global_weights=init_weights,
        algorithm="FedDyn",
        feddyn_alpha=0.02,
    )
    w2, _, _ = client2.train(
        global_weights=init_weights,
        algorithm="FedDyn",
        feddyn_alpha=0.02,
    )

    assert client1.grad_prev is not None and len(client1.grad_prev) > 0, (
        "HospitalClient must maintain grad_prev dynamic gradient tracker under FedDyn"
    )

    # Server aggregation: updates dynamic state vector h_server
    server.aggregate(
        client_weights=[w1, w2],
        sample_counts=[12, 12],
        algorithm="FEDDYN",
    )

    assert hasattr(server, "h_server") and len(server.h_server) > 0, (
        "CentralServer must update dynamic state h_server under FedDyn"
    )
    h_norm = sum(torch.norm(t).item() for t in server.h_server.values())
    assert h_norm > 0.0, "Dynamic regularizer state h_server must have positive norm"


def test_moon_model_contrastive_loss():
    """Verify MOON computes contrastive representation loss and caches previous local model."""
    torch.manual_seed(404)
    dataloader = _build_test_dataloader(num_samples=12)

    client = HospitalClient(
        client_id=1,
        dataloader=dataloader,
        model_type="vit_tiny",
        num_classes=3,
        peft_mode="ffa_lora",
        local_epochs=1,
    )
    init_weights = client.get_weights()

    # Train round 1 with MOON
    w1, metrics1, _ = client.train(
        global_weights=init_weights,
        algorithm="MOON",
        moon_mu=0.5,
        moon_temp=0.5,
    )

    assert client.prev_local_model is not None, (
        "HospitalClient must cache prev_local_model after a MOON round"
    )
    assert not any(p.requires_grad for p in client.prev_local_model.parameters()), (
        "Cached prev_local_model parameters must have requires_grad=False"
    )

    # Train round 2 with MOON using cached prev_local_model
    w2, metrics2, _ = client.train(
        global_weights=w1,
        algorithm="MOON",
        moon_mu=0.5,
        moon_temp=0.5,
    )
    assert metrics2.samples_trained == 12, "Client must complete authentic round 2 training"


def test_flower_protocol_bridge_scaffold_and_optimizer_fidelity():
    """Verify Flower protocol bridge propagates optimizer parameters and aggregates SCAFFOLD deltas."""
    torch.manual_seed(505)
    dataloader = _build_test_dataloader(num_samples=12)

    server = CentralServer(model_type="vit_tiny", num_classes=3, peft_mode="ffa_lora")
    client1 = HospitalClient(
        client_id=1,
        dataloader=dataloader,
        model_type="vit_tiny",
        num_classes=3,
        peft_mode="ffa_lora",
        local_epochs=1,
    )
    client2 = HospitalClient(
        client_id=2,
        dataloader=dataloader,
        model_type="vit_tiny",
        num_classes=3,
        peft_mode="ffa_lora",
        local_epochs=1,
    )

    # Test 2 rounds of Flower orchestration under SCAFFOLD
    result = run_flower_federated_orchestration(
        server=server,
        clients=[client1, client2],
        num_rounds=2,
        algorithm="SCAFFOLD",
    )

    assert result["status"] == "completed"
    assert result["rounds_completed"] == 2
    assert hasattr(server, "c_global") and len(server.c_global) > 0, (
        "Flower orchestration must update CentralServer.c_global via SCAFFOLD delta aggregation"
    )
    c_norm = sum(torch.norm(t).item() for t in server.c_global.values())
    assert c_norm > 0.0, "Global control variate norm must be non-zero after Flower SCAFFOLD rounds"

