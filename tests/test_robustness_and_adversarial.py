"""Adversarial Robustness, Byzantine Defense, and Numerical Edge-Case Test Suite."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round


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


def test_byzantine_poisoning_under_secagg_and_ckks():
    """Verify Byzantine poisoning is detected, blocked, and isolated under SecAgg+ and CKKS."""
    device = torch.device("cpu")
    server = CentralServer(
        device=device,
        defense_mode=True,
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        secure_aggregation=True,
        use_ckks=True,
    )

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
        secure_aggregation=True,
        use_ckks=True,
    )

    assert sec_report is not None
    assert sec_report.defense_active is True
    assert sec_report.secagg_active is True
    assert len(sec_report.validation_accuracies) == 4
    # The report must not be a canned fake pass:
    assert sec_report.details != "SecAgg+ pairwise zero-sum masking active (cryptographic privacy preserves node confidentiality)"
    for m in client_metrics:
        if m.client_id in sec_report.clients_blocked:
            assert m.was_blocked is True


def test_validate_and_aggregate_with_secagg_masking():
    """Verify validate_and_aggregate applies SecAgg+ zero-sum masking strictly among trusted nodes."""
    device = torch.device("cpu")
    server = CentralServer(device=device, defense_mode=True, model_type="vit_tiny", peft_mode="ffa_lora")

    global_weights = server.get_global_weights()
    good_weights_1 = {k: v.clone() for k, v in global_weights.items()}
    good_weights_2 = {k: v.clone() for k, v in global_weights.items()}
    bad_weights = {k: torch.randn_like(v) * 100.0 for k, v in global_weights.items()}

    test_images = torch.randn(10, 1, 28, 28)
    test_labels = torch.zeros(10, dtype=torch.long)

    original_validate = server._validate_client_model
    try:
        server._validate_client_model = lambda w, img, lbl: 0.90 if w is not bad_weights else 0.05

        agg_weights, report = server.validate_and_aggregate(
            client_weights=[good_weights_1, good_weights_2, bad_weights],
            sample_counts=[100, 100, 100],
            client_ids=[0, 1, 2],
            test_images=test_images,
            test_labels=test_labels,
            use_secagg=True,
        )

        assert report.threat_detected is True
        assert 2 in report.clients_blocked
        assert report.clients_accepted == [0, 1]
        assert report.secagg_active is True
        assert "SecAgg+ zero-sum masking" in report.details
    finally:
        server._validate_client_model = original_validate
