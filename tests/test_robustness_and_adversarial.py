"""Adversarial Robustness, Byzantine Defense, and Numerical Edge-Case Test Suite."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.privacy import ClientAttestation
from src.fed_xray.core.server import CentralServer, run_federated_round


def create_dummy_dataloader(n_samples: int = 40, label: int = 0) -> DataLoader:
    """Helper to create dummy dataloader."""
    images = torch.randn(n_samples, 1, 28, 28)
    labels = torch.full((n_samples,), label, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def test_byzantine_poisoning_detection_and_mitigation():
    """Verify Byzantine poisoning defense intercepts malicious client updates without self-incrimination."""
    device = torch.device("cpu")
    server = CentralServer(device=device, defense_mode=True, model_type="vit_tiny", peft_mode="ffa_lora")

    init_sd = server.global_model.state_dict()
    x_train = torch.ones(30, 1, 28, 28) * 2.0
    y_train = torch.zeros(30, dtype=torch.long)

    # 3 honest clients, 1 malicious client
    c0 = HospitalClient(0, DataLoader(TensorDataset(x_train.clone(), y_train.clone()), batch_size=10), device=device, model_type="vit_tiny", peft_mode="ffa_lora")
    c1 = HospitalClient(1, DataLoader(TensorDataset(x_train.clone(), y_train.clone()), batch_size=10), device=device, model_type="vit_tiny", peft_mode="ffa_lora")
    c2 = HospitalClient(2, DataLoader(TensorDataset(x_train.clone(), y_train.clone()), batch_size=10), device=device, malicious=True, model_type="vit_tiny", peft_mode="ffa_lora")
    c3 = HospitalClient(3, DataLoader(TensorDataset(x_train.clone(), y_train.clone()), batch_size=10), device=device, model_type="vit_tiny", peft_mode="ffa_lora")

    for cl in [c0, c1, c2, c3]:
        cl.model.load_state_dict(init_sd)

    test_images = torch.ones(20, 1, 28, 28) * 2.0
    test_labels = torch.zeros(20, dtype=torch.long)

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
    assert sec_report.defense_active is True
    assert sec_report.threat_detected is True
    assert 2 in sec_report.clients_blocked
    assert 0 in sec_report.clients_accepted
    assert 1 in sec_report.clients_accepted
    assert 3 in sec_report.clients_accepted
    assert len(sec_report.validation_accuracies) == 4

    m2 = next(m for m in client_metrics if m.client_id == 2)
    assert m2.was_blocked is True


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
    """Verify Byzantine poisoning is detected, blocked, and isolated under SecAgg+ and CKKS without self-incrimination."""
    device = torch.device("cpu")
    server = CentralServer(
        device=device,
        defense_mode=True,
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        secure_aggregation=True,
        use_ckks=True,
    )

    init_sd = server.global_model.state_dict()
    x_train = torch.ones(30, 1, 28, 28) * 2.0
    y_train = torch.zeros(30, dtype=torch.long)

    # 3 honest clients, 1 malicious client
    c0 = HospitalClient(0, DataLoader(TensorDataset(x_train.clone(), y_train.clone()), batch_size=10), device=device, model_type="vit_tiny", peft_mode="ffa_lora")
    c1 = HospitalClient(1, DataLoader(TensorDataset(x_train.clone(), y_train.clone()), batch_size=10), device=device, model_type="vit_tiny", peft_mode="ffa_lora")
    c2 = HospitalClient(2, DataLoader(TensorDataset(x_train.clone(), y_train.clone()), batch_size=10), device=device, malicious=True, model_type="vit_tiny", peft_mode="ffa_lora")
    c3 = HospitalClient(3, DataLoader(TensorDataset(x_train.clone(), y_train.clone()), batch_size=10), device=device, model_type="vit_tiny", peft_mode="ffa_lora")

    for cl in [c0, c1, c2, c3]:
        cl.model.load_state_dict(init_sd)

    test_images = torch.ones(20, 1, 28, 28) * 2.0
    test_labels = torch.zeros(20, dtype=torch.long)

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
    assert sec_report.threat_detected is True
    assert 2 in sec_report.clients_blocked
    assert 0 in sec_report.clients_accepted
    assert 1 in sec_report.clients_accepted
    assert 3 in sec_report.clients_accepted
    assert len(sec_report.validation_accuracies) == 4
    # The report must not be a canned fake pass:
    assert sec_report.details != "SecAgg+ pairwise zero-sum masking active (cryptographic privacy preserves node confidentiality)"

    m2 = next(m for m in client_metrics if m.client_id == 2)
    assert m2.was_blocked is True


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


def test_byzantine_poisoning_blocks_malicious_node_without_self_incrimination():
    """Verify Byzantine defense isolates adversaries via consensus without relying on self-incrimination."""
    device = torch.device("cpu")
    server = CentralServer(
        device=device,
        defense_mode=True,
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        secure_aggregation=True,
        use_ckks=True,
    )

    # Consortium foundation backbone state dict
    init_sd = server.global_model.state_dict()

    x_train = torch.ones(30, 1, 28, 28) * 2.0
    y_train = torch.zeros(30, dtype=torch.long)

    dataset_good_0 = TensorDataset(x_train.clone(), y_train.clone())
    dataset_good_1 = TensorDataset(x_train.clone(), y_train.clone())
    dataset_bad = TensorDataset(x_train.clone(), y_train.clone())
    dataset_good_3 = TensorDataset(x_train.clone(), y_train.clone())

    c0 = HospitalClient(0, DataLoader(dataset_good_0, batch_size=10), device=device, model_type="vit_tiny", peft_mode="ffa_lora")
    c1 = HospitalClient(1, DataLoader(dataset_good_1, batch_size=10), device=device, model_type="vit_tiny", peft_mode="ffa_lora")
    c2 = HospitalClient(2, DataLoader(dataset_bad, batch_size=10), device=device, malicious=True, model_type="vit_tiny", peft_mode="ffa_lora")
    c3 = HospitalClient(3, DataLoader(dataset_good_3, batch_size=10), device=device, model_type="vit_tiny", peft_mode="ffa_lora")

    for cl in [c0, c1, c2, c3]:
        cl.model.load_state_dict(init_sd)

    # Verify that HospitalClient natively generates authentic attestations with is_malicious=False
    # (In zero-trust enclaves, adversaries never self-incriminate in signed certificates)
    att_c2 = c2.generate_attestation()
    assert att_c2.is_malicious is False

    test_images = torch.ones(20, 1, 28, 28) * 2.0
    test_labels = torch.zeros(20, dtype=torch.long)

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
    assert sec_report.threat_detected is True
    assert 2 in sec_report.clients_blocked
    assert 0 in sec_report.clients_accepted
    assert 1 in sec_report.clients_accepted
    assert 3 in sec_report.clients_accepted


def test_byzantine_gradient_scaling_attack_detected_by_l2_norm():
    """Verify Byzantine gradient scaling attacks (>3.5x median norm) are blocked by L2 norm detection."""
    device = torch.device("cpu")
    server = CentralServer(device=device, defense_mode=True, model_type="vit_tiny", peft_mode="ffa_lora")

    global_weights = server.get_global_weights()
    w0 = {k: v.clone() for k, v in global_weights.items()}
    w1 = {k: v.clone() for k, v in global_weights.items()}
    w2 = {k: v.clone() for k, v in global_weights.items()}
    # Attacker amplifies update norm by 10x
    w_malicious = {k: v.clone() * 10.0 for k, v in global_weights.items()}

    test_images = torch.randn(10, 1, 28, 28)
    test_labels = torch.zeros(10, dtype=torch.long)

    # Server evaluates accuracy uniformly to isolate norm detection
    server._validate_client_model = lambda w, img, lbl: 0.90

    _, report = server.validate_and_aggregate(
        client_weights=[w0, w1, w2, w_malicious],
        sample_counts=[50, 50, 50, 50],
        client_ids=[0, 1, 2, 3],
        test_images=test_images,
        test_labels=test_labels,
    )

    assert report.threat_detected is True
    assert 3 in report.clients_blocked
    assert 0 in report.clients_accepted
    assert 1 in report.clients_accepted
    assert 2 in report.clients_accepted


def test_byzantine_attestation_spoofing_detection():
    """Verify that forged high accuracy in client attestation is caught by server-side verification."""
    device = torch.device("cpu")
    server = CentralServer(device=device, defense_mode=True, model_type="vit_tiny", peft_mode="ffa_lora")

    global_weights = server.get_global_weights()
    w0 = {k: v.clone() for k, v in global_weights.items()}
    w1 = {k: v.clone() for k, v in global_weights.items()}
    w2 = {k: v.clone() for k, v in global_weights.items()}

    # Honest attestations match true performance (90%)
    att0 = ClientAttestation(client_id=0, validation_accuracy=0.90, update_l2_norm=1.0, public_key_bytes=b"")
    att1 = ClientAttestation(client_id=1, validation_accuracy=0.90, update_l2_norm=1.0, public_key_bytes=b"")
    # Adversary claims 99% accuracy in attestation, but its weights achieve only 5% on server validation
    att2_spoofed = ClientAttestation(client_id=2, validation_accuracy=0.99, update_l2_norm=1.0, public_key_bytes=b"")

    test_images = torch.randn(10, 1, 28, 28)
    test_labels = torch.zeros(10, dtype=torch.long)

    # Server independently validates weights: node 0 and 1 get 0.90, node 2 gets 0.05
    server._validate_client_model = lambda w, img, lbl: 0.90 if w is not w2 else 0.05

    _, report = server.validate_and_aggregate(
        client_weights=[w0, w1, w2],
        sample_counts=[50, 50, 50],
        client_ids=[0, 1, 2],
        test_images=test_images,
        test_labels=test_labels,
        client_attestations=[att0, att1, att2_spoofed],
    )

    assert report.threat_detected is True
    assert 2 in report.clients_blocked
    assert 0 in report.clients_accepted
    assert 1 in report.clients_accepted


def test_client_generates_authentic_local_dataloader_accuracy():
    """Verify that when test_images is None, HospitalClient evaluates genuinely on its own dataloader."""
    device = torch.device("cpu")
    dataset = TensorDataset(torch.randn(20, 1, 28, 28), torch.zeros(20, dtype=torch.long))
    client = HospitalClient(0, DataLoader(dataset, batch_size=10), device=device, model_type="vit_tiny", peft_mode="ffa_lora")

    # Call generate_attestation without external test images
    attestation = client.generate_attestation(test_images=None, test_labels=None)

    assert isinstance(attestation.validation_accuracy, float)
    assert 0.0 <= attestation.validation_accuracy <= 1.0
    assert attestation.update_l2_norm >= 0.0
