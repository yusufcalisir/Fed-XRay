"""Unit and Integration Tests for SOTA 2026 Upgrades:
1. 224x224 ViT-Base Pretrained Foundation Backbone + FFA-LoRA
2. HNSW Vector Database Indexing for Federated RAG Case Retrieval
3. Flower (flwr v1.8+) gRPC Hospital Client and Server Strategy Bridge
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fed_xray.cdss.similarity import FederatedRAGCaseMatcher
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer
from src.fed_xray.federation.flower_protocol_bridge import (
    FlowerFedXRayStrategy,
    FlowerHospitalClient,
    InProcessClientProxy,
    flower_parameters_to_state_dict,
    run_flower_federated_orchestration,
    state_dict_to_flower_parameters,
)
from src.fed_xray.models.peft import inject_lora_to_model
from src.fed_xray.models.vit import create_medical_vit


def test_vit_base_224_pretrained_and_ffa_lora():
    """Verify 224x224 ViT-Base with pretrained foundation weights and FFA-LoRA injection."""
    vit = create_medical_vit(
        model_type="vit_base",
        img_size=224,
        patch_size=16,
        in_channels=1,
        num_classes=3,
        pretrained=True,
    )
    x = torch.randn(2, 1, 224, 224)
    logits = vit(x)
    assert logits.shape == (2, 3), "ViT-Base output logits shape mismatch"

    # Inject FFA-LoRA into deep self-attention layers
    info = inject_lora_to_model(vit, r=8, lora_alpha=16.0, mode="ffa_lora", deep_layers_only=True)
    assert info["injected_adapters"] > 0
    assert info["mode"] == "ffa_lora"
    assert info["trainable_parameters"] < 100_000, "Trainable PEFT parameters must be compact"
    assert info["total_parameters"] > 80_000_000, "Base foundation model must maintain 80M+ capacity"


def test_hnsw_vector_indexing_federated_rag():
    """Verify HNSW sub-millisecond vector indexing on verified clinical cases."""
    from src.fed_xray.data.real_world import RealWorldPatientRecord

    records = [
        RealWorldPatientRecord(
            patient_id=f"TEST-HNSW-{i}",
            site_id="Hospital-Beta",
            image_tensor=torch.rand(1, 28, 28, dtype=torch.float32),
            label=i % 3,
            clinical_notes=f"HNSW test record {i}",
        )
        for i in range(25)
    ]
    matcher = FederatedRAGCaseMatcher(reference_records=records)
    assert matcher.hnsw_index is not None, "HNSW vector index must be initialized"

    query_vec = np.random.randn(matcher.embedding_dim).astype(np.float32)
    results = matcher.find_similar(query_vec, top_k=3)
    assert len(results) == 3
    for r in results:
        assert "case_id" in r
        assert "similarity" in r
        assert 0.0 <= r["similarity"] <= 1.0

    dist = matcher.predict_rag_distribution(query_vec, top_k=4, num_classes=3)
    assert len(dist) == 3
    assert np.isclose(dist.sum(), 1.0, atol=1e-4)


def test_flower_parameter_conversion_roundtrip():
    """Verify state dict to Flower parameters and back preserves tensor values and device."""
    orig_state = {
        "weight_a": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "bias_b": torch.tensor([0.5, -0.5]),
    }
    ndarrays, keys = state_dict_to_flower_parameters(orig_state)
    assert len(ndarrays) == 2
    assert keys == ["weight_a", "bias_b"]

    recovered = flower_parameters_to_state_dict(ndarrays, keys, device=torch.device("cpu"))
    for k in orig_state:
        assert torch.allclose(orig_state[k], recovered[k])


def test_flower_hospital_client_and_strategy_flow():
    """Verify FlowerHospitalClient and FlowerFedXRayStrategy execution loop."""
    images = torch.randn(12, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    h_client = HospitalClient(
        client_id=0,
        dataloader=loader,
        learning_rate=0.01,
        local_epochs=1,
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        img_size=28,
        patch_size=4,
    )
    flower_client = FlowerHospitalClient(h_client)

    params = flower_client.get_parameters(config={})
    assert len(params) > 0

    # Test fit
    updated_params, num_samples, metrics = flower_client.fit(
        parameters=params,
        config={"current_round": 1, "total_rounds": 5, "algorithm": "FedAvg"},
    )
    assert num_samples == 12
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert len(updated_params) == len(params)

    # Test evaluate
    loss, eval_samples, eval_metrics = flower_client.evaluate(
        parameters=updated_params,
        config={"current_round": 1},
    )
    assert eval_samples == 12
    assert "accuracy" in eval_metrics
    assert loss >= 0.0

    # Test FlowerFedXRayStrategy initialization
    server = CentralServer(
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        img_size=28,
        patch_size=4,
    )
    strategy = FlowerFedXRayStrategy(
        central_server=server,
        total_rounds=5,
        algorithm="FedAvg",
        min_fit_clients=1,
        min_available_clients=1,
    )
    assert len(strategy.arrays) > 0


def test_flower_hospital_client_dp_parameter_propagation():
    """Verify FlowerHospitalClient.fit propagates Differential Privacy parameters to HospitalClient."""
    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    h_client = HospitalClient(
        client_id=1,
        dataloader=loader,
        learning_rate=0.01,
        local_epochs=1,
        model_type="cnn",
        dp_enabled=False,
    )
    flower_client = FlowerHospitalClient(h_client)
    params = flower_client.get_parameters(config={})

    # Fit with DP parameters forwarded via config dict
    updated_params, num_samples, metrics = flower_client.fit(
        parameters=params,
        config={
            "current_round": 1,
            "total_rounds": 3,
            "algorithm": "FedAvg",
            "dp_enabled": True,
            "dp_clipping_bound": 0.5,
            "dp_noise_multiplier": 0.2,
            "dp_mode": "distributed",
            "num_consortium_clients": 3,
        },
    )
    assert num_samples == 8
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "dp_clipped_norm" in metrics
    assert metrics["dp_clipped_norm"] > 0.0


def test_flower_fedxray_strategy_aggregation_with_defense_and_privacy():
    """Verify FlowerFedXRayStrategy.aggregate_fit validates updates through defense and reports DP."""
    server = CentralServer(
        model_type="cnn",
        privacy_noise=0.3,
        defense_mode=True,
        dp_mode="distributed",
        dp_clipping_bound=1.0,
    )
    test_img = torch.randn(6, 1, 28, 28)
    test_lbl = torch.randint(0, 3, (6,))

    strategy = FlowerFedXRayStrategy(
        central_server=server,
        total_rounds=3,
        algorithm="FedAvg",
        min_fit_clients=2,
        min_available_clients=2,
        test_images=test_img,
        test_labels=test_lbl,
        secure_aggregation=False,
        use_ckks=False,
        total_consortium_hospitals=2,
    )

    # Create 2 clients and run 1 fit to produce real FitRes objects
    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    c0 = HospitalClient(client_id=0, dataloader=loader, model_type="cnn")
    c1 = HospitalClient(client_id=1, dataloader=loader, model_type="cnn")
    fc0 = FlowerHospitalClient(c0)
    fc1 = FlowerHospitalClient(c1)

    proxy0 = InProcessClientProxy("0", fc0)
    proxy1 = InProcessClientProxy("1", fc1)

    from flwr.common import FitIns, ndarrays_to_parameters
    fit_ins = FitIns(ndarrays_to_parameters(strategy.arrays), {"current_round": 1, "algorithm": "FedAvg"})

    res0 = proxy0.fit(fit_ins)
    res1 = proxy1.fit(fit_ins)

    agg_params, metrics = strategy.aggregate_fit(
        server_round=1,
        results=[(proxy0, res0), (proxy1, res1)],
        failures=[],
    )

    assert agg_params is not None
    assert metrics["participating_clients"] == 2
    assert "blocked_clients_count" in metrics
    assert metrics["dp_active"] is True
    assert metrics["privacy_epsilon"] > 0.0
    assert metrics["uplink_bytes_per_client"] > 0
    assert metrics["full_model_bytes"] > 0


def test_flower_in_process_orchestration_lifecycle():
    """Verify end-to-end multi-round orchestration using InProcessClientProxy and FlowerFedXRayStrategy."""
    torch.manual_seed(42)
    server = CentralServer(
        model_type="cnn",
        privacy_noise=0.1,
        dp_mode="distributed",
    )

    clients = [
        HospitalClient(
            client_id=i,
            dataloader=DataLoader(
                TensorDataset(torch.randn(8, 1, 28, 28), torch.randint(0, 3, (8,))),
                batch_size=4,
            ),
            model_type="cnn",
        )
        for i in range(2)
    ]

    test_images = torch.randn(6, 1, 28, 28)
    test_labels = torch.randint(0, 3, (6,))

    rounds_recorded = []

    def on_round(rnd, info):
        rounds_recorded.append((rnd, info))

    summary = run_flower_federated_orchestration(
        server=server,
        clients=clients,
        num_rounds=2,
        algorithm="FedAvg",
        test_images=test_images,
        test_labels=test_labels,
        on_round_end=on_round,
    )

    assert summary["status"] == "completed"
    assert summary["rounds_completed"] == 2
    assert len(summary["round_history"]) == 2
    assert len(rounds_recorded) == 2
    assert summary["privacy_epsilon"] >= 0.0
    assert "eval_metrics" in summary["round_history"][-1]
    assert "blocked_nodes" in rounds_recorded[0][1]


def test_flower_byzantine_blocked_node_identity_authenticity():
    """Verify that FlowerFedXRayStrategy and orchestration accurately propagate authentic blocked node IDs.

    Guarantees that when Byzantine defenses isolate client node 2 (e.g. malicious gradient scaling attack),
    FlowerFedXRayStrategy metrics and on_round_end payloads report blocked node 2 rather than
    fabricating or defaulting to node 1 (DEFECT-11).
    """
    import json
    torch.manual_seed(42)
    device = torch.device("cpu")
    server = CentralServer(
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        defense_mode=True,
        device=device,
    )
    test_img = torch.randn(12, 1, 28, 28)
    test_lbl = torch.randint(0, 3, (12,))

    strategy = FlowerFedXRayStrategy(
        central_server=server,
        total_rounds=1,
        algorithm="FedAvg",
        min_fit_clients=3,
        min_available_clients=3,
        test_images=test_img,
        test_labels=test_lbl,
        total_consortium_hospitals=3,
    )

    # 2 honest clients and 1 malicious client (node 2)
    x_data = torch.randn(10, 1, 28, 28)
    y_data = torch.randint(0, 3, (10,))
    loader = DataLoader(TensorDataset(x_data, y_data), batch_size=5)

    c0 = HospitalClient(client_id=0, dataloader=loader, model_type="vit_tiny", peft_mode="ffa_lora")
    c1 = HospitalClient(client_id=1, dataloader=loader, model_type="vit_tiny", peft_mode="ffa_lora")
    c2 = HospitalClient(client_id=2, dataloader=loader, model_type="vit_tiny", peft_mode="ffa_lora", malicious=True)

    fc0 = FlowerHospitalClient(c0)
    fc1 = FlowerHospitalClient(c1)
    fc2 = FlowerHospitalClient(c2)

    proxy0 = InProcessClientProxy("0", fc0)
    proxy1 = InProcessClientProxy("1", fc1)
    proxy2 = InProcessClientProxy("2", fc2)

    from flwr.common import FitIns, FitRes, ndarrays_to_parameters, parameters_to_ndarrays
    fit_ins = FitIns(ndarrays_to_parameters(strategy.arrays), {"current_round": 1, "algorithm": "FedAvg"})

    res0 = proxy0.fit(fit_ins)
    res1 = proxy1.fit(fit_ins)
    res2 = proxy2.fit(fit_ins)

    # Corrupt res2 parameters to simulate gradient scaling / poison attack
    arrays2 = parameters_to_ndarrays(res2.parameters)
    poisoned_arrays = [arr * 15.0 for arr in arrays2]
    res2_poisoned = FitRes(
        status=res2.status,
        parameters=ndarrays_to_parameters(poisoned_arrays),
        num_examples=res2.num_examples,
        metrics=res2.metrics,
    )

    agg_params, metrics = strategy.aggregate_fit(
        server_round=1,
        results=[(proxy0, res0), (proxy1, res1), (proxy2, res2_poisoned)],
        failures=[],
    )

    assert agg_params is not None
    assert metrics["participating_clients"] == 3
    assert metrics["blocked_clients_count"] == 1
    assert 2 in strategy.last_blocked_nodes
    assert 1 not in strategy.last_blocked_nodes  # Proves node 1 is not falsely accused
    assert metrics["blocked_nodes"] == "2"
    assert json.loads(str(metrics["blocked_nodes_json"])) == [2]

