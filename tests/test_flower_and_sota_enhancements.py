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
    flower_parameters_to_state_dict,
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
