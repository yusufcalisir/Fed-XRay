"""Test authentic node-level differential privacy accounting and elimination of heuristic multipliers."""

import torch
from torch.utils.data import DataLoader, TensorDataset
from fastapi.testclient import TestClient

from api.main import app, state, get_default_strategy_e_hospitals
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer


def create_synthetic_client(client_id: int, num_samples: int, noise_multiplier: float = 0.8) -> HospitalClient:
    """Helper to instantiate a HospitalClient with a specific local sample count."""
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, 3, (num_samples,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=min(16, num_samples), shuffle=True)
    return HospitalClient(
        client_id=client_id,
        dataloader=loader,
        model_type="cnn",
        dp_enabled=True,
        dp_noise_multiplier=noise_multiplier,
        dp_clipping_bound=1.0,
        dp_mode="distributed",
    )


def test_client_initializes_with_zero_spent_epsilon():
    """Verify HospitalClient initializes with authentic unspent DP budget."""
    client = create_synthetic_client(client_id=1, num_samples=50)
    assert hasattr(client, "dp_accountant")
    assert client.get_spent_epsilon() == 0.0


def test_client_dp_accounting_reflects_local_sample_size():
    """Verify distinct sample counts produce mathematically distinct Renyi DP composed epsilons."""
    # Client A: Small local cohort (n=20) -> higher subsampling ratio q_A
    client_a = create_synthetic_client(client_id=1, num_samples=20, noise_multiplier=0.8)
    # Client B: Large local cohort (n=200) -> lower subsampling ratio q_B
    client_b = create_synthetic_client(client_id=2, num_samples=200, noise_multiplier=0.8)

    server = CentralServer(privacy_noise=0.8)
    weights = server.get_weights()

    # Train 2 rounds on both clients
    for round_idx in range(1, 3):
        client_a.train(global_weights=weights, current_round=round_idx, total_rounds=2)
        client_b.train(global_weights=weights, current_round=round_idx, total_rounds=2)

    eps_a = client_a.get_spent_epsilon()
    eps_b = client_b.get_spent_epsilon()

    assert eps_a > 0.0, "Client A must have accumulated positive privacy loss"
    assert eps_b > 0.0, "Client B must have accumulated positive privacy loss"
    # Small sample size has larger subsampling fraction q, hence higher privacy expenditure
    assert eps_a > eps_b, f"Expected eps_a ({eps_a}) > eps_b ({eps_b}) due to subsampling amplification"


def test_server_node_epsilon_registry():
    """Verify CentralServer accurately tracks authentic client epsilons without multipliers."""
    server = CentralServer()
    server.record_client_epsilon("node-1", 1.2345)
    server.record_client_epsilon("node-2", 0.9876)

    assert server.get_client_epsilon("node-1") == 1.2345
    assert server.get_client_epsilon("node-2") == 0.9876
    assert server.get_client_epsilon("node-3") == 0.0

    all_eps = server.get_all_client_epsilons()
    assert all_eps == {"node-1": 1.2345, "node-2": 0.9876}


def test_state_reports_authentic_node_epsilons(monkeypatch):
    """Verify /api/state reports authentic node-specific epsilons without fake multipliers."""
    test_client = TestClient(app)

    # 1. Unqueried state: All nodes report 0.0
    monkeypatch.setattr(state, "node_epsilons", {})
    monkeypatch.setattr(state, "server", None)
    monkeypatch.setattr(state, "training_history", {})
    monkeypatch.setattr(state.store, "get_training_history", lambda: {})
    res = test_client.get("/api/state")
    assert res.status_code == 200
    data = res.json()
    hospitals = data.get("hospitals", [])
    assert len(hospitals) >= 4
    for h in hospitals:
        assert h["dpEpsilon"] == 0.0

    # 2. Populated authentic node epsilons from client accountants
    authentic_epsilons = {
        "node-1": 1.4521,
        "node-2": 0.8934,
        "node-3": 1.7210,
        "node-4": 1.1042,
    }
    monkeypatch.setattr(state, "node_epsilons", authentic_epsilons)
    res2 = test_client.get("/api/state")
    assert res2.status_code == 200
    data2 = res2.json()
    hospitals2 = {h["id"]: h["dpEpsilon"] for h in data2.get("hospitals", [])}

    assert hospitals2["node-1"] == 1.4521
    assert hospitals2["node-2"] == 0.8934
    assert hospitals2["node-3"] == 1.7210
    assert hospitals2["node-4"] == 1.1042


def test_topology_helper_eliminates_arbitrary_multipliers():
    """Verify get_default_strategy_e_hospitals accepts and applies authentic node epsilons."""
    # When node_epsilons provided, exact values must be preserved
    custom_eps = {"node-1": 0.5, "node-2": 0.75, "node-3": 1.2, "node-4": 0.9}
    nodes = get_default_strategy_e_hospitals(active_epsilon=1.0, node_epsilons=custom_eps)
    for n in nodes:
        assert n["dpEpsilon"] == custom_eps[n["id"]]

    # When active_epsilon is 0.0 and no node_epsilons, all must be 0.0
    nodes_zero = get_default_strategy_e_hospitals(active_epsilon=0.0)
    for n in nodes_zero:
        assert n["dpEpsilon"] == 0.0
