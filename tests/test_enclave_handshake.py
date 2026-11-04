"""Tests for authentic cryptographic enclave attestation handshake (F-11)."""

import pytest
from fastapi.testclient import TestClient
from api.main import app, state, get_default_strategy_e_hospitals
from src.fed_xray.core.server import CentralServer


@pytest.fixture
def client():
    return TestClient(app)


def test_enclave_handshake_endpoint(client):
    """Verify GET /api/fl/enclave-handshake returns valid cryptographic certificate."""
    res = client.get("/api/fl/enclave-handshake")
    assert res.status_code == 200
    data = res.json()

    assert data["attestation_status"] == "verified"
    assert data["attestation_digest"].startswith("sha256:")
    assert len(data["attestation_digest"]) == 7 + 64
    assert data["is_hardware_tee"] is False
    assert "Software Cryptographic Simulation" in data["enclave_architecture"]

    # Verify server enclave
    server_enc = data["server_enclave"]
    assert server_enc["status"] == "healthy"
    assert "device" in server_enc
    assert "defense_mode" in server_enc
    assert server_enc["enclave_type"] == "software_cryptographic_simulation"
    assert server_enc["threshold_mpc_status"] == "in_process_prototype"

    # Verify 4 decentralized hospital enclaves
    nodes = data["nodes"]
    assert len(nodes) == 4
    for idx, node in enumerate(nodes):
        assert node["node_id"] == f"hospital_{idx}"
        assert node["enclave_status"] == "verified"
        assert node["peer_key_fingerprint"].startswith("ecdh:x25519:")
        assert node["scans_verified"] >= 1
        assert node["leakage_bytes"] == 0

    # Verify cryptographic suite
    crypto = data["cryptography"]
    assert "CKKS" in crypto["homomorphic_encryption"]
    assert "SecAgg+" in crypto["secagg_protocol"]
    assert "X25519" in crypto["key_exchange"]

    # Verify patient-level DP
    privacy = data["privacy"]
    assert privacy["patient_level_dp"] is True
    assert privacy["epsilon"] == 0.0
    assert privacy["target_epsilon"] == 2.0
    assert privacy["target_delta"] <= 1e-5
    assert privacy["budget_exhausted"] is False


def test_enclave_handshake_deterministic_identity_stability(client):
    """Verify that multiple enclave handshakes return stable node fingerprints and identical digests."""
    res1 = client.get("/api/fl/enclave-handshake")
    assert res1.status_code == 200
    data1 = res1.json()

    res2 = client.get("/api/fl/enclave-handshake")
    assert res2.status_code == 200
    data2 = res2.json()

    assert data1["attestation_digest"] == data2["attestation_digest"]
    assert len(data1["nodes"]) == len(data2["nodes"])

    for node1, node2 in zip(data1["nodes"], data2["nodes"]):
        assert node1["node_id"] == node2["node_id"]
        assert node1["peer_key_fingerprint"] == node2["peer_key_fingerprint"]


def test_strategy_e_hospitals_no_fake_multipliers():
    """Verify that hospital topology does not use arbitrary fake multipliers or hardcoded 1.42."""
    hospitals_zero = get_default_strategy_e_hospitals(active_epsilon=0.0)
    for h in hospitals_zero:
        assert h["dpEpsilon"] == 0.0

    hospitals_active = get_default_strategy_e_hospitals(active_epsilon=1.85)
    for h in hospitals_active:
        assert h["dpEpsilon"] == 1.85


def test_differential_privacy_budget_exhaustion_unclamped(client):
    """Verify that spent epsilon is accurately reported and not clamped to 2.0 when exhausted."""
    original_server = state.server
    try:
        # Create a real server with active DP
        server = CentralServer(privacy_noise=0.8, defense_mode=False, dp_mode="distributed")
        # Step accountant with 3 rounds so composed epsilon exceeds target (2.0)
        for _ in range(3):
            server.dp_accountant.step(q=0.5, sigma=0.8)
        state.server = server

        res = client.get("/api/fl/enclave-handshake")
        assert res.status_code == 200
        data = res.json()

        privacy = data["privacy"]
        # Epsilon must report the genuine > 2.0 spent, NOT clamped to 2.0
        assert privacy["epsilon"] > 2.0
        assert privacy["target_epsilon"] == 2.0
        assert privacy["budget_exhausted"] is True
        assert privacy["total_rounds_composed"] == 3
    finally:
        state.server = original_server
