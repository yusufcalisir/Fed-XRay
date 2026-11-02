"""Tests for authentic cryptographic enclave attestation handshake (F-11)."""

import pytest
from fastapi.testclient import TestClient
from api.main import app


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

    # Verify server enclave
    server_enc = data["server_enclave"]
    assert server_enc["status"] == "healthy"
    assert "device" in server_enc
    assert "defense_mode" in server_enc

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
    assert privacy["epsilon"] <= 2.0
    assert privacy["target_delta"] <= 1e-5
