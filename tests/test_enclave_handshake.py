"""Tests for authentic cryptographic enclave attestation handshake (F-11)."""

import pytest
from fastapi.testclient import TestClient
from api.main import app, state, get_default_strategy_e_hospitals
from src.fed_xray.core.server import CentralServer

try:
    import tenseal  # noqa: F401
    HAS_TENSEAL = True
except ImportError:
    HAS_TENSEAL = False


@pytest.fixture
def client():
    state.server = None
    state.dataloaders = None
    state.hospital_cohort_info = []
    state.ensure_cohort_loaded()
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


def test_differential_privacy_intermediate_budget_accumulation(client):
    """Verify that 1 round of DP accumulation reports genuine intermediate epsilon (0.0 < eps < 2.0)."""
    original_server = state.server
    try:
        server = CentralServer(privacy_noise=1.5, defense_mode=False, dp_mode="distributed")
        server.dp_accountant.step(q=0.1, sigma=1.5)
        state.server = server

        res = client.get("/api/fl/enclave-handshake")
        assert res.status_code == 200
        data = res.json()

        privacy = data["privacy"]
        assert 0.0 < privacy["epsilon"] < 2.0
        assert privacy["target_epsilon"] == 2.0
        assert privacy["budget_exhausted"] is False
        assert privacy["total_rounds_composed"] == 1
    finally:
        state.server = original_server


@pytest.mark.skipif(not HAS_TENSEAL, reason="tenseal optional dependency not installed")
def test_enclave_handshake_with_threshold_committee(client):
    """Verify handshake reports process-isolated threshold enclave when threshold committee is active."""
    original_comm = state.threshold_committee
    original_enclaves = dict(state.hospital_threshold_enclaves)
    try:
        comm, enclaves = state.get_or_create_threshold_committee(num_hospitals=4)
        res = client.get("/api/fl/enclave-handshake")
        assert res.status_code == 200
        data = res.json()

        assert "Option J Isolated Process Threshold Enclave" in data["enclave_architecture"]
        server_enc = data["server_enclave"]
        assert server_enc["enclave_type"] == "process_isolated_threshold_enclave"
        assert server_enc["threshold_mpc_status"] == "active_quorum_enforced"
    finally:
        state.threshold_committee = original_comm
        state.hospital_threshold_enclaves = original_enclaves


def test_dynamic_peft_subspace_reporting(client):
    """Verify that client nodes and server enclave dynamically reflect active PEFT modes and architectures."""
    orig_peft = state.peft_mode
    orig_model = state.model_type
    orig_lora_r = state.lora_r
    orig_server = state.server
    try:
        # 1. FedSA-LoRA with rank 8
        state.server = None
        state.peft_mode = "fedsa_lora"
        state.model_type = "vit_tiny"
        state.lora_r = 8
        res = client.get("/api/fl/enclave-handshake")
        assert res.status_code == 200
        data = res.json()
        assert "FedSA-LoRA (Rank-8 Shared-A / Local Personalized-B)" in data["nodes"][0]["peft_subspace"]
        assert "ViT-Tiny (128-dim / FedSA-LoRA" in data["server_enclave"]["foundation_model"]

        # 2. LoRA with rank 32
        state.peft_mode = "lora"
        state.model_type = "vit_base"
        state.lora_r = 32
        res = client.get("/api/fl/enclave-handshake")
        assert res.status_code == 200
        data = res.json()
        assert "LoRA (Rank-32 Trainable Low-Rank Adapters)" in data["nodes"][0]["peft_subspace"]
        assert "ViT-Base-224 (Pretrained ImageNet-1K / LoRA" in data["server_enclave"]["foundation_model"]

        # 3. Full convolutional architecture (CNN baseline without PEFT adapters)
        state.peft_mode = None
        state.model_type = "cnn"
        res = client.get("/api/fl/enclave-handshake")
        assert res.status_code == 200
        data = res.json()
        assert "Full Convolutional Weights" in data["nodes"][0]["peft_subspace"]
        assert data["server_enclave"]["foundation_model"] == "Custom 3-Layer Medical CNN Classifier (Radiological Feature Extractor)"

        # 4. CentralServer priority over AppState defaults
        server = CentralServer(model_type="vit_small", peft_mode="ffa_lora", lora_r=16, defense_mode=False)
        state.server = server
        res = client.get("/api/fl/enclave-handshake")
        assert res.status_code == 200
        data = res.json()
        assert "FFA-LoRA (Rank-16 Globally Frozen-A / Local-B)" in data["nodes"][0]["peft_subspace"]
        assert "ViT-Small-224 (384-dim / FFA-LoRA" in data["server_enclave"]["foundation_model"]
    finally:
        state.peft_mode = orig_peft
        state.model_type = orig_model
        state.lora_r = orig_lora_r
        state.server = orig_server


def test_cryptography_suite_active_vs_standby_states(client):
    """Verify that CryptographySuiteInfo accurately reports active vs standby status for CKKS and SecAgg+."""
    orig_server = state.server
    orig_ckks = state.use_ckks
    orig_secagg = state.secure_aggregation
    try:
        # 1. Plaintext mode (neither CKKS nor SecAgg+ active)
        state.server = None
        state.use_ckks = False
        state.secure_aggregation = False
        res = client.get("/api/fl/enclave-handshake")
        assert res.status_code == 200
        crypto = res.json()["cryptography"]
        assert crypto["ckks_active"] is False
        assert crypto["secagg_active"] is False
        assert "Standby" in crypto["homomorphic_encryption"] or "Inactive" in crypto["homomorphic_encryption"]
        assert "Standby" in crypto["secagg_protocol"] or "Inactive" in crypto["secagg_protocol"]
        assert "CKKS" in crypto["homomorphic_encryption"]
        assert "SecAgg+" in crypto["secagg_protocol"]

        # 2. CKKS active, SecAgg+ standby
        state.use_ckks = True
        state.secure_aggregation = False
        res = client.get("/api/fl/enclave-handshake")
        assert res.status_code == 200
        crypto = res.json()["cryptography"]
        assert crypto["ckks_active"] is True
        assert crypto["secagg_active"] is False
        assert "Active" in crypto["homomorphic_encryption"]
        assert "CKKS" in crypto["homomorphic_encryption"]

        # 3. SecAgg+ active, CKKS standby
        state.use_ckks = False
        state.secure_aggregation = True
        res = client.get("/api/fl/enclave-handshake")
        assert res.status_code == 200
        crypto = res.json()["cryptography"]
        assert crypto["ckks_active"] is False
        assert crypto["secagg_active"] is True
        assert "Active" in crypto["secagg_protocol"]
        assert "SecAgg+" in crypto["secagg_protocol"]
    finally:
        state.server = orig_server
        state.use_ckks = orig_ckks
        state.secure_aggregation = orig_secagg


def test_leakage_bytes_parity_under_plaintext_vs_cryptographic_protection(client):
    """Verify unmasked leakage byte footprint under plaintext and zero-leakage invariant under CKKS/SecAgg+."""
    orig_server = state.server
    orig_ckks = state.use_ckks
    orig_secagg = state.secure_aggregation
    try:
        # 1. Plaintext server with unmasked uplink payload
        server = CentralServer(defense_mode=False, use_ckks=False, secure_aggregation=False)
        server.last_uplink_bytes_per_client = 65432
        state.server = server
        state.use_ckks = False
        state.secure_aggregation = False

        res = client.get("/api/fl/enclave-handshake")
        assert res.status_code == 200
        nodes = res.json()["nodes"]
        # Under plaintext, unmasked update is visible to coordinator
        for node in nodes:
            assert node["leakage_bytes"] == 65432

        # 2. CKKS encryption active -> zero leakage to central server
        server.use_ckks = True
        state.use_ckks = True
        res = client.get("/api/fl/enclave-handshake")
        assert res.status_code == 200
        nodes = res.json()["nodes"]
        for node in nodes:
            assert node["leakage_bytes"] == 0

        # 3. SecAgg+ masking active -> zero leakage to central server
        server.use_ckks = False
        state.use_ckks = False
        server.secure_aggregation = True
        state.secure_aggregation = True
        res = client.get("/api/fl/enclave-handshake")
        assert res.status_code == 200
        nodes = res.json()["nodes"]
        for node in nodes:
            assert node["leakage_bytes"] == 0
    finally:
        state.server = orig_server
        state.use_ckks = orig_ckks
        state.secure_aggregation = orig_secagg


