"""Consortium Cryptography and Zero-Plaintext Aggregation Verification Suite.

Validates Option J dual-layer privacy: client-side CKKS Homomorphic Encryption,
Consortium Key Enclave public/secret key separation, and Byzantine defense via enclave attestations.
"""

from __future__ import annotations
import os
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fed_xray.core.privacy import (
    CKKSEncryptionEngine,
    ConsortiumKeyEnclave,
    ConsortiumThresholdCommittee,
    EncryptedTensor,
    IsolatedProcessEnclave,
    ThresholdQuorumNotMetError,
)
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round

try:
    import tenseal  # noqa: F401
    HAS_TENSEAL = True
except ImportError:
    HAS_TENSEAL = False

pytestmark = pytest.mark.skipif(not HAS_TENSEAL, reason='tenseal optional dependency not installed')


def test_ckks_public_context_key_separation():
    """Verify CentralServer cannot decrypt individual client ciphertexts."""
    enclave = ConsortiumKeyEnclave(scale_bits=40)
    pub_bytes = enclave.public_context_bytes

    server_engine = CKKSEncryptionEngine.from_public_bytes(pub_bytes, scale_bits=40)
    assert server_engine.is_public_only is True

    # Client encrypts parameter tensor
    client_tensor = torch.tensor([0.25, -0.75, 1.5, 3.0])
    ciphertext = server_engine.encrypt_vector(client_tensor)
    assert isinstance(ciphertext, EncryptedTensor)
    assert ciphertext.nbytes > 0

    # Server attempts direct decryption: must raise PermissionError under Option J standard
    with pytest.raises(PermissionError) as excinfo:
        server_engine.decrypt_vector(ciphertext)
    assert "cannot decrypt ciphertexts" in str(excinfo.value)

    # Consortium Key Enclave successfully decrypts
    decrypted = enclave.engine.decrypt_vector(ciphertext)
    assert torch.allclose(decrypted, client_tensor, atol=1e-2)


def test_client_side_ckks_encryption_and_zero_plaintext_aggregation():
    """Verify hospital clients transmit ciphertexts and server aggregates with zero plaintext observation."""
    enclave = ConsortiumKeyEnclave(scale_bits=40)
    pub_bytes = enclave.public_context_bytes

    # Two hospital clients with local data
    dataset0 = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 3, (10,)))
    dataset1 = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 3, (10,)))
    loader0 = DataLoader(dataset0, batch_size=5)
    loader1 = DataLoader(dataset1, batch_size=5)

    c0 = HospitalClient(0, loader0, model_type="cnn")
    c1 = HospitalClient(1, loader1, model_type="cnn")

    # Clients receive public context
    c0.init_ckks(pub_bytes)
    c1.init_ckks(pub_bytes)

    # Clients encrypt their parameters locally
    enc_w0 = c0.encrypt_local_weights()
    enc_w1 = c1.encrypt_local_weights()

    # Verify wire payload is strictly EncryptedTensor
    assert all(isinstance(v, EncryptedTensor) for v in enc_w0.values())
    assert all(isinstance(v, EncryptedTensor) for v in enc_w1.values())

    # CentralServer initialized with public context only
    server = CentralServer(model_type="cnn", use_ckks=True)
    assert server.ckks_engine is not None
    assert server.ckks_engine.is_public_only is True

    # Server aggregates ciphertexts
    agg_weights = server.aggregate(
        client_weights=[enc_w0, enc_w1],
        sample_counts=[100, 100],
        use_ckks=True,
    )

    # Global weights reconstructed correctly
    assert isinstance(agg_weights, dict)
    assert len(agg_weights) > 0
    for k in agg_weights.keys():
        assert isinstance(agg_weights[k], torch.Tensor)


def test_byzantine_defense_with_client_attestation_and_secagg():
    """Verify Byzantine defense works without server-side plaintext snooping."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    server = CentralServer(
        device=device,
        defense_mode=True,
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        secure_aggregation=True,
    )

    dataset_good = TensorDataset(torch.randn(15, 1, 28, 28), torch.zeros(15, dtype=torch.long))
    dataset_bad = TensorDataset(torch.randn(15, 1, 28, 28), torch.randint(0, 3, (15,)))

    c0 = HospitalClient(0, DataLoader(dataset_good, batch_size=5), device=device, model_type="vit_tiny", peft_mode="ffa_lora")
    c1 = HospitalClient(1, DataLoader(dataset_good, batch_size=5), device=device, model_type="vit_tiny", peft_mode="ffa_lora")
    c2 = HospitalClient(2, DataLoader(dataset_bad, batch_size=5), device=device, malicious=True, model_type="vit_tiny", peft_mode="ffa_lora")

    test_images = torch.randn(10, 1, 28, 28)
    test_labels = torch.zeros(10, dtype=torch.long)

    # Execute round under full defense and SecAgg+
    agg_metrics, client_metrics, test_metrics, sec_report = run_federated_round(
        server=server,
        clients=[c0, c1, c2],
        round_num=1,
        total_rounds=1,
        test_images=test_images,
        test_labels=test_labels,
        use_defense=True,
        secure_aggregation=True,
        use_ckks=False,
    )

    assert sec_report is not None
    assert sec_report.defense_active is True
    assert sec_report.secagg_active is True
    assert 2 in sec_report.clients_blocked
    assert 0 in sec_report.clients_accepted
    assert 1 in sec_report.clients_accepted
    assert "SecAgg+ zero-sum masking" in sec_report.details


def test_threshold_ckks_secret_sharing_and_quorum_enforcement():
    """Verify Option J (t, n) threshold secret sharing requires quorum to decrypt."""
    committee = ConsortiumThresholdCommittee(
        num_participants=4,
        threshold=3,
        scale_bits=40,
    )
    pub_bytes = committee.public_context_bytes

    # Server possesses strictly public context
    server_engine = CKKSEncryptionEngine.from_public_bytes(pub_bytes, scale_bits=40)
    assert server_engine.is_public_only is True

    # Parameter update
    test_tensor = torch.tensor([0.15, -0.45, 1.20, 2.50])
    ciphertext = server_engine.encrypt_vector(test_tensor)
    enc_dict = {"layer.weight": ciphertext}

    # Digest
    digest = committee.compute_ciphertext_digest(enc_dict)

    # Participant enclaves
    e0 = committee.get_hospital_enclave(0)
    e1 = committee.get_hospital_enclave(1)
    e2 = committee.get_hospital_enclave(2)
    e3 = committee.get_hospital_enclave(3)

    # 1. Quorum failure: only 2 authorizations presented when threshold is 3
    auth0 = e0.sign_partial_decryption_authorization(round_num=1, ciphertext_digest=digest)
    auth1 = e1.sign_partial_decryption_authorization(round_num=1, ciphertext_digest=digest)
    with pytest.raises(ThresholdQuorumNotMetError) as excinfo:
        committee.collaborative_decrypt(enc_dict, [auth0, auth1])
    assert "received 2 valid hospital shares, required 3" in str(excinfo.value)

    # 2. Quorum failure: duplicate authorization from participant 0
    with pytest.raises(ThresholdQuorumNotMetError):
        committee.collaborative_decrypt(enc_dict, [auth0, auth0])

    # 3. Quorum success: 3 distinct authorizations (participants 0, 1, 3)
    auth3 = e3.sign_partial_decryption_authorization(round_num=1, ciphertext_digest=digest)
    decrypted = committee.collaborative_decrypt(enc_dict, [auth0, auth1, auth3])
    assert "layer.weight" in decrypted
    assert torch.allclose(decrypted["layer.weight"], test_tensor, atol=1e-2)

    # 4. Quorum success: all 4 authorizations (participants 0, 1, 2, 3)
    auth2 = e2.sign_partial_decryption_authorization(round_num=1, ciphertext_digest=digest)
    decrypted_full = committee.collaborative_decrypt(enc_dict, [auth0, auth1, auth2, auth3])
    assert torch.allclose(decrypted_full["layer.weight"], test_tensor, atol=1e-2)


def test_isolated_process_enclave_memory_space_separation():
    """Verify Option J isolated OS process enclave operates in distinct address space."""
    with IsolatedProcessEnclave(num_participants=3, threshold=2, scale_bits=40) as isolated:
        # Verify genuine OS child process isolation
        assert isolated.worker_pid is not None
        assert isolated.worker_pid != os.getpid()

        # Server holds only public context
        server = CentralServer(
            model_type="cnn",
            use_ckks=True,
            threshold_committee=isolated,
        )
        assert server.ckks_engine is not None
        assert server.ckks_engine.is_public_only is True

        # Clients encrypt parameters with public context only
        client_tensor = torch.tensor([0.5, -1.0, 1.5])
        c_dict0 = {"test.weight": server.ckks_engine.encrypt_vector(client_tensor)}
        c_dict1 = {"test.weight": server.ckks_engine.encrypt_vector(client_tensor)}

        # Homomorphic aggregation in ciphertext domain
        agg_encrypted = server.ckks_engine.homomorphic_aggregate_state_dicts([c_dict0, c_dict1], [50.0, 50.0])

        # Decrypt via isolated OS process over IPC pipe
        decrypted = isolated.auto_collaborative_decrypt(agg_encrypted)
        assert "test.weight" in decrypted
        assert torch.allclose(decrypted["test.weight"], client_tensor, atol=1e-2)


def test_federated_round_with_threshold_committee_and_byzantine_isolation():
    """Verify Byzantine poisoned node is quarantined and honest threshold quorum decrypts."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    committee = ConsortiumThresholdCommittee(num_participants=3, threshold=2, scale_bits=40)
    server = CentralServer(
        device=device,
        defense_mode=True,
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        use_ckks=True,
        threshold_committee=committee,
    )
    assert server.ckks_engine is not None
    assert server.ckks_engine.is_public_only is True

    # Hospital clients with their individual threshold enclaves
    dataset_good = TensorDataset(torch.randn(15, 1, 28, 28), torch.zeros(15, dtype=torch.long))
    dataset_bad = TensorDataset(torch.randn(15, 1, 28, 28), torch.randint(0, 3, (15,)))

    c0 = HospitalClient(
        0, DataLoader(dataset_good, batch_size=5),
        device=device, model_type="vit_tiny", peft_mode="ffa_lora",
        threshold_enclave=committee.get_hospital_enclave(0),
    )
    c1 = HospitalClient(
        1, DataLoader(dataset_good, batch_size=5),
        device=device, model_type="vit_tiny", peft_mode="ffa_lora",
        threshold_enclave=committee.get_hospital_enclave(1),
    )
    c2 = HospitalClient(
        2, DataLoader(dataset_bad, batch_size=5),
        device=device, malicious=True, model_type="vit_tiny", peft_mode="ffa_lora",
        threshold_enclave=committee.get_hospital_enclave(2),
    )

    test_images = torch.randn(10, 1, 28, 28)
    test_labels = torch.zeros(10, dtype=torch.long)

    # Execute round under full defense and CKKS threshold decryption
    agg_metrics, client_metrics, test_metrics, sec_report = run_federated_round(
        server=server,
        clients=[c0, c1, c2],
        round_num=1,
        total_rounds=1,
        test_images=test_images,
        test_labels=test_labels,
        use_defense=True,
        secure_aggregation=True,
        use_ckks=True,
        threshold_committee=committee,
    )

    assert sec_report is not None
    assert sec_report.defense_active is True
    # Malicious node 2 is quarantined
    assert 2 in sec_report.clients_blocked
    # Honest nodes 0 and 1 authorized decryption (quorum 2 >= 2 met)
    assert 0 in sec_report.clients_accepted
    assert 1 in sec_report.clients_accepted
    assert agg_metrics is not None
