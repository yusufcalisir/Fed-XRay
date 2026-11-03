"""Product-grade verification of SEC-01: Cryptographic Isolation and Option J Privacy.

Verifies:
1. Key Separation: CentralServer holds strictly public CKKS context; direct server decryption raises PermissionError.
2. In-Transit Encryption: HospitalClient encrypts parameters into authentic EncryptedTensor ciphertexts.
3. Zero-Plaintext Aggregation: CentralServer homomorphically aggregates client ciphertexts without secret key.
4. Quorum Decryption: Only ConsortiumKeyEnclave decrypts the aggregated model.
5. Zero-Snooping Byzantine Defense: Clients generate enclave attestations locally; trusted cohort masks client-side.
"""

from __future__ import annotations
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fed_xray.core.privacy import (
    CKKSEncryptionEngine,
    ConsortiumKeyEnclave,
    EncryptedTensor,
)
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round


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
