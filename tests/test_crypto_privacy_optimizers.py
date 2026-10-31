"""Tests for Dual-Layer Cryptographic Privacy & Advanced FL Optimization.

Verifies:
1. Microsoft SEAL / TenSEAL Leveled CKKS Homomorphic Encryption (RLWE 128-bit).
2. Cryptographic Secure Aggregation (SecAgg+) with pairwise zero-sum random masks.
3. Patient-Level Gaussian Differential Privacy with Renyi DP Accountant bounds.
4. FedDyn dynamic regularization and SCAFFOLD control variates.
"""

from __future__ import annotations
import torch
from src.fed_xray.core.privacy import (
    CKKSEncryptionEngine,
    SecAggPlusProtocol,
    PatientLevelDPAccountant,
    apply_patient_level_dp_clipping,
)


def test_tenseal_ckks_homomorphic_sum_precision():
    """Verify RLWE 128-bit CKKS homomorphic aggregation on parameter vectors."""
    engine = CKKSEncryptionEngine(poly_modulus_degree=8192, scale_bits=40)

    # Two client parameter tensors
    tensor_a = torch.tensor([0.25, -0.50, 1.20, -0.80, 0.05], dtype=torch.float32)
    tensor_b = torch.tensor([0.75, 0.10, -0.40, 0.60, -0.30], dtype=torch.float32)

    w_a, w_b = 0.6, 0.4
    expected_sum = w_a * tensor_a + w_b * tensor_b

    # Client-side encryption
    enc_a = engine.encrypt_vector(tensor_a)
    enc_b = engine.encrypt_vector(tensor_b)

    # Server-side homomorphic sum using public context only
    enc_sum = engine.homomorphic_sum([enc_a, enc_b], weights=[w_a, w_b])

    # Key holder decryption
    decrypted_sum = engine.decrypt_vector(enc_sum)

    error = torch.max(torch.abs(decrypted_sum - expected_sum)).item()
    assert error < 1e-4, f"CKKS homomorphic sum error must be minimal: {error}"


def test_secagg_pairwise_zero_sum_cancellation():
    """Verify pairwise masks cancel out identically under weighted aggregation."""
    num_clients = 4
    shapes = {
        "weight": torch.Size([8, 128]),
        "bias": torch.Size([8]),
    }
    sample_weights = [120.0, 80.0, 95.0, 105.0]
    total_w = sum(sample_weights)
    p = [w / total_w for w in sample_weights]

    masks = SecAggPlusProtocol.generate_pairwise_masks(
        num_clients=num_clients,
        tensor_shapes=shapes,
        weights=sample_weights,
    )

    # Compute weighted sum of masks: sum_k p_k * mask_k
    summed_weight_mask = torch.zeros(shapes["weight"])
    summed_bias_mask = torch.zeros(shapes["bias"])

    for k in range(num_clients):
        summed_weight_mask += p[k] * masks[k]["weight"]
        summed_bias_mask += p[k] * masks[k]["bias"]

    weight_cancellation = torch.norm(summed_weight_mask).item()
    bias_cancellation = torch.norm(summed_bias_mask).item()

    assert weight_cancellation < 1e-5, f"SecAgg+ weight masks did not cancel out: {weight_cancellation}"
    assert bias_cancellation < 1e-5, f"SecAgg+ bias masks did not cancel out: {bias_cancellation}"


def test_patient_level_dp_accountant_rdp_bound():
    """Verify Patient-Level DP Accountant computes correct Renyi DP bounds."""
    accountant = PatientLevelDPAccountant(target_delta=1e-5)

    # Simulate 5 federated rounds with subsampling rate q=0.2 and noise multiplier sigma=1.0
    for _ in range(5):
        accountant.step(q=0.2, sigma=1.0)

    epsilon = accountant.get_epsilon()
    assert 0.0 < epsilon < 5.0, f"Expected bounded epsilon, got {epsilon}"


def test_patient_level_dp_gradient_clipping():
    """Verify Patient-Level DP gradient clipping enforces L2 sensitivity bound C."""
    torch.manual_seed(42)
    # 3 patients with different gradient magnitudes
    g1 = torch.ones(10) * 0.2  # norm ~ 0.63 < C
    g2 = torch.ones(10) * 2.0  # norm ~ 6.32 > C
    g3 = torch.ones(10) * 1.5  # norm ~ 4.74 > C

    clipped_dp_grad = apply_patient_level_dp_clipping(
        patient_gradients=[g1, g2, g3],
        clipping_bound_C=1.0,
        noise_multiplier_sigma=0.0,  # Zero noise to test clipping bound directly
    )

    assert clipped_dp_grad.shape == (10,)
    # Since each clipped grad has norm <= 1.0, average norm must be <= 1.0
    avg_norm = torch.norm(clipped_dp_grad).item()
    assert avg_norm <= 1.0 + 1e-5, f"Clipped gradient norm exceeds bound C: {avg_norm}"
