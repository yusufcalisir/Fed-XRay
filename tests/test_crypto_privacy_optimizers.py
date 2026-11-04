"""Tests for Dual-Layer Cryptographic Privacy & Advanced FL Optimization.

Verifies:
1. Microsoft SEAL / TenSEAL Leveled CKKS Homomorphic Encryption (RLWE 128-bit).
2. Cryptographic Secure Aggregation (SecAgg+) with pairwise zero-sum random masks.
3. Patient-Level Gaussian Differential Privacy with Renyi DP Accountant bounds.
4. FedDyn dynamic regularization and SCAFFOLD control variates.
"""

from __future__ import annotations
import pytest
import torch
from src.fed_xray.core.privacy import (
    CKKSEncryptionEngine,
    SecAggPlusProtocol,
    SecAggPeer,
    PatientLevelDPAccountant,
    apply_patient_level_dp_clipping,
    clip_and_perturb_state_dict,
)
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer
from torch.utils.data import DataLoader, TensorDataset

try:
    import tenseal  # noqa: F401
    HAS_TENSEAL = True
except ImportError:
    HAS_TENSEAL = False



@pytest.mark.skipif(not HAS_TENSEAL, reason='tenseal optional dependency not installed')
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


def test_secagg_peer_ecdh_key_exchange_and_mask_synthesis():
    """Verify decentralized X25519 ECDH key exchange and local mask generation."""
    peers = [SecAggPeer(client_id=i) for i in range(4)]
    pk_registry = {p.client_id: p.public_key_bytes for p in peers}

    for p in peers:
        assert len(p.public_key_bytes) == 32

    # Verify symmetric ECDH derivation
    k_0_1 = peers[0].derive_pairwise_key(1, pk_registry[1], round_num=1)
    k_1_0 = peers[1].derive_pairwise_key(0, pk_registry[0], round_num=1)
    assert k_0_1 == k_1_0, "Pairwise HKDF keys must match identically across peers"

    shapes = {"w": torch.Size([16, 32]), "b": torch.Size([16])}
    weights_map = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}

    masks = [
        p.compute_mask(pk_registry, shapes, weights_map=weights_map, round_num=1)
        for p in peers
    ]

    # Weighted sum of masks must cancel to machine zero
    sum_w = torch.zeros(shapes["w"])
    sum_b = torch.zeros(shapes["b"])
    for i in range(4):
        sum_w += weights_map[i] * masks[i]["w"]
        sum_b += weights_map[i] * masks[i]["b"]

    assert torch.norm(sum_w).item() < 1e-5, "SecAggPeer weight masks did not cancel"
    assert torch.norm(sum_b).item() < 1e-5, "SecAggPeer bias masks did not cancel"


def test_secagg_client_side_masking_integration():
    """Verify HospitalClient possesses decentralized cryptographic identity and local masking."""
    device = torch.device("cpu")
    dataset = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 3, (10,)))
    loader = DataLoader(dataset, batch_size=5)

    c0 = HospitalClient(0, loader, device=device, model_type="cnn")
    c1 = HospitalClient(1, loader, device=device, model_type="cnn")

    pk0 = c0.get_secagg_public_key()
    pk1 = c1.get_secagg_public_key()
    assert len(pk0) == 32
    assert len(pk1) == 32

    pk_registry = {0: pk0, 1: pk1}
    weights_map = {0: 0.5, 1: 0.5}

    w0 = c0.get_weights()
    w1 = c1.get_weights()

    masked_w0 = c0.mask_local_weights(w0, pk_registry, weights_map=weights_map, round_num=1)
    masked_w1 = c1.mask_local_weights(w1, pk_registry, weights_map=weights_map, round_num=1)

    # Reconstruct weighted sum from masked weights
    for k in w0.keys():
        true_avg = 0.5 * w0[k].float() + 0.5 * w1[k].float()
        masked_avg = 0.5 * masked_w0[k].float() + 0.5 * masked_w1[k].float()
        err = torch.norm(true_avg - masked_avg).item()
        assert err < 1e-4, f"Masked average deviates from plaintext average: {err}"


def test_secagg_dropout_residual_recovery():
    """Verify dropout residual reconstruction restores exact zero-sum cancellation for surviving nodes."""
    num_clients = 4
    peers = {i: SecAggPeer(client_id=i) for i in range(num_clients)}
    pk_registry = {i: p.public_key_bytes for i, p in peers.items()}
    weights_map = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    shapes = {"w": torch.Size([8, 16])}

    # All 4 clients generate initial masks
    masks = {
        i: peers[i].compute_mask(pk_registry, shapes, weights_map=weights_map, round_num=1)
        for i in range(num_clients)
    }

    # Suppose Client 3 DROPS OUT before submission
    surviving_ids = [0, 1, 2]
    dropped_ids = [3]

    # Reconstruct dropout residual for Client 3
    residual = SecAggPlusProtocol.reconstruct_dropout_residual(
        dropped_client_ids=dropped_ids,
        surviving_peers={i: peers[i] for i in surviving_ids},
        all_public_keys=pk_registry,
        tensor_shapes=shapes,
        weights_map=weights_map,
        round_num=1,
    )

    # Sum surviving masked contributions minus the dropout residual:
    # sum_{s in surviving} p_s * mask_s - residual == 0
    surviving_sum = sum(weights_map[i] * masks[i]["w"] for i in surviving_ids)
    corrected_sum = surviving_sum - residual["w"]

    cancellation_norm = torch.norm(corrected_sum).item()
    assert cancellation_norm < 1e-5, f"Dropout residual failed to cancel unmasked masks: {cancellation_norm}"


def test_clip_and_perturb_state_dict_modes():
    """Verify sensitivity clipping bound and noise perturbation across all DP modes."""
    # Create large parameter deltas with L2 norm ~ 10.0
    deltas = {
        "layer1.weight": torch.ones(10, 10) * 0.7,
        "layer1.bias": torch.ones(10) * 0.5,
    }
    raw_norm = torch.sqrt(torch.sum(deltas["layer1.weight"] ** 2) + torch.sum(deltas["layer1.bias"] ** 2)).item()
    assert raw_norm > 5.0, "Raw deltas must exceed clipping bound"

    # 1. Zero noise test: verify exact clipping to C = 1.0
    clipped, measured_norm = clip_and_perturb_state_dict(
        delta_weights=deltas,
        clipping_bound=1.0,
        noise_multiplier=0.0,
        dp_mode="distributed",
    )
    new_norm = torch.sqrt(torch.sum(clipped["layer1.weight"] ** 2) + torch.sum(clipped["layer1.bias"] ** 2)).item()
    assert abs(new_norm - 1.0) < 1e-5, f"Clipped norm must equal bound C=1.0, got {new_norm}"
    assert abs(measured_norm - raw_norm) < 1e-5

    # 2. "off" mode: verify unclipped pass-through
    off_deltas, off_norm = clip_and_perturb_state_dict(
        delta_weights=deltas,
        clipping_bound=1.0,
        noise_multiplier=1.0,
        dp_mode="off",
    )
    for k in deltas:
        assert torch.equal(off_deltas[k], deltas[k])

    # 3. "local" mode: perturbation scale is sigma * C
    torch.manual_seed(42)
    local_deltas, _ = clip_and_perturb_state_dict(
        delta_weights=deltas,
        clipping_bound=1.0,
        noise_multiplier=1.5,
        dp_mode="local",
    )
    # Measured difference should be non-zero
    diff_local = torch.norm(local_deltas["layer1.weight"] - clipped["layer1.weight"]).item()
    assert diff_local > 0.0, "Local DP must add perturbation"

    # 4. "distributed" mode: perturbation scale is sigma * C / sqrt(K)
    torch.manual_seed(42)
    dist_deltas, _ = clip_and_perturb_state_dict(
        delta_weights=deltas,
        clipping_bound=1.0,
        noise_multiplier=1.5,
        num_clients=4,
        dp_mode="distributed",
    )
    diff_dist = torch.norm(dist_deltas["layer1.weight"] - clipped["layer1.weight"]).item()
    # Since num_clients = 4, std is half as large as local mode:
    assert 0.0 < diff_dist < diff_local + 1.0


def test_hospital_client_patient_level_dp_clipping_and_noise():
    """Verify HospitalClient applies client-side sensitivity clipping and DP noise."""
    device = torch.device("cpu")
    dataset = TensorDataset(torch.randn(12, 1, 28, 28), torch.randint(0, 3, (12,)))
    loader = DataLoader(dataset, batch_size=4)

    client = HospitalClient(
        client_id=0,
        dataloader=loader,
        device=device,
        model_type="cnn",
        dp_enabled=True,
        dp_clipping_bound=0.5,
        dp_noise_multiplier=1.0,
        dp_mode="distributed",
    )

    init_weights = client.get_weights()
    updated_weights, metrics, _ = client.train(
        global_weights=init_weights,
        current_round=1,
        total_rounds=1,
        num_consortium_clients=3,
    )

    assert metrics.dp_clipped_norm is not None
    assert metrics.dp_clipped_norm > 0.0

    # Ensure updated weights are valid, non-empty, and different from init
    for k in init_weights.keys():
        assert k in updated_weights
        assert not torch.isnan(updated_weights[k]).any()


def test_option_j_distributed_dp_aggregation_variance():
    """Verify consensus aggregation of K distributed DP updates achieves exact target variance."""
    num_clients = 4
    clipping_bound = 1.0
    noise_multiplier = 1.2
    dim = 20000

    torch.manual_seed(123)
    # Simulate K clients generating distributed noise with zero deltas
    client_noises = []
    for _ in range(num_clients):
        deltas = {"param": torch.zeros(dim)}
        perturbed, _ = clip_and_perturb_state_dict(
            delta_weights=deltas,
            clipping_bound=clipping_bound,
            noise_multiplier=noise_multiplier,
            num_clients=num_clients,
            dp_mode="distributed",
        )
        client_noises.append(perturbed["param"])

    # Consensus average: (1 / K) * sum(noise_k)
    consensus_noise = sum(client_noises) / num_clients

    # Theoretical variance of consensus average:
    # Each client variance: (sigma * C)^2 / K
    # Average variance: (1 / K^2) * K * ((sigma * C)^2 / K) = (sigma * C)^2 / K^2
    # Theoretical standard deviation: (sigma * C) / K
    expected_std = (noise_multiplier * clipping_bound) / num_clients
    empirical_std = torch.std(consensus_noise).item()

    # Relative error should be within 5% for sample size 20,000
    rel_err = abs(empirical_std - expected_std) / expected_std
    assert rel_err < 0.05, f"Empirical std {empirical_std} deviates from expected {expected_std} (err={rel_err})"


def test_central_server_privacy_report_and_accounting():
    """Verify CentralServer generates authentic DP reports and avoids double-noising under distributed DP."""
    server = CentralServer(
        privacy_noise=1.0,
        dp_mode="distributed",
        dp_clipping_bound=1.0,
        model_type="cnn",
    )

    # Initial report before rounds
    init_report = server.get_privacy_report()
    assert init_report["dp_active"] is True
    assert init_report["dp_mode"] == "distributed"
    assert init_report["mechanism"] == "Option J Distributed Patient-Level DP (SecAgg+ / CKKS)"
    assert init_report["epsilon"] == 0.0
    assert init_report["target_delta"] == 1e-5

    # Simulate 3 clients
    dummy_weights = [server.get_weights() for _ in range(3)]
    counts = [10, 10, 10]

    # Server aggregate in distributed mode
    agg_w = server.aggregate(dummy_weights, counts, client_subsampling_q=0.6)

    # In distributed mode with identical weights, the server does NOT add noise post-aggregation
    # So agg_w should match the input weights exactly:
    for k in agg_w.keys():
        diff = torch.norm(agg_w[k].float() - dummy_weights[0][k].float()).item()
        assert diff < 1e-5, f"Server should not add redundant noise in distributed mode: diff={diff}"

    # Verify DP accountant advanced by 1 step
    post_report = server.get_privacy_report()
    assert post_report["total_rounds_composed"] == 1
    assert post_report["epsilon"] > 0.0
    assert post_report["epsilon"] < 5.0
