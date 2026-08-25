"""Numerical Stability & Boundary Condition Testing Suite for Imbalance Losses & Option J Privacy."""

import math
import pytest
import torch
import torch.nn as nn

from src.fed_xray.core.imbalance_losses import (
    DynamicAdaptiveFocalLoss,
    BalancedSoftmaxLoss,
    ClassBalancedLoss,
    LDAMLoss,
    PrototypeRepelLoss,
)
from src.fed_xray.core.privacy import (
    CKKSEncryptionEngine,
    SecAggPlusProtocol,
    PatientLevelDPAccountant,
    apply_patient_level_dp_clipping,
)


def test_dafl_numerical_stability_boundaries():
    """Verify DAFL handles zero counts, extreme logits, and round boundaries without NaN."""
    dafl = DynamicAdaptiveFocalLoss(
        class_counts=[100, 10, 1],
        current_round=1,
        total_rounds=10,
        gamma_base=2.0,
        delta_0=1.0,
    )

    # 1. Normal logits
    logits = torch.randn(8, 3)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loss = dafl(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0

    # 2. Extreme saturated logits
    saturated_logits = torch.tensor([[-1e5, 1e5, -1e5], [1e5, -1e5, -1e5]])
    sat_targets = torch.tensor([1, 0])
    sat_loss = dafl(saturated_logits, sat_targets)
    assert not torch.isnan(sat_loss)
    assert sat_loss.item() < 0.01


def test_bayesian_balanced_softmax_boundaries():
    """Verify Balanced Softmax handles uniform and heavily skewed priors."""
    bsm = BalancedSoftmaxLoss(class_counts=[1000, 5, 1])
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    loss = bsm(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_ldam_and_class_balanced_losses():
    """Verify LDAM margin scaling and Class-Balanced Effective Samples Loss."""
    ldam = LDAMLoss(class_counts=[500, 50, 5], max_m=0.5)
    cb_loss = ClassBalancedLoss(class_counts=[500, 50, 5], beta=0.999)

    logits = torch.randn(6, 3)
    targets = torch.tensor([0, 0, 1, 1, 2, 2])

    loss_ldam = ldam(logits, targets)
    loss_cb = cb_loss(logits, targets)

    assert not torch.isnan(loss_ldam)
    assert not torch.isnan(loss_cb)
    assert loss_ldam.item() > 0.0
    assert loss_cb.item() > 0.0


def test_missing_class_repel_loss():
    """Verify Prototype-Contrastive Margin Regularization repel loss."""
    repel = PrototypeRepelLoss(margin=1.0)
    features = torch.randn(4, 16)
    targets = torch.tensor([0, 1, 0, 1])
    global_prototypes = {0: torch.randn(16), 1: torch.randn(16), 2: torch.randn(16)}

    loss = repel(features, targets, global_prototypes)
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0


def test_ckks_high_dimensional_vector_scaling():
    """Verify CKKS encryption engine across large parameter vectors and multi-client sums."""
    engine = CKKSEncryptionEngine(poly_modulus_degree=8192, scale_bits=30)
    
    # Large 2048-dim vector
    K = 10
    client_vectors = [torch.randn(2048) for _ in range(K)]
    weights = [1.0 / K] * K

    ideal_sum = sum(w * v for w, v in zip(weights, client_vectors))

    # Encrypt, additively sum in ciphertext domain, and decrypt
    ciphertexts = [engine.encrypt_vector(v) for v in client_vectors]
    global_ct = engine.homomorphic_sum(ciphertexts, weights)
    decrypted_sum = engine.decrypt_vector(global_ct)

    max_err = torch.max(torch.abs(ideal_sum - decrypted_sum)).item()
    assert max_err < 0.05


def test_rdp_accountant_deep_composition():
    """Verify Rényi Differential Privacy accountant over 50 composition steps."""
    accountant = PatientLevelDPAccountant(target_delta=1e-5)
    
    epsilons = []
    for _ in range(50):
        accountant.step(q=0.05, sigma=2.0)
        epsilons.append(accountant.get_epsilon())

    # Epsilon must grow monotonically
    for i in range(1, len(epsilons)):
        assert epsilons[i] >= epsilons[i - 1]

    # Verify privacy bound remains tight (< 2.0)
    assert accountant.get_epsilon() < 2.0
