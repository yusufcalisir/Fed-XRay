"""Granular Unit Test Suite: Medical Class Imbalance Loss Functions.

Verifies mathematical formulation, gradient backpropagation,
dynamic parameter decay, and prototype repulsion invariants.
"""

from __future__ import annotations
import numpy as np
import torch
from src.fed_xray.core.imbalance_losses import (
    DynamicAdaptiveFocalLoss,
    BalancedSoftmaxLoss,
    ClassBalancedLoss,
    LDAMLoss,
    PrototypeRepelLoss,
)


def test_dafl_loss_computation_and_backward():
    """Verify DAFL calculates non-negative scalar loss and backpropagates valid gradients."""
    class_counts = [1000, 100, 10]
    loss_fn = DynamicAdaptiveFocalLoss(class_counts=class_counts, current_round=1, total_rounds=10)

    logits = torch.randn(8, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 1, 0], dtype=torch.long)

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_dafl_dynamic_decay():
    """Verify DAFL imbalance exponent decays from round 1 to final round."""
    class_counts = [500, 50, 5]
    dafl_r1 = DynamicAdaptiveFocalLoss(class_counts=class_counts, current_round=1, total_rounds=10)
    dafl_r10 = DynamicAdaptiveFocalLoss(class_counts=class_counts, current_round=10, total_rounds=10)

    # In round 10, imbalance term decays, bringing weights closer to pure effective number weights
    assert not torch.allclose(dafl_r1.alpha_c, dafl_r10.alpha_c, atol=1e-3)


def test_balanced_softmax_logit_adjustment():
    """Verify Balanced Softmax shifts logits proportionally to negative log prior."""
    class_counts = [1000, 100, 10]
    bsm = BalancedSoftmaxLoss(class_counts=class_counts)

    # Prior of class 0 is largest, so log_prior of class 2 is lowest
    assert bsm.log_prior[0] > bsm.log_prior[1] > bsm.log_prior[2]

    logits = torch.zeros(3, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2])
    loss = bsm(logits, targets)
    assert loss.item() > 0.0


def test_class_balanced_effective_samples():
    """Verify Class-Balanced loss assigns higher weights to rarer classes."""
    class_counts = [1000, 100, 10]
    cb_loss = ClassBalancedLoss(class_counts=class_counts, beta=0.999)

    # Weights for class 2 must be strictly greater than class 0
    assert cb_loss.weights[2] > cb_loss.weights[1] > cb_loss.weights[0]

    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])
    loss = cb_loss(logits, targets)
    assert loss.item() > 0.0


def test_ldam_margin_scaling():
    """Verify LDAM imposes larger margin penalty on minority classes."""
    class_counts = [1000, 100, 10]
    ldam = LDAMLoss(class_counts=class_counts, max_m=0.5)

    # Minority class margin must be larger
    assert ldam.m_list[2] > ldam.m_list[1] > ldam.m_list[0]

    logits = torch.randn(6, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    loss = ldam(logits, targets)
    assert loss.item() > 0.0


def test_prototype_repel_loss_margin():
    """Verify PrototypeRepelLoss penalizes features closer to other class prototypes than margin."""
    repel_fn = PrototypeRepelLoss(margin=2.0)

    # Feature at origin
    features = torch.zeros(1, 16)
    targets = torch.tensor([0])

    # Prototype for class 1 at distance 0.5 (< margin 2.0 -> should trigger penalty)
    close_proto = torch.zeros(16)
    close_proto[0] = 0.5
    global_protos = {1: close_proto}

    loss_close = repel_fn(features, targets, global_protos)
    assert loss_close.item() > 0.0
    assert np.isclose(loss_close.item(), 1.5, atol=1e-3)

    # Prototype for class 1 at distance 3.0 (> margin 2.0 -> zero penalty)
    far_proto = torch.zeros(16)
    far_proto[0] = 3.0
    loss_far = repel_fn(features, targets, {1: far_proto})
    assert loss_far.item() == 0.0
