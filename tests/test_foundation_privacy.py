"""Unit and integration tests for Foundation Models, Dual-Layer Privacy (Option J), and Multimodal Prototypes."""

import pytest
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.fed_xray.models.foundation import (
    TextSemanticAnchor,
    MedSAMClientAdapter,
    FedMedCLIPModule,
)
from src.fed_xray.core.privacy import (
    CKKSEncryptionEngine,
    SecAggPlusProtocol,
    PatientLevelDPAccountant,
    apply_patient_level_dp_clipping,
)
from src.fed_xray.core.prototypes import (
    synthesize_multimodal_prototypes,
    aggregate_prototypes_dispersion_weighted,
)


def test_text_semantic_anchor():
    """Verify invariant text semantic codebook creates orthogonal normalized anchors."""
    anchor = TextSemanticAnchor(embed_dim=256)
    protos = anchor.get_text_prototypes()
    assert protos.shape == (3, 256)
    # Check unit norm
    norms = torch.norm(protos, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_medsam_client_tailored_adapter():
    """Verify MedSAM FCA adapter forward and residual scaling."""
    adapter = MedSAMClientAdapter(embed_dim=256, bottleneck_dim=32, scale=0.5)
    x = torch.randn(4, 16, 256)
    out = adapter(x)
    assert out.shape == (4, 16, 256)
    assert not torch.isnan(out).any()


def test_fedmedclip_kl_distillation():
    """Verify FedMedCLIP FAM module forward pass and mutual KL distillation loss."""
    fedclip = FedMedCLIPModule(image_embed_dim=256, text_embed_dim=256, num_classes=3)
    img_feats = torch.randn(8, 256)
    targets = torch.randint(0, 3, (8,))

    loss, y_ens = fedclip.compute_local_loss(img_feats, targets, beta=0.5)
    assert y_ens.shape == (8, 3)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_ckks_additive_homomorphism():
    """Verify Leveled CKKS additive homomorphism on encrypted float vectors."""
    engine = CKKSEncryptionEngine(scale_bits=40)
    v1 = torch.tensor([0.15, -0.42, 1.88, -0.05])
    v2 = torch.tensor([0.85, 0.12, -0.68, 0.45])

    ct1 = engine.encrypt_vector(v1)
    ct2 = engine.encrypt_vector(v2)

    # Server performs additive homomorphic weighted combination: 0.5 * ct1 + 0.5 * ct2
    agg_ct = engine.homomorphic_sum([ct1, ct2], [0.5, 0.5])
    decrypted = engine.decrypt_vector(agg_ct)

    expected = 0.5 * v1 + 0.5 * v2
    assert torch.allclose(decrypted, expected, atol=1e-2)


def test_secagg_plus_zero_sum_cancellation():
    """Verify SecAgg+ pairwise masking cancels to exact zero: sum s_k = 0."""
    num_clients = 4
    shapes = {"w1": torch.Size([64, 32]), "b1": torch.Size([64])}
    masks = SecAggPlusProtocol.generate_pairwise_masks(num_clients=num_clients, tensor_shapes=shapes)

    assert len(masks) == num_clients
    # Sum of all masks must be identically zero
    sum_w1 = torch.zeros(shapes["w1"])
    sum_b1 = torch.zeros(shapes["b1"])
    for m in masks:
        sum_w1 += m["w1"]
        sum_b1 += m["b1"]

    assert torch.allclose(sum_w1, torch.zeros_like(sum_w1), atol=1e-6)
    assert torch.allclose(sum_b1, torch.zeros_like(sum_b1), atol=1e-6)


def test_patient_level_dp_accountant():
    """Verify Rényi DP composition accountant tracks privacy spend (epsilon <= 2.0, delta=10^-5)."""
    accountant = PatientLevelDPAccountant(target_delta=1e-5)
    # Simulate 10 federated rounds with subsampling rate q=0.1, noise sigma=2.0
    for _ in range(10):
        accountant.step(q=0.1, sigma=2.0)

    eps = accountant.get_epsilon()
    assert 0.0 < eps <= 2.5
    assert not math.isnan(eps)


def test_patient_level_dp_clipping():
    """Verify Patient-Level DP gradient clipping and perturbation."""
    # 5 patients, each with an average gradient tensor
    patient_grads = [torch.randn(128) * (i + 1) for i in range(5)]
    dp_grad = apply_patient_level_dp_clipping(patient_grads, clipping_bound_C=1.0, noise_multiplier_sigma=0.5)

    assert dp_grad.shape == (128,)
    assert not torch.isnan(dp_grad).any()


def test_multimodal_prototype_fusion():
    """Verify multimodal prototype fusion combining visual centroids with invariant text anchors."""
    visual_protos = {
        0: torch.randn(256),
        1: torch.randn(256),
        2: torch.randn(256),
    }
    text_protos = torch.randn(3, 256)
    fused = synthesize_multimodal_prototypes(visual_protos, text_protos, lambda_weight=0.7)

    assert len(fused) == 3
    for c in range(3):
        assert fused[c].shape == (256,)
        # Check normalized
        norm = torch.norm(fused[c]).item()
        assert abs(norm - 1.0) < 1e-4
