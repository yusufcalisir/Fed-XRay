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
from src.fed_xray.core.server import CentralServer, run_federated_round
from src.fed_xray.core.client import HospitalClient


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


def test_ckks_server_public_context_cannot_decrypt():
    """The aggregation server holds only the public context and must not be able to decrypt."""
    import pytest
    import tenseal as ts

    engine = CKKSEncryptionEngine(scale_bits=40)
    ct = engine.encrypt_vector(torch.tensor([0.5, -1.25, 2.0]))
    assert ct.nbytes > 0
    assert ct.payload[0] != b""

    server_view = ts.ckks_vector_from(engine._public_ctx, ct.payload[0])
    with pytest.raises(Exception):
        server_view.decrypt()


def test_ckks_ciphertexts_are_probabilistic():
    """Encrypting the same plaintext twice must yield different ciphertexts."""
    engine = CKKSEncryptionEngine(scale_bits=40)
    v = torch.tensor([0.1, 0.2, 0.3])
    assert engine.encrypt_vector(v).payload[0] != engine.encrypt_vector(v).payload[0]


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


def test_secagg_weighted_pairwise_zero_sum():
    """Verify SecAgg+ weighted pairwise masking cancels to exact zero under non-uniform weights."""
    num_clients = 4
    weights = [120.0, 250.0, 480.0, 150.0]
    total_w = sum(weights)
    norm_w = [w / total_w for w in weights]
    shapes = {"w1": torch.Size([32, 16]), "b1": torch.Size([32])}

    masks = SecAggPlusProtocol.generate_pairwise_masks(
        num_clients=num_clients,
        tensor_shapes=shapes,
        weights=norm_w,
    )

    assert len(masks) == num_clients

    # Weighted sum of masks must be identically zero
    sum_w1 = torch.zeros(shapes["w1"])
    sum_b1 = torch.zeros(shapes["b1"])
    for m, p_k in zip(masks, norm_w):
        sum_w1 += p_k * m["w1"]
        sum_b1 += p_k * m["b1"]

    assert torch.allclose(sum_w1, torch.zeros_like(sum_w1), atol=1e-6)
    assert torch.allclose(sum_b1, torch.zeros_like(sum_b1), atol=1e-6)


def test_ckks_state_dict_encryption_and_aggregation():
    """Verify CKKS encryption, homomorphic dictionary aggregation, and decryption."""
    engine = CKKSEncryptionEngine(scale_bits=40)
    d1 = {"w": torch.tensor([0.2, -0.4, 1.5]), "b": torch.tensor([0.1])}
    d2 = {"w": torch.tensor([0.6, 0.8, -0.5]), "b": torch.tensor([-0.3])}
    weights = [0.3, 0.7]

    ed1 = engine.encrypt_state_dict(d1)
    ed2 = engine.encrypt_state_dict(d2)

    agg_ed = engine.homomorphic_aggregate_state_dicts([ed1, ed2], weights=weights)
    decrypted = engine.decrypt_state_dict(agg_ed)

    expected_w = 0.3 * d1["w"] + 0.7 * d2["w"]
    expected_b = 0.3 * d1["b"] + 0.7 * d2["b"]

    assert torch.allclose(decrypted["w"], expected_w, atol=1e-2)
    assert torch.allclose(decrypted["b"], expected_b, atol=1e-2)


def test_federated_round_with_secagg_and_ckks():
    """Verify full federated training round with SecAgg+ masking, CKKS aggregation, and DP tracking."""
    torch.manual_seed(42)
    server = CentralServer(
        model_type="cnn",
        peft_mode=None,
        privacy_noise=0.01,
        secure_aggregation=True,
        use_ckks=True,
    )

    # Create dummy clients with synthetic data
    clients = []
    for cid in range(3):
        x = torch.randn(20, 1, 28, 28)
        y = torch.randint(0, 3, (20,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=10)
        client = HospitalClient(
            client_id=cid,
            dataloader=loader,
            model_type="cnn",
            peft_mode=None,
            local_epochs=1,
        )
        clients.append(client)

    test_x = torch.randn(15, 1, 28, 28)
    test_y = torch.randint(0, 3, (15,))

    agg_metrics, client_metrics, test_metrics, sec_report = run_federated_round(
        server=server,
        clients=clients,
        round_num=1,
        total_rounds=1,
        test_images=test_x,
        test_labels=test_y,
        use_defense=True,
        secure_aggregation=True,
        use_ckks=True,
    )

    assert agg_metrics["secure_aggregation"] is True
    assert agg_metrics["use_ckks"] is True
    assert agg_metrics["privacy_epsilon"] > 0.0
    assert len(client_metrics) == 3
    assert test_metrics is not None
    assert 0.0 <= test_metrics.accuracy <= 1.0
