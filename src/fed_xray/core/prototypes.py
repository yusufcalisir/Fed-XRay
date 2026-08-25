"""Fed-XRay: Personalized Prototype Learning & Dispersion-Weighted Synthesis.

Implements:
- Feature embedding extraction across CNN, ViT, and Medical Foundation models
- Intra-class feature covariance trace computation (dispersion metric)
- Confidence & dispersion weighted prototype aggregation:
  alpha_{k,c} ~ n_{k,c} / (Tr(Sigma_{k,c}) + eps)
- Multimodal Prototype Fusion with frozen semantic text anchors:
  p_c = lambda * p_c^{img} + (1 - lambda) * p_c^{txt}
- Metric distance alignment loss
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def extract_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract penultimate feature embeddings across CNN, ViT, and Foundation backbones."""
    if hasattr(model, "extract_features"):
        return model.extract_features(x)
    elif hasattr(model, "forward_features"):
        return model.forward_features(x)
    elif hasattr(model, "conv1") and hasattr(model, "conv2") and hasattr(model, "fc1"):
        x = model.conv1(x)
        x = F.relu(x)
        x = model.pool1(x)

        x = model.conv2(x)
        x = F.relu(x)
        x = model.pool2(x)

        x = x.view(x.size(0), -1)
        x = model.dropout(x)
        features = F.relu(model.fc1(x))
        return features
    else:
        # Fallback forward pass
        return model(x)


def compute_local_prototypes_and_dispersion(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 3,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, float], Dict[int, int]]:
    """Compute local empirical class prototypes and intra-class covariance traces:
    
    p_{k,c} = (1 / n_{k,c}) * sum_{i: y_i = c} f(x_i)
    Sigma_{k,c} = (1 / n_{k,c}) * sum (f(x_i) - p_{k,c})(f(x_i) - p_{k,c})^T
    Trace = Tr(Sigma_{k,c})
    """
    model.eval()
    class_features: Dict[int, List[torch.Tensor]] = {c: [] for c in range(num_classes)}

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            feats = extract_features(model, images)

            for f, y in zip(feats, labels):
                c = y.item()
                if c < num_classes:
                    class_features[c].append(f.detach().cpu())

    prototypes: Dict[int, torch.Tensor] = {}
    traces: Dict[int, float] = {}
    counts: Dict[int, int] = {}

    # Infer feature dimension
    sample_feat = next((f[0] for f in class_features.values() if len(f) > 0), None)
    feature_dim = sample_feat.shape[0] if sample_feat is not None else 128

    for c in range(num_classes):
        feats_c = class_features[c]
        counts[c] = len(feats_c)

        if len(feats_c) > 0:
            stack_f = torch.stack(feats_c)  # [N_c, D]
            proto = torch.mean(stack_f, dim=0)  # [D]
            prototypes[c] = proto

            if len(feats_c) > 1:
                # Sum of sample feature variances = Tr(Sigma)
                var_sum = torch.sum(torch.var(stack_f, dim=0, unbiased=True)).item()
                traces[c] = max(var_sum, 1e-4)
            else:
                traces[c] = 1.0
        else:
            # Fallback zero prototype if unobserved on client
            prototypes[c] = torch.zeros(feature_dim)
            traces[c] = 10.0

    return prototypes, traces, counts


def aggregate_prototypes_dispersion_weighted(
    client_prototypes: List[Dict[int, torch.Tensor]],
    client_traces: List[Dict[int, float]],
    client_counts: List[Dict[int, int]],
    num_classes: int = 3,
    epsilon: float = 1e-5,
) -> Dict[int, torch.Tensor]:
    """Synthesize global prototypes using confidence & dispersion trace weighting:
    
    alpha_{k,c} ~ n_{k,c} / (Tr(Sigma_{k,c}) + eps)
    p_c = sum_k alpha_{k,c} * p_{k,c}
    """
    global_prototypes: Dict[int, torch.Tensor] = {}

    for c in range(num_classes):
        weights = []
        protos = []

        for k in range(len(client_prototypes)):
            if c in client_prototypes[k] and client_counts[k].get(c, 0) > 0:
                n_kc = client_counts[k][c]
                tr_kc = client_traces[k].get(c, 1.0)
                alpha_kc = n_kc / (tr_kc + epsilon)

                weights.append(alpha_kc)
                protos.append(client_prototypes[k][c].float())

        if weights and sum(weights) > 0:
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            weights_norm = weights_tensor / weights_tensor.sum()

            global_p = torch.zeros_like(protos[0])
            for w, p in zip(weights_norm, protos):
                global_p += w * p
            global_prototypes[c] = global_p
        else:
            dim = (
                client_prototypes[0][0].shape[0]
                if client_prototypes and 0 in client_prototypes[0]
                else 128
            )
            global_prototypes[c] = torch.zeros(dim)

    return global_prototypes


def synthesize_multimodal_prototypes(
    visual_prototypes: Dict[int, torch.Tensor],
    text_prototypes: torch.Tensor,
    lambda_weight: float = 0.7,
) -> Dict[int, torch.Tensor]:
    """Combines aggregated visual prototypes with frozen semantic text anchors:
    
    p_c = lambda * p_{c}^{img} + (1 - lambda) * p_{c}^{txt}
    """
    fused_prototypes: Dict[int, torch.Tensor] = {}
    for c, p_img in visual_prototypes.items():
        if c < text_prototypes.shape[0]:
            p_txt = text_prototypes[c].to(p_img.device)
            # Ensure dimension alignment
            if p_img.shape[0] == p_txt.shape[0]:
                fused = lambda_weight * F.normalize(p_img, p=2, dim=-1) + (1.0 - lambda_weight) * p_txt
                fused_prototypes[c] = F.normalize(fused, p=2, dim=-1)
            else:
                fused_prototypes[c] = p_img
        else:
            fused_prototypes[c] = p_img
    return fused_prototypes


def compute_prototype_distance_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    global_prototypes: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Compute MSE alignment loss between local representations and global class prototypes."""
    loss = torch.tensor(0.0, device=features.device)
    count = 0

    for i in range(features.size(0)):
        y_i = targets[i].item()
        if y_i in global_prototypes:
            proto = global_prototypes[y_i].to(features.device)
            loss += F.mse_loss(features[i], proto)
            count += 1

    return loss / max(count, 1)
