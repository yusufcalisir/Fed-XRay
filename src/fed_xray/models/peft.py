"""Parameter-Efficient Fine-Tuning (PEFT) Modules for Federated Vision Transformers.

Implements:
1. Low-Rank Adaptation (LoRA) for linear layers (Hu et al., ICLR 2022).
2. Federated Freeze-A LoRA (FFA-LoRA) to eliminate bilinear aggregation error.
3. Federated Share-A LoRA (FedSA-LoRA / Fed-ALAS) for global subspace + site personalization.
4. Federated Adaptive Factor Sharing LoRA (FedAS-LoRA) with RSS subspace routing.
5. FlexLoRA Server-Side Truncated SVD Aggregator for heterogeneous low-rank alignment.
6. HetLoRA Dynamic Rank Slicing and zero-padding.
7. FedPerfix Attention-Decoupled Plugins for deep transformer layers.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) layer wrapping a frozen base linear layer.
    
    Forward pass:
        y = x W_0^T + (alpha / r) * x A^T B^T
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        freeze_a: bool = False,
        share_a: bool = False,
        share_b: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.freeze_a = freeze_a
        self.share_a = share_a
        self.share_b = share_b

        # Base layer - permanently frozen
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        if r > 0:
            # Matrix A: in_features -> r (initialized with Gaussian)
            self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
            # Matrix B: r -> out_features (initialized to zero so delta W = 0 initially)
            self.lora_B = nn.Parameter(torch.empty(self.out_features, r))
            self.reset_parameters()

            if freeze_a:
                self.lora_A.requires_grad = False
            else:
                self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True

            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.dropout = nn.Identity()

    def reset_parameters(self) -> None:
        """Kaiming uniform for matrix A, zeros for matrix B."""
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self.r > 0:
            adapter_out = self.dropout(x) @ self.lora_A.t()
            adapter_out = adapter_out @ self.lora_B.t()
            result = result + self.scaling * adapter_out
        return result

    def get_effective_delta(self) -> torch.Tensor:
        """Compute the reconstructed full weight delta: Delta W = (alpha/r) * B @ A."""
        if self.r > 0:
            return self.scaling * (self.lora_B @ self.lora_A)
        return torch.zeros(self.out_features, self.in_features, device=self.base_layer.weight.device)


class FFALoRALinear(LoRALinear):
    """Federated Freeze-A LoRA: Matrix A is frozen globally; only B is trained & federated."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_layer=base_layer,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_a=True,
            share_a=False,
        )


class FedSALoRALinear(LoRALinear):
    """Federated Share-A LoRA (Fed-ALAS): Matrix A is shared/federated, Matrix B is local/personalized."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_layer=base_layer,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_a=False,
            share_a=True,
            share_b=False,
        )


class FedASLoRALinear(LoRALinear):
    """Federated Adaptive Factor Sharing LoRA: Dynamically routes sharing mode (Share-A vs Share-B)."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        sharing_mode: str = "share_a",  # "share_a" or "share_b"
    ) -> None:
        share_a = sharing_mode == "share_a"
        share_b = sharing_mode == "share_b"
        super().__init__(
            base_layer=base_layer,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_a=False,
            share_a=share_a,
            share_b=share_b,
        )
        self.sharing_mode = sharing_mode


class HetLoRALinear(LoRALinear):
    """HetLoRA: Variable rank r_k per client with dynamic zero-padding for global aggregation."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        max_r: int = 32,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_layer=base_layer,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.max_r = max_r

    def get_padded_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pads local matrices A and B with zeros up to max_r for aggregation."""
        if self.r >= self.max_r:
            return self.lora_A.data, self.lora_B.data

        pad_r = self.max_r - self.r
        padded_A = torch.cat([self.lora_A.data, torch.zeros(pad_r, self.in_features, device=self.lora_A.device)], dim=0)
        padded_B = torch.cat([self.lora_B.data, torch.zeros(self.out_features, pad_r, device=self.lora_B.device)], dim=1)
        return padded_A, padded_B


class FedPerfixPlugin(nn.Module):
    """FedPerfix: Parallel attention plugin adapter transferring global trajectories into local attention maps."""

    def __init__(
        self,
        embed_dim: int = 768,
        plugin_dim: int = 32,
        scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.down_proj = nn.Linear(embed_dim, plugin_dim, bias=False)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(plugin_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Parallel residual attention branch
        x_norm = self.norm(x)
        adapter_out = self.up_proj(self.act(self.down_proj(x_norm)))
        return self.scale * adapter_out


def compute_rank_aware_subspace_sufficiency(
    client_features_list: List[torch.Tensor],
    rank: int = 16,
) -> Dict[str, float]:
    """Computes Rank-Aware Shared-Subspace Sufficiency (RSS) metrics across institutions:
    
    RSS_input = (1/K) * sum_k Tr(U_A^(k)^T \bar{U}_A)
    """
    K = len(client_features_list)
    if K < 2:
        return {"rss_input": 1.0, "rss_output": 1.0, "recommended_mode": "share_a"}

    subspace_bases = []
    for feats in client_features_list:
        # feats: [N, D]
        centered = feats - feats.mean(dim=0, keepdim=True)
        # SVD on local feature covariance
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        subspace_bases.append(Vh[:rank])  # Top-r eigenvectors

    # Global shared principal subspace
    stacked_bases = torch.cat(subspace_bases, dim=0)  # [K*rank, D]
    _, _, Vh_global = torch.linalg.svd(stacked_bases, full_matrices=False)
    V_global = Vh_global[:rank]  # [rank, D]

    rss_scores = []
    for V_k in subspace_bases:
        # Cosine subspace overlap: Tr(V_k @ V_global^T @ V_global @ V_k^T) / rank
        overlap = torch.norm(V_k @ V_global.t(), p="fro") ** 2 / rank
        rss_scores.append(overlap.item())

    avg_rss = sum(rss_scores) / K
    recommended_mode = "share_a" if avg_rss >= 0.50 else "share_b"

    return {
        "rss_input": avg_rss,
        "rss_output": 1.0 - avg_rss,
        "recommended_mode": recommended_mode,
    }


def reconstruct_and_svd_aggregate_lora(
    client_lora_A: List[torch.Tensor],
    client_lora_B: List[torch.Tensor],
    weights: List[float],
    target_rank: int = 16,
    scaling: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""FlexLoRA Server-Side SVD Aggregator:
    
    1. Reconstruct full weight delta: \bar{\Delta W} = sum p_k (B_k @ A_k)
    2. Compute Truncated SVD: \bar{\Delta W} = U_r \Sigma_r V_r^T
    3. Synthesize balanced factor matrices: \bar{B} = U_r \Sigma_r^{1/2}, \bar{A} = \Sigma_r^{1/2} V_r^T
    """
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]

    d_out, r = client_lora_B[0].shape
    _, d_in = client_lora_A[0].shape

    # Reconstruct weighted sum of outer products
    delta_W = torch.zeros(d_out, d_in, dtype=torch.float32)
    for A_k, B_k, p_k in zip(client_lora_A, client_lora_B, norm_weights):
        delta_W += p_k * (B_k @ A_k)

    # Truncated SVD
    U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
    actual_r = min(target_rank, len(S))

    U_r = U[:, :actual_r]
    S_r = S[:actual_r]
    Vh_r = Vh[:actual_r, :]

    sqrt_S = torch.diag(torch.sqrt(torch.clamp(S_r, min=1e-8)))

    # Synthesize new low-rank factors
    new_B = (U_r @ sqrt_S) / math.sqrt(max(1e-8, scaling))
    new_A = (sqrt_S @ Vh_r) / math.sqrt(max(1e-8, scaling))

    return new_A, new_B


def inject_lora_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    r: int = 16,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    mode: str = "lora",  # "lora", "ffa_lora", "fedsa_lora", "fedas_lora", "het_lora", "fedperfix"
    deep_layers_only: bool = False,
    num_deep_layers: int = 4,
    total_layers: int = 12,
) -> Dict[str, Any]:
    """Injects PEFT LoRA adapters or FedPerfix plugins into targeted Vision Transformer layers."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "query", "value", "to_q", "to_v"]

    adapter_cls = LoRALinear
    if mode == "ffa_lora":
        adapter_cls = FFALoRALinear
    elif mode == "fedsa_lora":
        adapter_cls = FedSALoRALinear
    elif mode == "fedas_lora":
        adapter_cls = FedASLoRALinear
    elif mode == "het_lora":
        adapter_cls = HetLoRALinear

    min_layer_idx = (total_layers - num_deep_layers) if deep_layers_only else 0

    # Freeze entire backbone first
    for param in model.parameters():
        param.requires_grad = False

    injected_count = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                is_target = any(tgt in child_name for tgt in target_modules)
                is_deep = True
                if deep_layers_only:
                    parts = name.split(".")
                    for p in parts:
                        if p.isdigit():
                            idx = int(p)
                            if idx < min_layer_idx:
                                is_deep = False
                            break

                if is_target and is_deep:
                    wrapped = adapter_cls(
                        base_layer=child,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                    )
                    setattr(module, child_name, wrapped)
                    injected_count += 1

    # Keep top-level classification head trainable
    for name, param in model.named_parameters():
        if name.startswith("head.") or name.startswith("classifier.") or name.startswith("fc.") or name in ("head", "classifier", "fc"):
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    compression_pct = (1.0 - (trainable_params / max(1, total_params))) * 100.0

    return {
        "injected_adapters": injected_count,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "compression_percentage": compression_pct,
        "mode": mode,
    }


def extract_peft_state_dict(
    model: nn.Module,
    mode: str = "lora",
) -> Dict[str, torch.Tensor]:
    """Extracts only trainable PEFT adapter weights for communication."""
    state = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # If FFA-LoRA, only B is communicated
        if mode == "ffa_lora" and "lora_A" in name:
            continue
        # If FedSA-LoRA, only A is federated (B is kept private for site personalization)
        if mode == "fedsa_lora" and "lora_B" in name:
            continue
        state[name] = param.data.clone().cpu()
    return state


def load_peft_state_dict(
    model: nn.Module,
    peft_state_dict: Dict[str, torch.Tensor],
    mode: str = "lora",
) -> None:
    """Loads received PEFT state dict into the model parameters."""
    model_dict = model.state_dict()
    for k, v in peft_state_dict.items():
        if k in model_dict:
            model_dict[k].copy_(v.to(model_dict[k].device))
