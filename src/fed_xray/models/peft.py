"""Parameter-Efficient Fine-Tuning (PEFT) Modules for Federated Vision Transformers.

Implements:
1. Low-Rank Adaptation (LoRA) for linear layers (Hu et al., ICLR 2022).
2. Federated Freeze-A LoRA (FFA-LoRA) to eliminate bilinear aggregation error.
3. Federated Share-A LoRA (FedSA-LoRA / Fed-ALAS) for global subspace + site personalization.
4. FedPerfix attention-decoupled selective injection for deep transformer layers.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Set, Tuple
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
    ) -> None:
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.freeze_a = freeze_a
        self.share_a = share_a

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
            # x: [..., in_features]
            # lora_A: [r, in_features] -> x @ A^T -> [..., r]
            # lora_B: [out_features, r] -> (x @ A^T) @ B^T -> [..., out_features]
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
        )


def inject_lora_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    r: int = 16,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    mode: str = "lora",  # "lora", "ffa_lora", "fedsa_lora"
    deep_layers_only: bool = False,
    num_deep_layers: int = 4,
    total_layers: int = 12,
) -> Dict[str, int]:
    """Injects PEFT LoRA adapters into targeted linear projection layers of a Vision Transformer.
    
    Args:
        model: PyTorch model/Vision Transformer.
        target_modules: Names of module substrings to adapt (e.g. ["q_proj", "v_proj"]).
        r: Rank of the low-rank adaptation decomposition.
        lora_alpha: Scaling factor.
        mode: Adaptation mode ("lora", "ffa_lora", "fedsa_lora").
        deep_layers_only: If True, enforce FedPerfix principle (adapt only deep layers L-k ... L).
        num_deep_layers: Number of deep layers to adapt when deep_layers_only=True.
        total_layers: Total number of transformer layers in backbone.
    
    Returns:
        Dictionary with parameter statistics (total, trainable, frozen, compression ratio).
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "query", "value", "to_q", "to_v"]

    adapter_cls = LoRALinear
    if mode == "ffa_lora":
        adapter_cls = FFALoRALinear
    elif mode == "fedsa_lora":
        adapter_cls = FedSALoRALinear

    min_layer_idx = (total_layers - num_deep_layers) if deep_layers_only else 0

    # Freeze entire backbone first
    for param in model.parameters():
        param.requires_grad = False

    injected_count = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Check target module match
                is_target = any(tgt in child_name for tgt in target_modules)
                
                # Check deep layer filter
                is_deep = True
                if deep_layers_only:
                    # Look for layer index in module name (e.g., "blocks.10.attn")
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

    # Keep final classification head trainable
    for name, param in model.named_parameters():
        if name.startswith("head.") or name.startswith("classifier.") or name.startswith("fc.") or name == "head" or name == "classifier" or name == "fc":
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
        # If FFA-LoRA, only B parameters are sent
        if mode == "ffa_lora" and "lora_A" in name:
            continue
        # If FedSA-LoRA, only A parameters are federated globally (B is kept private)
        if mode == "fedsa_lora" and "lora_B" in name:
            continue
        state[name] = param.data.clone().cpu()
    return state


def load_peft_state_dict(
    model: nn.Module,
    peft_state_dict: Dict[str, torch.Tensor],
    mode: str = "lora",
) -> None:
    """Loads received PEFT state dict into the local model parameters."""
    model_dict = model.state_dict()
    for k, v in peft_state_dict.items():
        if k in model_dict:
            model_dict[k].copy_(v.to(model_dict[k].device))
