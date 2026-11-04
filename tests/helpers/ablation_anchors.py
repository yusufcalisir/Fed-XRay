"""Geometric ablation class anchors for offline non-multimodal comparative studies.

Relocated from src/fed_xray/models/foundation.py per Rule 12 (Zero Dormant / Disconnected Code)
and Rule 8 (Frozen Medical Foundation Grounding).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalClassAnchor(nn.Module):
    """Purely geometric orthonormal class coordinate anchors on unit hypersphere S^{d-1}.

    Used strictly as an abstract geometric prototype baseline for offline ablation benchmarks
    to assess performance when multimodal language supervision is unavailable.
    For genuine clinical multimodal semantic anchoring in production, use TextSemanticAnchor
    (Rule 8, Rule 12).
    """

    semantic_codebook: torch.Tensor

    def __init__(self, embed_dim: int = 512, num_classes: int = 3, seed: int = 42) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.seed = seed
        self.register_buffer("semantic_codebook", torch.empty(num_classes, embed_dim))
        self._initialize_codebook()
        self.requires_grad_(False)

    def _initialize_codebook(self) -> None:
        gen = torch.Generator().manual_seed(self.seed)
        vectors = torch.randn(self.num_classes, self.embed_dim, generator=gen)
        q, _ = torch.linalg.qr(vectors.t())
        orthogonal_anchors = q.t()[: self.num_classes]
        self.semantic_codebook.copy_(F.normalize(orthogonal_anchors, p=2, dim=-1))

    def get_text_prototypes(self) -> torch.Tensor:
        """Return the fixed anchor matrix [num_classes, embed_dim] (interface compatibility)."""
        return self.semantic_codebook.clone()

    def get_prototypes(self) -> torch.Tensor:
        """Return the fixed anchor matrix [num_classes, embed_dim]."""
        return self.semantic_codebook.clone()

    def forward(self, class_indices: torch.Tensor) -> torch.Tensor:
        return self.semantic_codebook[class_indices]

