"""Vision Transformer (ViT) Architectures for Federated Medical Imaging.

Provides isotropic Vision Transformer backbones with:
- Patch Embedding & Positional Encodings
- Multi-Head Self-Attention (MHSA) with explicit Q, K, V, O projections for PEFT adaptation
- Transformer Encoder Blocks with Pre-LayerNorm architecture
- Representation extraction for Prototype Metric Alignment and Federated Contrastive Learning (MOON)
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Splits 2D image into non-overlapping patches and projects them into embedding space."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # proj: [B, embed_dim, grid, grid] -> flatten to [B, embed_dim, num_patches] -> transpose to [B, num_patches, embed_dim]
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with explicit linear projection layers for PEFT / LoRA adaptation."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Explicit individual projection layers to enable target-specific LoRA hooks
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = self.out_proj  # Alias for backward compatibility

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_weights = attn
        attn = self.attn_drop(attn)

        # Weighted values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        if return_attn:
            return out, attn_weights
        return out


class MLP(nn.Module):
    """Transformer Feed-Forward Network with GELU non-linearity."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Single Vision Transformer Encoder block (Pre-LN formulation)."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if return_attn:
            attn_out, weights = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, weights
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Modular Vision Transformer for Medical Imaging & Federated Learning."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        num_classes: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Weight authenticity & provenance tracking (Rule 12, MOD-02)
        self.model_type: str = "vit"
        self.is_pretrained: bool = False
        self.checkpoint_provenance: str = "uninitialized_scratch_weights"
        self.checkpoint_hash: Optional[str] = None
        self.strict_checkpoint: bool = False
        self.last_load_error: Optional[str] = None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate feature embedding z for prototype alignment and contrastive loss."""
        if x.shape[-2:] != (self.patch_embed.img_size, self.patch_embed.img_size):
            x = F.interpolate(
                x,
                size=(self.patch_embed.img_size, self.patch_embed.img_size),
                mode="bilinear",
                align_corners=False,
            )
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]  # [B, embed_dim]

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits

    def get_model_provenance(self) -> Dict[str, Any]:
        """Return structured weight authenticity and provenance metadata (MOD-02)."""
        return {
            "model_type": getattr(self, "model_type", "vit"),
            "is_pretrained": self.is_pretrained,
            "checkpoint_provenance": self.checkpoint_provenance,
            "checkpoint_hash": self.checkpoint_hash,
            "strict_checkpoint": self.strict_checkpoint,
            "is_fallback_distilled": "torchvision" in str(self.checkpoint_provenance),
            "last_load_error": self.last_load_error,
        }


def load_pretrained_vit_weights(
    model: VisionTransformer,
    checkpoint_path: Optional[str] = None,
    strict: bool = False,
) -> bool:
    """Transfer official torchvision ViT pretrained weights or custom checkpoint into VisionTransformer.

    Supports:
    - Loading from local or custom checkpoint path (if provided)
    - Full-parameter weight transfer for ViT-Base (768-dim, 12 layers)
    - Progressive subspace projection for scaled variants (ViT-Small 256-dim, ViT-Tiny 128-dim)
    - Dynamic bicubic resampling for patch embedding projection filter (1-channel medical scans)
    - 2D bicubic grid interpolation for position embeddings (supporting 28x28, 224x224, etc.)
    - Direct loading into q_proj, k_proj, v_proj, and out_proj layers
    """
    import hashlib
    import logging
    logger = logging.getLogger("fed_xray.models")

    try:
        state: Dict[str, torch.Tensor]
        provenance: str
        if checkpoint_path is not None:
            p = Path(checkpoint_path)
            if not p.exists():
                model.is_pretrained = False
                model.checkpoint_provenance = "uninitialized_scratch_weights"
                model.checkpoint_hash = None
                raise RuntimeError(f"VisionTransformer checkpoint not found at: {checkpoint_path}")
            state = torch.load(p, map_location="cpu", weights_only=True)
            provenance = f"local_checkpoint:{checkpoint_path}"
        else:
            try:
                import torchvision.models as models

                weights = models.ViT_B_16_Weights.DEFAULT
                state = weights.get_state_dict(progress=False)
                provenance = "official_torchvision_vit_b16"
            except Exception as exc:
                model.is_pretrained = False
                model.checkpoint_provenance = "uninitialized_scratch_weights"
                model.checkpoint_hash = None
                if strict:
                    raise RuntimeError(f"Failed to load official torchvision ViT_B_16 weights: {exc}") from exc
                logger.warning(
                    "Could not load official ViT_B_16 weights from torchvision: %s. "
                    "Model running with scratch weights (strict=False).",
                    exc,
                )
                return False

        dim = model.embed_dim

        # 1. Patch projection
        if "conv_proj.weight" in state:
            w = state["conv_proj.weight"]  # [768, 3, 16, 16]
            if model.patch_embed.proj.in_channels == 1:
                w = w.mean(dim=1, keepdim=True)  # [768, 1, 16, 16]
            kh, kw = model.patch_embed.proj.kernel_size
            if (kh, kw) != (w.shape[2], w.shape[3]):
                w = F.interpolate(w, size=(kh, kw), mode="bicubic", align_corners=True)
            out_c = model.patch_embed.proj.out_channels
            model.patch_embed.proj.weight.data.copy_(w[:out_c])
        if "conv_proj.bias" in state and model.patch_embed.proj.bias is not None:
            out_c = model.patch_embed.proj.out_channels
            model.patch_embed.proj.bias.data.copy_(state["conv_proj.bias"][:out_c])

        # 2. Tokens & Positional Embeddings
        if "class_token" in state:
            model.cls_token.data.copy_(state["class_token"][:, :, :dim])
        if "encoder.pos_embedding" in state:
            target_tokens = model.pos_embed.shape[1]
            if target_tokens == state["encoder.pos_embedding"].shape[1]:
                model.pos_embed.data.copy_(state["encoder.pos_embedding"][:, :, :dim])
            else:
                cls_p = state["encoder.pos_embedding"][:, :1, :dim]
                patch_p = state["encoder.pos_embedding"][:, 1:, :dim]
                orig_grid = math.isqrt(patch_p.shape[1])
                target_grid = math.isqrt(target_tokens - 1)
                if target_grid * target_grid == target_tokens - 1:
                    patch_2d = patch_p.reshape(1, orig_grid, orig_grid, dim).permute(0, 3, 1, 2)
                    patch_interp = F.interpolate(
                        patch_2d, size=(target_grid, target_grid), mode="bicubic", align_corners=True
                    )
                    patch_interp = patch_interp.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos = torch.cat([cls_p, patch_interp], dim=1)
                    model.pos_embed.data.copy_(new_pos)
                else:
                    p_interp = F.interpolate(
                        state["encoder.pos_embedding"][:, :, :dim].transpose(1, 2),
                        size=target_tokens,
                        mode="linear",
                        align_corners=True,
                    ).transpose(1, 2)
                    model.pos_embed.data.copy_(p_interp)

        # 3. Transformer Encoder Blocks
        for i, raw_blk in enumerate(model.blocks):
            blk: Any = raw_blk
            src_idx = min(i, 11)
            prefix = f"encoder.layers.encoder_layer_{src_idx}"

            # LayerNorm 1
            if f"{prefix}.ln_1.weight" in state:
                blk.norm1.weight.data.copy_(state[f"{prefix}.ln_1.weight"][:dim])
                blk.norm1.bias.data.copy_(state[f"{prefix}.ln_1.bias"][:dim])

            # Self-Attention
            if f"{prefix}.self_attention.in_proj_weight" in state:
                in_w = state[f"{prefix}.self_attention.in_proj_weight"]
                q_weight, k_weight, v_weight = in_w.chunk(3, dim=0)
                blk.attn.q_proj.weight.data.copy_(q_weight[:dim, :dim])
                blk.attn.k_proj.weight.data.copy_(k_weight[:dim, :dim])
                blk.attn.v_proj.weight.data.copy_(v_weight[:dim, :dim])

            if f"{prefix}.self_attention.in_proj_bias" in state:
                in_b = state[f"{prefix}.self_attention.in_proj_bias"]
                qb, kb, vb = in_b.chunk(3, dim=0)
                blk.attn.q_proj.bias.data.copy_(qb[:dim])
                blk.attn.k_proj.bias.data.copy_(kb[:dim])
                blk.attn.v_proj.bias.data.copy_(vb[:dim])

            if f"{prefix}.self_attention.out_proj.weight" in state:
                out_p = getattr(blk.attn, "out_proj", getattr(blk.attn, "proj", None))
                if out_p is not None:
                    out_p.weight.data.copy_(state[f"{prefix}.self_attention.out_proj.weight"][:dim, :dim])
                    out_p.bias.data.copy_(state[f"{prefix}.self_attention.out_proj.bias"][:dim])

            # LayerNorm 2
            if f"{prefix}.ln_2.weight" in state:
                blk.norm2.weight.data.copy_(state[f"{prefix}.ln_2.weight"][:dim])
                blk.norm2.bias.data.copy_(state[f"{prefix}.ln_2.bias"][:dim])

            # MLP
            mlp_dim = blk.mlp.fc1.out_features
            if f"{prefix}.mlp.linear_1.weight" in state:
                blk.mlp.fc1.weight.data.copy_(state[f"{prefix}.mlp.linear_1.weight"][:mlp_dim, :dim])
                blk.mlp.fc1.bias.data.copy_(state[f"{prefix}.mlp.linear_1.bias"][:mlp_dim])
            if f"{prefix}.mlp.linear_2.weight" in state:
                blk.mlp.fc2.weight.data.copy_(state[f"{prefix}.mlp.linear_2.weight"][:dim, :mlp_dim])
                blk.mlp.fc2.bias.data.copy_(state[f"{prefix}.mlp.linear_2.bias"][:dim])

        # 4. Final LayerNorm
        if "encoder.ln.weight" in state:
            model.norm.weight.data.copy_(state["encoder.ln.weight"][:dim])
            model.norm.bias.data.copy_(state["encoder.ln.bias"][:dim])

        model.is_pretrained = True
        model.checkpoint_provenance = provenance
        model.checkpoint_hash = hashlib.sha256(model.patch_embed.proj.weight.data.cpu().numpy().tobytes()).hexdigest()[:16]
        model.last_load_error = None
        return True
    except Exception as exc:
        model.is_pretrained = False
        model.checkpoint_provenance = "uninitialized_scratch_weights"
        model.checkpoint_hash = None
        model.last_load_error = str(exc)
        if strict:
            raise RuntimeError(f"Failed to load pretrained ViT weights: {exc}") from exc
        logger.warning("Failed to load pretrained ViT weights: %s", exc)
        return False


# Backward compatibility alias
load_pretrained_vit_b_weights = load_pretrained_vit_weights


def create_medical_vit(
    model_type: str = "vit_tiny",
    img_size: int = 28,
    patch_size: int = 4,
    in_channels: int = 1,
    num_classes: int = 3,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    strict_checkpoint: bool = False,
) -> VisionTransformer:
    """Factory helper to build scaled Vision Transformers suited for medical benchmarks."""
    configs = {
        "vit_tiny": {"embed_dim": 128, "depth": 4, "num_heads": 4},
        "vit_small": {"embed_dim": 256, "depth": 6, "num_heads": 8},
        "vit_base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    }
    cfg = configs.get(model_type, configs["vit_tiny"])
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
    )
    model.model_type = model_type
    model.strict_checkpoint = strict_checkpoint
    if pretrained:
        success = load_pretrained_vit_weights(
            model,
            checkpoint_path=checkpoint_path,
            strict=strict_checkpoint,
        )
        if not success and strict_checkpoint:
            raise RuntimeError(
                f"Failed to initialize {model_type} with authenticated pretrained weights in strict mode."
            )
    return model
