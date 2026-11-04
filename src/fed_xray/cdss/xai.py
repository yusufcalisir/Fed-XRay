"""Fed-XRay Explainable AI Engine (Grad-CAM & Attention Saliency).

Implements:
1. Gradient-weighted Class Activation Mapping (Grad-CAM) for CNNs and Vision Transformers.
2. Saliency Heatmap generation and normalization.
3. Multimodal overlay blending and clinical findings text generation.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.ndimage import zoom
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMUnavailableError(RuntimeError):
    """Raised when a genuine Grad-CAM map cannot be computed for the given input."""


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNNs and Vision Transformers:

    L^c = ReLU(sum_k alpha_k^c * A^k)
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hook_handles: List[Any] = []
        self.target_layer_name = ""
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the last feature extraction layer.

        For Vision Transformers (ViT), hooks on the deepest TransformerEncoderBlock
        to extract contextual patch-token representations rather than shallow patch-embed convs.
        For CNNs, hooks on the last Conv2d layer.
        """
        target_layer: Optional[nn.Module] = None

        # 1. Vision Transformer: hook on last encoder block
        if hasattr(self.model, "blocks") and len(self.model.blocks) > 0:
            target_layer = self.model.blocks[-1]
            self.target_layer_name = f"blocks.{len(self.model.blocks) - 1}"
        else:
            last_block_module = None
            last_block_name = ""
            for name, module in self.model.named_modules():
                if "TransformerEncoderBlock" in module.__class__.__name__ or ".blocks." in name:
                    last_block_module = module
                    last_block_name = name
            if last_block_module is not None:
                target_layer = last_block_module
                self.target_layer_name = last_block_name

        # 2. CNN fallback: look for last Conv2d layer
        if target_layer is None:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    self.target_layer_name = name

        # 3. LayerNorm fallback
        if target_layer is None:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.LayerNorm) and "norm" in name:
                    target_layer = module
                    self.target_layer_name = name

        if target_layer is None:
            target_layer = self.model

        def forward_hook(module: nn.Module, input: Any, output: Any) -> None:
            self.activations = output.detach() if isinstance(output, torch.Tensor) else output[0].detach()

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Any) -> None:
            self.gradients = grad_output[0].detach()

        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles = [handle_fwd, handle_bwd]

    def remove_hooks(self) -> None:
        """Remove registered PyTorch forward/backward hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Generate Grad-CAM saliency heatmap for input image."""
        self.model.eval()
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        try:
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]

            if torch.isnan(output).any() or torch.isinf(output).any():
                raise GradCAMUnavailableError("Model produced non-finite logits for this input")

            probs = F.softmax(output, dim=1)
            confidence, predicted = probs.max(dim=1)

            pred_class = int(predicted.item())
            conf_score = float(confidence.item())
            chosen_class = pred_class if target_class is None else target_class

            self.model.zero_grad()
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()

            one_hot = torch.zeros_like(output)
            one_hot[0, chosen_class] = 1.0
            output.backward(gradient=one_hot, retain_graph=True)

            gradients = self.gradients
            activations = self.activations

            if gradients is None or activations is None:
                # Gradient-input fallback
                if input_tensor.grad is not None:
                    grad_input = input_tensor.grad.abs().mean(dim=1).squeeze().cpu().numpy()
                    grad_input = (grad_input - grad_input.min()) / (grad_input.max() - grad_input.min() + 1e-8)
                    return grad_input.astype(np.float32), pred_class, conf_score
                raise GradCAMUnavailableError("No gradients or activations were captured")

            cam_arr: np.ndarray
            if gradients.dim() == 4 and activations.dim() == 4:
                # CNN / Conv2d patch embedding formulation
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam_t = (weights * activations).sum(dim=1, keepdim=True)
                cam_t_relu = F.relu(cam_t)
                if cam_t_relu.max() > cam_t_relu.min():
                    cam_arr = cam_t_relu.squeeze().cpu().detach().numpy()
                else:
                    cam_arr = torch.abs(cam_t).squeeze().cpu().detach().numpy()
            elif gradients.dim() == 3 and activations.dim() == 3:
                # Token sequence formulation [B, N, D]
                weights = gradients.mean(dim=1, keepdim=True)
                cam_seq = (weights * activations).sum(dim=-1).squeeze().cpu().detach().numpy()
                # Exclude CLS token if present
                if len(cam_seq) > 1:
                    cam_seq = cam_seq[1:]
                cam_seq_relu = np.maximum(cam_seq, 0.0)
                if cam_seq_relu.max() > cam_seq_relu.min():
                    cam_seq = cam_seq_relu
                else:
                    cam_seq = np.abs(cam_seq)
                grid_side = math.isqrt(len(cam_seq))
                if grid_side * grid_side == len(cam_seq):
                    cam_arr = cam_seq.reshape(grid_side, grid_side)
                else:
                    raise GradCAMUnavailableError(
                        f"Token grid of length {len(cam_seq)} is not a square patch grid"
                    )
            else:
                raise GradCAMUnavailableError(
                    f"Unsupported activation rank (gradients {gradients.dim()}D, activations {activations.dim()}D)"
                )

            if not np.isfinite(cam_arr).all() or cam_arr.max() == cam_arr.min():
                raise GradCAMUnavailableError("Class activation map is degenerate (constant or non-finite)")

            cam_arr = (cam_arr - cam_arr.min()) / (cam_arr.max() - cam_arr.min() + 1e-8)

            if cam_arr.shape[0] != input_h or cam_arr.shape[1] != input_w:
                zoom_h = input_h / cam_arr.shape[0]
                zoom_w = input_w / cam_arr.shape[1]
                cam_arr = zoom(cam_arr, (zoom_h, zoom_w), order=2)

            cam_final = np.clip(cam_arr, 0.0, 1.0).astype(np.float32)
            return cam_final, pred_class, conf_score

        except GradCAMUnavailableError:
            raise
        except Exception as exc:
            raise GradCAMUnavailableError(f"Grad-CAM computation failed: {exc}") from exc


def create_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend heatmap with grayscale or RGB image."""
    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image

    heatmap_colored = np.zeros((*heatmap.shape, 3))
    heatmap_colored[:, :, 0] = np.clip(heatmap * 2, 0, 1)
    heatmap_colored[:, :, 1] = np.clip((heatmap - 0.5) * 2, 0, 1)
    heatmap_colored[:, :, 2] = np.clip((heatmap - 0.75) * 4, 0, 1)

    overlay = (1 - alpha) * image_rgb + alpha * heatmap_colored
    return np.clip(overlay, 0.0, 1.0)


def get_explanation_text(
    predicted_class: int,
    confidence: float,
    probabilities: Optional[Sequence[float]] = None,
    heatmap: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
    language: str = "en",
) -> str:
    """Summarize only quantities measured from the model output and Grad-CAM map.

    No morphological or anatomical claim is made that the model did not compute.
    ``confidence`` and ``probabilities`` are on the [0, 1] scale.
    """
    labels = class_names if class_names is not None else {0: "Normal", 1: "Pneumonia", 2: "COVID-19"}
    is_tr = (language or "en").lower().startswith("tr")

    tr_labels = {
        "Normal": "Normal Parankim",
        "Pneumonia": "Pnömoni (Konsolidasyon)",
        "COVID-19": "COVID-19 (Buzlu Cam)",
        "Benign Dysplastic Nevus": "Benign Displastik Nevüs",
        "Malignant Melanoma": "Malign Melanom",
        "Basal Cell Carcinoma": "Bazal Hücreli Karsinom",
        "Actinic Keratosis": "Aktinik Keratoz",
        "Benign Keratosis (Solar Lentigo)": "Benign Keratoz (Solar Lentigo)",
        "Dermatofibroma": "Dermatofibrom",
        "Vascular Lesion": "Vasküler Lezyon",
        "Squamous Cell Carcinoma": "Skuamöz Hücreli Karsinom",
        "Adipose Tissue": "Yağ Dokusu",
        "Background": "Arka Plan",
        "Debris": "Nekrotik Doku / Debris",
        "Immune Lymphocytes": "İmmün Lenfositler",
        "Mucus": "Mukus",
        "Smooth Muscle": "Düz Kas Dokusu",
        "Normal Colon Mucosa": "Normal Kolon Mukozası",
        "Cancer-Associated Stroma": "Kanserle İlişkili Stroma",
        "Colorectal Adenocarcinoma Epithelium": "Kolorektal Adenokarsinom Epiteli",
    }

    pred_name = labels.get(predicted_class, "Unknown")
    pred_display = tr_labels.get(pred_name, pred_name) if is_tr else pred_name

    if is_tr:
        lines = [
            f"Model tahmini: {pred_display} "
            f"(softmax güven skoru: %{confidence * 100:.1f})"
        ]
    else:
        lines = [
            f"Model prediction: {pred_display} "
            f"(softmax confidence: {confidence * 100:.1f}%)"
        ]

    if probabilities is not None and len(probabilities) > 1:
        ranked = sorted(range(len(probabilities)), key=lambda k: -float(probabilities[k]))
        runner = ranked[1]
        margin = (float(probabilities[ranked[0]]) - float(probabilities[runner])) * 100.0
        runner_name = labels.get(runner, "Unknown")
        runner_display = tr_labels.get(runner_name, runner_name) if is_tr else runner_name
        if is_tr:
            lines.append(
                f"İkinci olası sınıf: {runner_display} "
                f"(%{float(probabilities[runner]) * 100:.1f}), fark: {margin:.1f} yüzdelik puan."
            )
        else:
            lines.append(
                f"Runner-up class: {runner_display} "
                f"({float(probabilities[runner]) * 100:.1f}%), margin {margin:.1f} percentage points."
            )

    if heatmap is not None:
        h = np.asarray(heatmap, dtype=np.float32)
        rows, cols = h.shape
        peak_r, peak_c = np.unravel_index(int(np.argmax(h)), h.shape)
        vertical = "upper" if peak_r < rows / 3 else ("lower" if peak_r >= 2 * rows / 3 else "middle")
        horizontal = "left" if peak_c < cols / 3 else ("right" if peak_c >= 2 * cols / 3 else "center")
        active_pct = float((h >= 0.5).mean() * 100.0)
        if is_tr:
            v_tr = {"upper": "üst", "middle": "orta", "lower": "alt"}.get(vertical, vertical)
            h_tr = {"left": "sol", "center": "merkez", "right": "sağ"}.get(horizontal, horizontal)
            lines.append(
                f"Grad-CAM aktivasyon odağı görüntünün {v_tr}-{h_tr} bölgesinde yer almaktadır; "
                f"piksellerin %{active_pct:.1f}'i 0.5 normalize eşiği aşmaktadır."
            )
        else:
            lines.append(
                f"Grad-CAM peak relevance lies in the {vertical}-{horizontal} part of the image frame; "
                f"{active_pct:.1f}% of pixels exceed 0.5 normalized relevance."
            )

    is_derm_case = any("melanom" in str(v).lower() or "carcinoma" in str(v).lower() or "nevus" in str(v).lower() for v in labels.values())
    if is_tr:
        specialist_tr = "uzman hekim veya patoloji incelemesi" if is_derm_case else "uzman hekim veya radyolog incelemesi"
        lines.append(
            f"Araştırma modeli klinik karar destek çıktısıdır. Kesin klinik tanı niteliği taşımaz ve {specialist_tr} gerektirir."
        )
    else:
        specialist_en = "specialist physician and pathology review" if is_derm_case else "specialist physician or radiologist review"
        lines.append(
            f"Decision-support output of a research model. It is not a clinical diagnosis and requires {specialist_en}."
        )
    return "\n\n".join(lines)
