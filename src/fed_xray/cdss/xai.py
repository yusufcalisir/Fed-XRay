"""Fed-XRay Explainable AI Engine (Grad-CAM & Attention Saliency).

Implements:
1. Gradient-weighted Class Activation Mapping (Grad-CAM) for CNNs and Vision Transformers.
2. Saliency Heatmap generation and normalization.
3. Multimodal overlay blending and clinical findings text generation.
"""

from __future__ import annotations
import math
from typing import Any, List, Optional, Sequence, Tuple
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
        """Register forward and backward hooks on the last feature extraction layer."""
        target_layer: Optional[nn.Module] = None
        # Look for last Conv2d layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
                self.target_layer_name = name

        # If no Conv2d found (e.g. Pure ViT without Conv patch embed), hook on last transformer block norm or head
        if target_layer is None:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.LayerNorm) and "norm" in name:
                    target_layer = module
                    self.target_layer_name = name

        if target_layer is None:
            # Fallback to model root
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
                cam_arr = zoom(cam_arr, (zoom_h, zoom_w), order=1)

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
) -> str:
    """Summarize only quantities measured from the model output and Grad-CAM map.

    No morphological or anatomical claim is made that the model did not compute.
    ``confidence`` and ``probabilities`` are on the [0, 1] scale.
    """
    class_names = {0: "Normal", 1: "Pneumonia", 2: "COVID-19"}
    lines = [
        f"**Model prediction: {class_names.get(predicted_class, 'Unknown')}** "
        f"(softmax confidence: {confidence * 100:.1f}%)"
    ]

    if probabilities is not None and len(probabilities) > 1:
        ranked = sorted(range(len(probabilities)), key=lambda k: -float(probabilities[k]))
        runner = ranked[1]
        margin = (float(probabilities[ranked[0]]) - float(probabilities[runner])) * 100.0
        lines.append(
            f"Runner-up class: {class_names.get(runner, 'Unknown')} "
            f"({float(probabilities[runner]) * 100:.1f}%), margin {margin:.1f} percentage points."
        )

    if heatmap is not None:
        h = np.asarray(heatmap, dtype=np.float32)
        rows, cols = h.shape
        peak_r, peak_c = np.unravel_index(int(np.argmax(h)), h.shape)
        vertical = "upper" if peak_r < rows / 3 else ("lower" if peak_r >= 2 * rows / 3 else "middle")
        horizontal = "left" if peak_c < cols / 3 else ("right" if peak_c >= 2 * cols / 3 else "center")
        active_pct = float((h >= 0.5).mean() * 100.0)
        lines.append(
            f"Grad-CAM peak relevance lies in the {vertical}-{horizontal} part of the image frame; "
            f"{active_pct:.1f}% of pixels exceed 0.5 normalized relevance."
        )

    lines.append(
        "Decision-support output of a research model. It is not a clinical diagnosis and "
        "requires radiologist review."
    )
    return "\n\n".join(lines)
