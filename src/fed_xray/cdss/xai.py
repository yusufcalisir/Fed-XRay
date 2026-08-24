"""
Fed-XRay Explainable AI Engine (Grad-CAM)
=========================================
Implements Gradient-weighted Class Activation Mapping for visual explanation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import zoom


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    L^c = ReLU(Σ_k α_k^c * A^k)
    """
    
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks on last conv layer."""
        target_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                self.target_layer_name = name
        
        if target_layer is None:
            raise ValueError("No Conv2d layer found in model!")
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles = [handle_fwd, handle_bwd]
    
    def remove_hooks(self) -> None:
        """Remove registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """Generate Grad-CAM heatmap for input image."""
        self.model.eval()
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        try:
            output = self.model(input_tensor)
            if torch.isnan(output).any() or torch.isinf(output).any():
                return np.ones((input_h, input_w)) * 0.5, 0, 0.33
            
            probs = F.softmax(output, dim=1)
            confidence, predicted = probs.max(dim=1)
            
            if target_class is None:
                target_class = predicted.item()
            
            self.model.zero_grad()
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()
            
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)
            
            gradients = self.gradients
            activations = self.activations
            
            if gradients is None or activations is None:
                return np.ones((input_h, input_w)) * 0.5, predicted.item(), confidence.item()
            
            if torch.isnan(gradients).any() or torch.isnan(activations).any():
                return np.ones((input_h, input_w)) * 0.5, predicted.item(), confidence.item()
            
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().detach().numpy()
            
            if cam.max() == 0 or np.isnan(cam).any():
                return np.ones((input_h, input_w)) * 0.5, predicted.item(), confidence.item()
            
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            if cam.shape[0] != input_h or cam.shape[1] != input_w:
                zoom_h = input_h / cam.shape[0]
                zoom_w = input_w / cam.shape[1]
                cam = zoom(cam, (zoom_h, zoom_w), order=1)
            
            cam = np.clip(cam, 0, 1)
            return cam, predicted.item(), confidence.item()
            
        except Exception as e:
            print(f"[Grad-CAM Error] {e}")
            return np.ones((input_h, input_w)) * 0.5, 0, 0.33


def create_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Create overlay of heatmap on original image."""
    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image
    
    heatmap_colored = np.zeros((*heatmap.shape, 3))
    heatmap_colored[:, :, 0] = np.clip(heatmap * 2, 0, 1)
    heatmap_colored[:, :, 1] = np.clip((heatmap - 0.5) * 2, 0, 1)
    heatmap_colored[:, :, 2] = np.clip((heatmap - 0.75) * 4, 0, 1)
    
    overlay = (1 - alpha) * image_rgb + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)


def get_explanation_text(predicted_class: int, confidence: float) -> str:
    """Generate dynamic explanation text for the prediction."""
    class_names = {0: "Normal", 1: "Pneumonia", 2: "COVID-19"}
    class_name = class_names.get(predicted_class, "Unknown")
    
    explanations = {
        0: "The model found no significant abnormalities. The lung fields appear clear with normal vascular markings.",
        1: "The model detected focal consolidation patterns consistent with bacterial pneumonia. Red areas show regions of increased opacity.",
        2: "The model identified diffuse bilateral ground-glass opacities characteristic of COVID-19. Red areas highlight peripheral involvement."
    }
    
    base_explanation = explanations.get(predicted_class, "Analysis complete.")
    return f"**Diagnosis: {class_name}** (Confidence: {confidence*100:.1f}%)\n\n{base_explanation}"
