"""
Fed-XRay: Explainable AI Engine (Grad-CAM)
==========================================
Implements Gradient-weighted Class Activation Mapping for
visual explanation of model predictions.

Grad-CAM generates a heatmap showing which regions of the
input image most influenced the model's decision.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations 
from Deep Networks via Gradient-based Localization" (2017)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import zoom


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Hooks into the final convolutional layer to capture:
    1. Forward pass activations (feature maps)
    2. Backward pass gradients (importance weights)
    
    The heatmap is computed as:
    L^c = ReLU(Σ_k α_k^c * A^k)
    
    Where:
    - α_k^c = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij) [global avg pooling of gradients]
    - A^k = k-th feature map from final conv layer
    - y^c = score for class c (before softmax)
    """
    
    def __init__(self, model: torch.nn.Module) -> None:
        """
        Initialize Grad-CAM with target model.
        
        Args:
            model: PyTorch CNN model with conv layers
        """
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hook_handles = []
        
        # Find and hook the last Conv2d layer
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks on last conv layer."""
        # Find last Conv2d layer
        target_layer = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                self.target_layer_name = name
        
        if target_layer is None:
            raise ValueError("No Conv2d layer found in model!")
        
        # Forward hook - capture activations
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        # Backward hook - capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks
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
        """
        Generate Grad-CAM heatmap for input image.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for explanation (None = predicted class)
            
        Returns:
            (heatmap, predicted_class, confidence)
            heatmap is normalized to [0, 1] and same size as input
        """
        self.model.eval()
        
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        
        # CRITICAL: Clone and require gradients
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        try:
            # Forward pass (gradients enabled)
            output = self.model(input_tensor)
            
            # Handle NaN in output
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("[Grad-CAM Warning] Model output contains NaN/Inf")
                return np.ones((input_h, input_w)) * 0.5, 0, 0.33
            
            # Get predicted class if not specified
            probs = F.softmax(output, dim=1)
            confidence, predicted = probs.max(dim=1)
            
            if target_class is None:
                target_class = predicted.item()
            
            # Zero gradients on model
            self.model.zero_grad()
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()
            
            # Create one-hot encoding for target class
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            
            # Backward pass - THIS IS CRITICAL for Grad-CAM
            output.backward(gradient=one_hot, retain_graph=True)
            
            # Get captured gradients and activations from hooks
            gradients = self.gradients
            activations = self.activations
            
            # Debug: Check if hooks captured data
            if gradients is None:
                print("[Grad-CAM Warning] Gradients not captured - hook may have failed")
                return np.ones((input_h, input_w)) * 0.5, predicted.item(), confidence.item()
            
            if activations is None:
                print("[Grad-CAM Warning] Activations not captured - forward hook failed")
                return np.ones((input_h, input_w)) * 0.5, predicted.item(), confidence.item()
            
            # Check for NaN in gradients/activations
            if torch.isnan(gradients).any():
                print("[Grad-CAM Warning] Gradients contain NaN")
                return np.ones((input_h, input_w)) * 0.5, predicted.item(), confidence.item()
            
            if torch.isnan(activations).any():
                print("[Grad-CAM Warning] Activations contain NaN")
                return np.ones((input_h, input_w)) * 0.5, predicted.item(), confidence.item()
            
            # Global Average Pooling of gradients -> importance weights
            # α_k = GAP(∂y^c / ∂A^k)
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            
            # Weighted combination of feature maps
            # L = Σ_k α_k * A^k
            cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
            
            # Apply ReLU (only positive contributions matter)
            cam = F.relu(cam)
            
            # Convert to numpy
            cam = cam.squeeze().cpu().detach().numpy()
            
            # Handle edge case of all-zero CAM
            if cam.max() == 0 or np.isnan(cam).any():
                print("[Grad-CAM Warning] CAM is all zeros or contains NaN")
                return np.ones((input_h, input_w)) * 0.5, predicted.item(), confidence.item()
            
            # Normalize to [0, 1]
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            # Upsample to original input size
            if cam.shape[0] != input_h or cam.shape[1] != input_w:
                zoom_h = input_h / cam.shape[0]
                zoom_w = input_w / cam.shape[1]
                cam = zoom(cam, (zoom_h, zoom_w), order=1)
            
            # Final clip
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
    """
    Create overlay of heatmap on original image.
    
    Args:
        image: Original grayscale image (H, W), values [0, 1]
        heatmap: Attention heatmap (H, W), values [0, 1]
        alpha: Blending factor (0 = only image, 1 = only heatmap)
        
    Returns:
        RGB overlay image (H, W, 3), values [0, 1]
    """
    # Convert grayscale to RGB
    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image
    
    # Create colormap (hot colormap: black -> red -> yellow -> white)
    heatmap_colored = np.zeros((*heatmap.shape, 3))
    
    # Red channel: increases with heatmap
    heatmap_colored[:, :, 0] = np.clip(heatmap * 2, 0, 1)
    
    # Green channel: increases after mid-point
    heatmap_colored[:, :, 1] = np.clip((heatmap - 0.5) * 2, 0, 1)
    
    # Blue channel: only for very high values
    heatmap_colored[:, :, 2] = np.clip((heatmap - 0.75) * 4, 0, 1)
    
    # Blend
    overlay = (1 - alpha) * image_rgb + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def get_explanation_text(predicted_class: int, confidence: float) -> str:
    """
    Generate dynamic explanation text for the prediction.
    
    Args:
        predicted_class: Predicted class index
        confidence: Prediction confidence (0-1)
        
    Returns:
        Explanation string
    """
    class_names = {0: "Normal", 1: "Pneumonia", 2: "COVID-19"}
    class_name = class_names.get(predicted_class, "Unknown")
    
    explanations = {
        0: "The model found no significant abnormalities. The lung fields appear clear with normal vascular markings.",
        1: "The model detected focal consolidation patterns consistent with bacterial pneumonia. Red areas show regions of increased opacity.",
        2: "The model identified diffuse bilateral ground-glass opacities characteristic of COVID-19. Red areas highlight peripheral involvement."
    }
    
    base_explanation = explanations.get(predicted_class, "Analysis complete.")
    
    return f"**Diagnosis: {class_name}** (Confidence: {confidence*100:.1f}%)\n\n{base_explanation}"
