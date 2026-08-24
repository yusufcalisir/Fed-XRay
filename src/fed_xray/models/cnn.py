"""
Fed-XRay CNN Model Architecture
================================
Convolutional Neural Network for lung X-Ray / Medical Image classification
into Normal, Pneumonia, and COVID-19 categories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, OrderedDict


class XRayClassifier(nn.Module):
    """
    Convolutional Neural Network for X-Ray image classification.
    
    Architecture (Simplified for Stability):
    - 2 Convolutional layers with ReLU activation and MaxPooling
    - NO BatchNorm (removed for FL stability under extreme non-IID partitions)
    - Dropout for regularization
    - 2 Fully connected layers for final classification
    
    Input: 28x28 grayscale images (1 channel)
    Output: 3 classes (Normal, Pneumonia, COVID-19)
    """
    
    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.3):
        """
        Initialize the XRayClassifier.
        
        Args:
            num_classes: Number of output classes (default 3)
            dropout_rate: Dropout probability for regularization
        """
        super(XRayClassifier, self).__init__()
        
        # ===== Convolutional Feature Extractor =====
        self.conv1 = nn.Conv2d(
            in_channels=1,      # Grayscale X-Ray
            out_channels=32,    # 32 feature maps
            kernel_size=3,      # 3x3 filter
            padding=1           # Same padding
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # Downsample: 28x28 -> 14x14
        
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # Downsample: 14x14 -> 7x7
        
        # ===== Classifier Head =====
        self.dropout = nn.Dropout(dropout_rate)
        
        # After conv layers: 64 channels * 7 * 7 = 3136 features
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.num_classes = num_classes
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def check_valid(self) -> bool:
        """Check if model parameters are valid (no NaNs or Infs)."""
        for name, param in self.state_dict().items():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"[ERROR] Parameter {name} contains NaN/Inf!")
                return False
        return True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning output logits of shape (batch, num_classes)."""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities (softmax output)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get model weights as a dictionary of detached cloned tensors."""
        return {name: param.clone().detach() 
                for name, param in self.state_dict().items()}
    
    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Set model weights from a dictionary."""
        self.load_state_dict(weights)


def create_model(num_classes: int = 3, dropout_rate: float = 0.3) -> XRayClassifier:
    """Factory function to create an initialized XRayClassifier model."""
    return XRayClassifier(num_classes=num_classes, dropout_rate=dropout_rate)


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
