"""
Fed-XRay: CNN Model for X-Ray Classification
=============================================
This module defines the neural network architecture for classifying
lung X-Ray images into Normal, Pneumonia, or COVID-19 categories.

Key Concept (Federated Learning):
---------------------------------
In FL, this same model architecture is shared across all hospital clients.
Each hospital trains a local copy on their private data, then sends only
the model WEIGHTS (not the data!) back to the central server.
This preserves patient privacy while enabling collaborative learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, OrderedDict


class XRayClassifier(nn.Module):
    """
    Convolutional Neural Network for X-Ray image classification.
    
    Architecture:
    - 2 Convolutional layers with ReLU activation and MaxPooling
    - Dropout for regularization (crucial in medical AI to prevent overconfidence)
    - 2 Fully connected layers for final classification
    
    Input: 28x28 grayscale images (1 channel)
    Output: 3 classes (Normal, Pneumonia, COVID-19)
    
    This architecture is intentionally lightweight to:
    1. Enable fast training on each hospital's local device
    2. Minimize communication overhead in FL (smaller model = faster sync)
    3. Still capture essential visual patterns in medical images
    """
    
    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.5):
        """
        Initialize the XRayClassifier.
        
        Args:
            num_classes: Number of output classes (default 3)
            dropout_rate: Dropout probability for regularization
        """
        super(XRayClassifier, self).__init__()
        
        # ===== Convolutional Feature Extractor =====
        # These layers learn to detect patterns like edges, textures,
        # and consolidation patterns relevant to lung diseases
        
        # First conv block: 1 input channel -> 32 filters
        # Learns low-level features (edges, basic textures)
        self.conv1 = nn.Conv2d(
            in_channels=1,      # Grayscale X-Ray
            out_channels=32,    # 32 feature maps
            kernel_size=3,      # 3x3 filter
            padding=1           # Same padding
        )
        self.bn1 = nn.BatchNorm2d(32)  # Stabilizes training
        self.pool1 = nn.MaxPool2d(2, 2)  # Downsample: 28x28 -> 14x14
        
        # Second conv block: 32 -> 64 filters
        # Learns higher-level features (opacity patterns, consolidations)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Downsample: 14x14 -> 7x7
        
        # ===== Classifier Head =====
        # Dropout helps prevent overfitting - especially important in 
        # medical AI where training data may be limited
        self.dropout = nn.Dropout(dropout_rate)
        
        # After conv layers: 64 channels * 7 * 7 = 3136 features
        # (assuming 28x28 input with two 2x2 max pools)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Track number of classes for logging
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # First convolutional block
        x = self.conv1(x)           # -> (batch, 32, 28, 28)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)           # -> (batch, 32, 14, 14)
        
        # Second convolutional block
        x = self.conv2(x)           # -> (batch, 64, 14, 14)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)           # -> (batch, 64, 7, 7)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)   # -> (batch, 3136)
        
        # Apply dropout before classification
        x = self.dropout(x)
        
        # Fully connected classification head
        x = F.relu(self.fc1(x))     # -> (batch, 128)
        x = self.dropout(x)
        x = self.fc2(x)             # -> (batch, num_classes)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities (softmax output).
        
        Args:
            x: Input tensor of shape (batch, 1, 28, 28)
            
        Returns:
            Probability distribution over classes
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get model weights as a dictionary.
        
        FL Concept: This is what gets sent from hospitals to the central server.
        Only the model parameters are transmitted - NOT the training data!
        This is the key privacy-preserving aspect of Federated Learning.
        
        Returns:
            Dictionary of parameter names to tensors
        """
        return {name: param.clone().detach() 
                for name, param in self.state_dict().items()}
    
    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """
        Set model weights from a dictionary.
        
        FL Concept: After aggregation, hospitals receive the averaged
        global model weights to continue training in the next round.
        
        Args:
            weights: Dictionary of parameter names to tensors
        """
        self.load_state_dict(weights)


def create_model(num_classes: int = 3, dropout_rate: float = 0.5) -> XRayClassifier:
    """
    Factory function to create a new XRayClassifier model.
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        
    Returns:
        Initialized XRayClassifier model
    """
    return XRayClassifier(num_classes=num_classes, dropout_rate=dropout_rate)


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters.
    
    Useful for understanding communication costs in FL -
    more parameters = larger model updates to transmit.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
