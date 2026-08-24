"""
Fed-XRay Hospital Client Implementation
========================================
Implements local training, differential data partitioning, personalized prototype
learning, imbalance loss functions, and attack simulations.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any, List

from ..models.cnn import create_model
from .metrics import TrainingMetrics
from .imbalance_losses import (
    DynamicAdaptiveFocalLoss,
    BalancedSoftmaxLoss,
    ClassBalancedLoss,
    LDAMLoss,
    PrototypeRepelLoss
)
from .prototypes import (
    extract_features,
    compute_local_prototypes_and_dispersion,
    compute_prototype_distance_loss
)


class HospitalClient:
    """
    Hospital client node with standard training, personalized prototypes,
    imbalance loss functions, and optional adversarial simulation.
    
    Attack Mode: Label Flipping
    - When malicious=True, labels are flipped during training:
      Normal(0) -> Pneumonia(1) -> COVID-19(2) -> Normal(0)
    """
    
    LABEL_FLIP_MAP = {0: 1, 1: 2, 2: 0}
    
    def __init__(
        self,
        client_id: int,
        dataloader: DataLoader,
        device: torch.device = None,
        learning_rate: float = 0.001,
        local_epochs: int = 1,
        malicious: bool = False
    ) -> None:
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device or torch.device('cpu')
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.malicious = malicious
        
        self.model = create_model().to(self.device)
        self.n_samples = len(dataloader.dataset) if dataloader and hasattr(dataloader, 'dataset') and dataloader.dataset is not None else 0
        
        # Calculate local class distribution counts
        self.class_counts = self._compute_class_counts()
    
    def _compute_class_counts(self) -> List[int]:
        """Compute sample count per class from local dataloader."""
        counts = [0, 0, 0]
        if self.dataloader and hasattr(self.dataloader, 'dataset') and self.dataloader.dataset is not None:
            for _, labels in self.dataloader:
                for y in labels:
                    c = y.item()
                    if c < 3:
                        counts[c] += 1
        # Avoid all zeros
        for i in range(len(counts)):
            if counts[i] == 0:
                counts[i] = 1
        return counts
    
    def get_num_samples(self) -> int:
        """Return local training sample count."""
        return self.n_samples
    
    def is_malicious(self) -> bool:
        """Check if this client is in attack mode."""
        return self.malicious
    
    def _flip_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Perform label flipping data poisoning attack."""
        flipped = labels.clone()
        for old_label, new_label in self.LABEL_FLIP_MAP.items():
            flipped[labels == old_label] = new_label
        return flipped
    
    def _get_loss_function(
        self,
        loss_fn_name: str,
        current_round: int,
        total_rounds: int
    ) -> nn.Module:
        """Factory for empirical loss functions."""
        if loss_fn_name.upper() == "DAFL":
            return DynamicAdaptiveFocalLoss(
                class_counts=self.class_counts,
                current_round=current_round,
                total_rounds=total_rounds
            )
        elif loss_fn_name.upper() in ("BALANCED_SOFTMAX", "BSM"):
            return BalancedSoftmaxLoss(class_counts=self.class_counts)
        elif loss_fn_name.upper() in ("CLASS_BALANCED", "CB"):
            return ClassBalancedLoss(class_counts=self.class_counts)
        elif loss_fn_name.upper() == "LDAM":
            return LDAMLoss(class_counts=self.class_counts)
        else:
            return nn.CrossEntropyLoss()
    
    def train(
        self, 
        global_weights: Dict[str, torch.Tensor],
        loss_fn_name: str = "CE",
        current_round: int = 1,
        total_rounds: int = 10,
        global_prototypes: Optional[Dict[int, torch.Tensor]] = None,
        proto_weight: float = 0.0,
        prox_mu: float = 0.0
    ) -> Tuple[Dict[str, torch.Tensor], TrainingMetrics]:
        """Perform local training on private client dataloader."""
        self.model.load_state_dict(copy.deepcopy(global_weights))
        self.model.train()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = self._get_loss_function(loss_fn_name, current_round, total_rounds)
        repel_criterion = PrototypeRepelLoss(margin=1.0)
        
        # Keep copy of initial global weights for FedProx proximal term
        initial_weights = {k: v.clone().detach().to(self.device) for k, v in global_weights.items()} if prox_mu > 0 else None
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if self.malicious:
                    labels = self._flip_labels(labels)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Prototype metric alignment loss
                if global_prototypes is not None and proto_weight > 0:
                    features = extract_features(self.model, images)
                    p_loss = compute_prototype_distance_loss(features, labels, global_prototypes)
                    r_loss = repel_criterion(features, labels, global_prototypes)
                    loss = loss + proto_weight * (p_loss + 0.1 * r_loss)
                
                # FedProx proximal term
                if prox_mu > 0 and initial_weights is not None:
                    prox_term = 0.0
                    for name, param in self.model.named_parameters():
                        if name in initial_weights:
                            prox_term += torch.sum((param - initial_weights[name]) ** 2)
                    loss = loss + (prox_mu / 2.0) * prox_term
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total_samples += images.size(0)
        
        avg_loss = total_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)
        
        if math.isnan(avg_loss) or math.isinf(avg_loss):
            avg_loss = 100.0
            
        is_valid = True
        for name, tensor in self.model.state_dict().items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"[ERROR] Client {self.client_id}: Found NaN in {name}")
                is_valid = False
                break
        
        if is_valid:
            updated_weights = {
                name: param.clone().detach().cpu()
                for name, param in self.model.state_dict().items()
            }
        else:
            print(f"[WARNING] Client {self.client_id} produced NaN weights. Discarding update.")
            updated_weights = copy.deepcopy(global_weights)
        
        metrics = TrainingMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            samples_trained=total_samples,
            client_id=self.client_id,
            is_malicious=self.malicious
        )
        
        return updated_weights, metrics
    
    def compute_prototypes_and_dispersion(self) -> Tuple[Dict[int, torch.Tensor], Dict[int, float], Dict[int, int]]:
        """Extract empirical class prototypes and covariance traces from local data."""
        return compute_local_prototypes_and_dispersion(
            model=self.model,
            dataloader=self.dataloader,
            device=self.device,
            num_classes=3
        )
