"""
Fed-XRay Hospital Client Implementation
========================================
Implements local training, differential data partitioning, and attack simulations.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple

from ..models.cnn import create_model
from .metrics import TrainingMetrics


class HospitalClient:
    """
    Hospital client node with standard training and optional adversarial simulation.
    
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
        self.criterion = nn.CrossEntropyLoss()
        self.n_samples = len(dataloader.dataset) if dataloader and hasattr(dataloader, 'dataset') and dataloader.dataset else 0
    
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
    
    def train(
        self, 
        global_weights: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], TrainingMetrics]:
        """Perform local training on private client dataloader."""
        self.model.load_state_dict(copy.deepcopy(global_weights))
        self.model.train()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
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
                loss = self.criterion(outputs, labels)
                
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
