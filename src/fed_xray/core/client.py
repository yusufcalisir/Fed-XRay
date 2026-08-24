"""
Fed-XRay Hospital Client Implementation
========================================
Implements local training, differential data partitioning, adversarial attack simulations,
and advanced optimization algorithms (FedAvg, FedProx, SCAFFOLD, FedDyn, MOON).
"""

import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any

from ..models.cnn import create_model
from .metrics import TrainingMetrics
from .algorithms import (
    compute_fedprox_loss, 
    compute_feddyn_loss, 
    compute_moon_contrastive_loss
)


class HospitalClient:
    """
    Hospital client node with multi-algorithm training and adversarial simulation.
    
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
        self.prev_model: Optional[nn.Module] = None
        self.prev_grads: Dict[str, torch.Tensor] = {}
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
        global_weights: Dict[str, torch.Tensor],
        algorithm: str = "FedAvg",
        mu: float = 0.01,
        alpha: float = 0.01,
        temperature: float = 0.5,
        server_control: Optional[Dict[str, torch.Tensor]] = None,
        client_control: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], TrainingMetrics, Optional[Dict[str, torch.Tensor]]]:
        """
        Perform local training with selected federated optimization algorithm.
        
        Args:
            global_weights: Current global model parameters
            algorithm: One of 'FedAvg', 'FedProx', 'SCAFFOLD', 'FedDyn', 'MOON'
            mu: Proximal / contrastive parameter for FedProx / MOON
            alpha: Dynamic penalty for FedDyn
            temperature: MOON contrastive temperature
            server_control: Global control variate c for SCAFFOLD
            client_control: Local control variate c_k for SCAFFOLD
            
        Returns:
            Tuple of (updated_weights, training_metrics, updated_client_control_delta)
        """
        # Save previous local model for MOON
        if algorithm == "MOON" and self.prev_model is None:
            self.prev_model = create_model().to(self.device)
            self.prev_model.load_state_dict(copy.deepcopy(global_weights))
            
        # Global reference model for MOON
        global_model_ref = None
        if algorithm == "MOON":
            global_model_ref = create_model().to(self.device)
            global_model_ref.load_state_dict(copy.deepcopy(global_weights))
            global_model_ref.eval()
            
        self.model.load_state_dict(copy.deepcopy(global_weights))
        self.model.train()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        step_count = 0
        
        for epoch in range(self.local_epochs):
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if self.malicious:
                    labels = self._flip_labels(labels)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Apply Algorithm-Specific Regularization
                if algorithm == "FedProx":
                    loss = loss + compute_fedprox_loss(self.model, global_weights, mu=mu)
                elif algorithm == "FedDyn":
                    loss = loss + compute_feddyn_loss(self.model, global_weights, self.prev_grads, alpha=alpha)
                elif algorithm == "MOON" and global_model_ref is not None:
                    with torch.no_grad():
                        z_glob = global_model_ref(images)
                        z_prev = self.prev_model(images) if self.prev_model is not None else None
                    z_loc = outputs
                    loss = loss + compute_moon_contrastive_loss(z_loc, z_glob, z_prev, temperature=temperature, mu=mu)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                
                # SCAFFOLD Control Variate Gradient Correction
                if algorithm == "SCAFFOLD" and server_control is not None and client_control is not None:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and name in server_control and name in client_control:
                            c_s = server_control[name].to(self.device)
                            c_i = client_control[name].to(self.device)
                            param.grad.data.add_(c_s - c_i)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                step_count += 1
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total_samples += images.size(0)
        
        # Save model for next round in MOON
        if algorithm == "MOON":
            self.prev_model = create_model().to(self.device)
            self.prev_model.load_state_dict(self.model.state_dict())
            
        # Compute SCAFFOLD Control Variate Delta: delta_c_k = c_k^+ - c_k
        control_delta: Optional[Dict[str, torch.Tensor]] = None
        if algorithm == "SCAFFOLD" and server_control is not None and client_control is not None:
            control_delta = {}
            steps = max(step_count, 1)
            for name, param in self.model.named_parameters():
                if name in global_weights and name in client_control and name in server_control:
                    w_glob = global_weights[name].to(self.device)
                    c_i = client_control[name].to(self.device)
                    c_s = server_control[name].to(self.device)
                    
                    # c_i^+ = c_i - c + (w_glob - w_local) / (K * eta_l)
                    c_i_plus = c_i - c_s + (w_glob - param.data) / (steps * self.learning_rate)
                    control_delta[name] = (c_i_plus - c_i).cpu()
                    client_control[name] = c_i_plus.cpu()

        # Update FedDyn gradient history
        if algorithm == "FedDyn":
            for name, param in self.model.named_parameters():
                if name in global_weights:
                    w_glob = global_weights[name].to(self.device)
                    if name not in self.prev_grads:
                        self.prev_grads[name] = torch.zeros_like(param.data)
                    self.prev_grads[name] -= alpha * (param.data - w_glob)
        
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
        
        return updated_weights, metrics, control_delta
