"""
Fed-XRay: Adversarial Federated Learning Core (Level 3 Security)
==================================================================
Implements:
1. Rigorous FedAvg (McMahan et al., 2017)
2. Label Flipping Attack simulation (malicious clients)
3. Validation-based defense mechanism (Byzantine-robust aggregation)

Security Features:
- Malicious client detection via validation accuracy
- Filtering of poisoned model updates
- Trusted validation strategy using global hold-out set
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    accuracy_score
)

from .cnn_model import XRayClassifier, create_model


@dataclass
class TrainingMetrics:
    """Metrics from a single client's local training."""
    loss: float
    accuracy: float
    samples_trained: int
    client_id: int = 0
    is_malicious: bool = False
    was_blocked: bool = False


@dataclass 
class EvaluationMetrics:
    """Comprehensive evaluation metrics for medical AI."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    loss: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'loss': self.loss
        }


@dataclass
class SecurityReport:
    """
    Security report from defense mechanism.
    
    Tracks:
    - Which clients were evaluated
    - Which were detected as malicious
    - Which were blocked from aggregation
    """
    total_clients: int
    malicious_detected: List[int]
    clients_accepted: List[int]
    clients_blocked: List[int]
    validation_accuracies: Dict[int, float]
    defense_active: bool = False
    
    def get_summary(self) -> str:
        if not self.defense_active:
            return "Defense Shield: OFF"
        if self.clients_blocked:
            return f"⚠️ ALERT: Blocked {len(self.clients_blocked)} malicious node(s): {self.clients_blocked}"
        return "✅ All nodes validated - no threats detected"


class HospitalClient:
    """
    Hospital client with optional malicious behavior simulation.
    
    Attack Mode: Label Flipping
    - When malicious=True, labels are flipped during training
    - Normal(0) -> Pneumonia(1)  
    - Pneumonia(1) -> COVID-19(2)
    - COVID-19(2) -> Normal(0)
    
    This creates a "poisoned" model that learns incorrect patterns.
    """
    
    # Label flip mapping for attack simulation
    LABEL_FLIP_MAP = {0: 1, 1: 2, 2: 0}
    
    def __init__(
        self,
        client_id: int,
        dataloader: DataLoader,
        device: torch.device = None,
        learning_rate: float = 0.001,
        local_epochs: int = 1,
        malicious: bool = False  # NEW: Attack flag
    ) -> None:
        """
        Initialize hospital client.
        
        Args:
            client_id: Unique identifier
            dataloader: Local training data  
            device: Compute device
            learning_rate: Learning rate
            local_epochs: Epochs per round
            malicious: If True, performs label flipping attack
        """
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device or torch.device('cpu')
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.malicious = malicious  # Attack mode flag
        
        self.model = create_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.n_samples = len(dataloader.dataset)
    
    def get_num_samples(self) -> int:
        """Return local training sample count."""
        return self.n_samples
    
    def is_malicious(self) -> bool:
        """Check if this client is in attack mode."""
        return self.malicious
    
    def _flip_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Perform Label Flipping Attack.
        
        This is a common data poisoning attack where the attacker
        trains with corrupted labels, causing the model to learn
        incorrect decision boundaries.
        """
        flipped = labels.clone()
        for old_label, new_label in self.LABEL_FLIP_MAP.items():
            flipped[labels == old_label] = new_label
        return flipped
    
    def train(
        self, 
        global_weights: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], TrainingMetrics]:
        """
        Perform local training (potentially poisoned if malicious).
        """
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
                
                # ===== ATTACK: Label Flipping =====
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
            avg_loss = 100.0  # High penalty for instability
            
        # Verify weights before sending (Check ALL state_dict items including BatchNorm buffers)
        is_valid = True
        for name, tensor in self.model.state_dict().items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"[ERROR] Client {self.client_id}: Found NaN in {name}")
                is_valid = False
                break
        
        updated_weights = {}
        if is_valid:
            updated_weights = {
                name: param.clone().detach().cpu()
                for name, param in self.model.state_dict().items()
            }
        else:
            # Fallback to global weights if training failed
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


class CentralServer:
    """
    Central server with Byzantine-robust aggregation.
    
    Defense Mechanism: Trusted Validation
    - Before aggregating, each client's model is evaluated on global test set
    - Clients with accuracy below threshold (e.g., < 30%) are flagged as malicious
    - Flagged clients are EXCLUDED from aggregation
    
    This prevents poisoned models from corrupting the global model.
    """
    
    MALICIOUS_THRESHOLD = 0.30  # Below 30% accuracy = suspicious
    
    def __init__(
        self,
        device: torch.device = None,
        privacy_noise: float = 0.0,
        defense_mode: bool = False  # NEW: Defense flag
    ) -> None:
        """
        Initialize central server.
        
        Args:
            device: Compute device
            privacy_noise: Differential privacy noise
            defense_mode: If True, validate and filter malicious clients
        """
        self.device = device or torch.device('cpu')
        self.privacy_noise = privacy_noise
        self.defense_mode = defense_mode
        self.global_model = create_model().to(self.device)
        
        # Security tracking
        self.blocked_count = 0
        self.last_security_report: Optional[SecurityReport] = None
        
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global model weights."""
        return {
            name: param.clone().detach().cpu()
            for name, param in self.global_model.state_dict().items()
        }
    
    def _validate_client_model(
        self,
        client_weights: Dict[str, torch.Tensor],
        test_images: torch.Tensor,
        test_labels: torch.Tensor
    ) -> float:
        """
        Validate a client's model on global test set.
        
        Returns accuracy on test set (0.0 to 1.0)
        """
        temp_model = create_model().to(self.device)
        temp_model.load_state_dict(client_weights)
        temp_model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            batch_size = 64
            n_samples = test_images.size(0)
            
            for i in range(0, n_samples, batch_size):
                batch_images = test_images[i:i+batch_size].to(self.device)
                batch_labels = test_labels[i:i+batch_size].to(self.device)
                
                outputs = temp_model(batch_images)
                
                # Check for NaNs during validation
                if torch.isnan(outputs).any():
                    return 0.0
                    
                _, predicted = outputs.max(1)
                
                correct += predicted.eq(batch_labels).sum().item()
                total += batch_labels.size(0)
        
        return correct / max(total, 1)
    
    def validate_and_aggregate(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        sample_counts: List[int],
        client_ids: List[int],
        test_images: torch.Tensor,
        test_labels: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], SecurityReport]:
        """
        Validate clients and aggregate only trusted ones.
        
        Defense Strategy:
        1. Evaluate each client's model on trusted test set
        2. Flag clients with accuracy < threshold as malicious
        3. Exclude malicious clients from aggregation
        4. Aggregate remaining clients using FedAvg
        
        Returns:
            (aggregated_weights, security_report)
        """
        validation_accuracies: Dict[int, float] = {}
        malicious_detected: List[int] = []
        clients_accepted: List[int] = []
        clients_blocked: List[int] = []
        
        # Step 1: Validate each client
        for i, (weights, client_id) in enumerate(zip(client_weights, client_ids)):
            acc = self._validate_client_model(weights, test_images, test_labels)
            validation_accuracies[client_id] = acc
            
            if self.defense_mode and acc < self.MALICIOUS_THRESHOLD:
                malicious_detected.append(client_id)
                clients_blocked.append(client_id)
                self.blocked_count += 1
            else:
                clients_accepted.append(client_id)
        
        # Step 2: Filter out malicious clients
        if clients_blocked:
            filtered_weights = []
            filtered_counts = []
            
            for i, client_id in enumerate(client_ids):
                if client_id not in clients_blocked:
                    filtered_weights.append(client_weights[i])
                    filtered_counts.append(sample_counts[i])
            
            client_weights = filtered_weights
            sample_counts = filtered_counts
        
        # Step 3: Aggregate remaining clients using FedAvg
        aggregated = self._fedavg_aggregate(client_weights, sample_counts)
        
        # Create security report
        report = SecurityReport(
            total_clients=len(client_ids),
            malicious_detected=malicious_detected,
            clients_accepted=clients_accepted,
            clients_blocked=clients_blocked,
            validation_accuracies=validation_accuracies,
            defense_active=self.defense_mode
        )
        
        self.last_security_report = report
        return aggregated, report
    
    def _fedavg_aggregate(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        sample_counts: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Standard FedAvg aggregation: w = Σ (n_k / n) * w_k
        """
        if not client_weights:
            # No valid clients - return current global weights
            return self.get_global_weights()
        
        total_samples = sum(sample_counts)
        if total_samples == 0:
            return self.get_global_weights()
        
        weight_coefficients = [n_k / total_samples for n_k in sample_counts]
        
        aggregated: Dict[str, torch.Tensor] = {}
        
        for key in client_weights[0].keys():
            param_dtype = client_weights[0][key].dtype
            aggregated[key] = torch.zeros_like(client_weights[0][key], dtype=torch.float32)
            
            for client_weight, coeff in zip(client_weights, weight_coefficients):
                weight_tensor = client_weight[key].float()
                
                # Skip if weight is corrupted
                if torch.isnan(weight_tensor).any():
                    continue
                    
                aggregated[key] += coeff * weight_tensor
            
            if param_dtype in (torch.int64, torch.int32, torch.long):
                aggregated[key] = aggregated[key].to(param_dtype)
        
        # Optional differential privacy
        if self.privacy_noise > 0:
            for key in aggregated.keys():
                if aggregated[key].dtype not in (torch.int64, torch.int32, torch.long):
                    noise = torch.randn_like(aggregated[key]) * self.privacy_noise
                    aggregated[key] += noise
        
        self.global_model.load_state_dict(aggregated)
        return aggregated
    
    def aggregate(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        sample_counts: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Standard aggregation (no defense)."""
        return self._fedavg_aggregate(client_weights, sample_counts)
    
    def evaluate_on_test_set(
        self,
        test_images: torch.Tensor,
        test_labels: torch.Tensor
    ) -> EvaluationMetrics:
        """Evaluate global model on hold-out test set."""
        self.global_model.eval()
        
        all_preds: List[int] = []
        all_labels: List[int] = []
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            batch_size = 64
            n_samples = test_images.size(0)
            
            for i in range(0, n_samples, batch_size):
                batch_images = test_images[i:i+batch_size].to(self.device)
                batch_labels = test_labels[i:i+batch_size].to(self.device)
                
                outputs = self.global_model(batch_images)
                loss = criterion(outputs, batch_labels)
                
                total_loss += loss.item() * batch_images.size(0)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(batch_labels.cpu().numpy().tolist())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        avg_loss = total_loss / max(n_samples, 1)
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=conf_matrix,
            loss=avg_loss
        )
    
    def get_model(self) -> XRayClassifier:
        """Get the current global model."""
        return self.global_model
    
    def get_blocked_count(self) -> int:
        """Get total number of blocked malicious updates."""
        return self.blocked_count


def run_federated_round(
    server: CentralServer,
    clients: List[HospitalClient],
    round_num: int,
    test_images: Optional[torch.Tensor] = None,
    test_labels: Optional[torch.Tensor] = None,
    use_defense: bool = False
) -> Tuple[Dict[str, Any], List[TrainingMetrics], Optional[EvaluationMetrics], Optional[SecurityReport]]:
    """
    Execute one FL round with optional security defense.
    
    Returns:
        (aggregated_metrics, client_metrics, test_metrics, security_report)
    """
    global_weights = server.get_global_weights()
    
    client_updates: List[Dict[str, torch.Tensor]] = []
    client_metrics: List[TrainingMetrics] = []
    sample_counts: List[int] = []
    client_ids: List[int] = []
    
    # Collect updates from all clients
    for client in clients:
        updated_weights, metrics = client.train(global_weights)
        client_updates.append(updated_weights)
        client_metrics.append(metrics)
        sample_counts.append(client.get_num_samples())
        client_ids.append(client.client_id)
    
    # Aggregate with or without defense
    security_report: Optional[SecurityReport] = None
    
    if use_defense and test_images is not None and test_labels is not None:
        _, security_report = server.validate_and_aggregate(
            client_updates, sample_counts, client_ids,
            test_images, test_labels
        )
        
        # Mark blocked clients in metrics
        if security_report:
            for metrics in client_metrics:
                if metrics.client_id in security_report.clients_blocked:
                    metrics.was_blocked = True
    else:
        server.aggregate(client_updates, sample_counts)
    
    # Calculate training metrics
    total_samples = sum(sample_counts)
    
    if total_samples > 0:
        avg_loss = sum(m.loss * m.samples_trained for m in client_metrics) / total_samples
        avg_accuracy = sum(m.accuracy * m.samples_trained for m in client_metrics) / total_samples
    else:
        avg_loss = 0.0
        avg_accuracy = 0.0
    
    if math.isnan(avg_loss):
        avg_loss = 0.0
    if math.isnan(avg_accuracy):
        avg_accuracy = 0.0
    
    aggregated_metrics = {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'total_samples': total_samples,
        'round': round_num
    }
    
    # Evaluate on test set
    test_metrics: Optional[EvaluationMetrics] = None
    if test_images is not None and test_labels is not None:
        test_metrics = server.evaluate_on_test_set(test_images, test_labels)
    
    return aggregated_metrics, client_metrics, test_metrics, security_report
