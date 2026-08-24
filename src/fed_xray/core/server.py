"""
Fed-XRay Central Server & Federated Aggregation
================================================
Implements FedAvg, trusted validation Byzantine defense, and round orchestration.
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    accuracy_score
)

from ..models.cnn import XRayClassifier, create_model
from .client import HospitalClient
from .metrics import TrainingMetrics, EvaluationMetrics, SecurityReport


class CentralServer:
    """
    Central coordinator server with Byzantine-robust aggregation.
    
    Defense Mechanism: Trusted Validation
    - Evaluates client updates on a trusted hold-out validation set.
    - Excludes updates below malicious threshold.
    - Aggregates remaining updates using sample-weighted FedAvg.
    """
    
    MALICIOUS_THRESHOLD = 0.30
    
    def __init__(
        self,
        device: torch.device = None,
        privacy_noise: float = 0.0,
        defense_mode: bool = False
    ) -> None:
        self.device = device or torch.device('cpu')
        self.privacy_noise = privacy_noise
        self.defense_mode = defense_mode
        self.global_model = create_model().to(self.device)
        
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
        """Validate a client's model on trusted global validation set."""
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
        """Validate clients and aggregate only trusted updates."""
        validation_accuracies: Dict[int, float] = {}
        malicious_detected: List[int] = []
        clients_accepted: List[int] = []
        clients_blocked: List[int] = []
        
        for i, (weights, client_id) in enumerate(zip(client_weights, client_ids)):
            acc = self._validate_client_model(weights, test_images, test_labels)
            validation_accuracies[client_id] = acc
            
            if self.defense_mode and acc < self.MALICIOUS_THRESHOLD:
                malicious_detected.append(client_id)
                clients_blocked.append(client_id)
                self.blocked_count += 1
            else:
                clients_accepted.append(client_id)
        
        if clients_blocked:
            filtered_weights = []
            filtered_counts = []
            for i, client_id in enumerate(client_ids):
                if client_id not in clients_blocked:
                    filtered_weights.append(client_weights[i])
                    filtered_counts.append(sample_counts[i])
            
            client_weights = filtered_weights
            sample_counts = filtered_counts
        
        aggregated = self._fedavg_aggregate(client_weights, sample_counts)
        
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
        """Standard FedAvg aggregation: w = Σ (n_k / n) * w_k."""
        if not client_weights:
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
                if torch.isnan(weight_tensor).any():
                    continue
                aggregated[key] += coeff * weight_tensor
            
            if param_dtype in (torch.int64, torch.int32, torch.long):
                aggregated[key] = aggregated[key].to(param_dtype)
        
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
        """Standard aggregation without active defense."""
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
    """Execute one federated learning round with optional security defense."""
    global_weights = server.get_global_weights()
    
    client_updates: List[Dict[str, torch.Tensor]] = []
    client_metrics: List[TrainingMetrics] = []
    sample_counts: List[int] = []
    client_ids: List[int] = []
    
    for client in clients:
        updated_weights, metrics = client.train(global_weights)
        client_updates.append(updated_weights)
        client_metrics.append(metrics)
        sample_counts.append(client.get_num_samples())
        client_ids.append(client.client_id)
    
    security_report: Optional[SecurityReport] = None
    
    if use_defense and test_images is not None and test_labels is not None:
        _, security_report = server.validate_and_aggregate(
            client_updates, sample_counts, client_ids,
            test_images, test_labels
        )
        if security_report:
            for metrics in client_metrics:
                if metrics.client_id in security_report.clients_blocked:
                    metrics.was_blocked = True
    else:
        server.aggregate(client_updates, sample_counts)
    
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
    
    test_metrics: Optional[EvaluationMetrics] = None
    if test_images is not None and test_labels is not None:
        test_metrics = server.evaluate_on_test_set(test_images, test_labels)
    
    return aggregated_metrics, client_metrics, test_metrics, security_report
