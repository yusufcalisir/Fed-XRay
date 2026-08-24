"""
Fed-XRay Central Server & Federated Aggregation
================================================
Implements FedAvg, FedProx, Byzantine defense, dispersion-weighted prototype
synthesis, and round orchestration.
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
from .prototypes import aggregate_prototypes_dispersion_weighted


class CentralServer:
    """
    Central coordinator server with Byzantine-robust aggregation and
    dispersion-weighted prototype metric synthesis.
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
        self.global_prototypes: Optional[Dict[int, torch.Tensor]] = None
        
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global model weights."""
        return {
            name: param.clone().detach().cpu()
            for name, param in self.global_model.state_dict().items()
        }
    
    def get_global_prototypes(self) -> Optional[Dict[int, torch.Tensor]]:
        """Get current global class prototypes."""
        return self.global_prototypes
    
    def set_global_prototypes(self, prototypes: Dict[int, torch.Tensor]) -> None:
        """Set or update global prototypes."""
        self.global_prototypes = prototypes
    
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
        trusted_weights: List[Dict[str, torch.Tensor]] = []
        trusted_counts: List[int] = []
        clients_blocked: List[int] = []
        
        for weights, count, cid in zip(client_weights, sample_counts, client_ids):
            val_acc = self._validate_client_model(weights, test_images, test_labels)
            validation_accuracies[cid] = val_acc
            
            if val_acc < self.MALICIOUS_THRESHOLD:
                print(f"[SECURITY ALERT] Node {cid} failed validation (Acc: {val_acc:.1%}). BLOCKED!")
                clients_blocked.append(cid)
                self.blocked_count += 1
            else:
                trusted_weights.append(weights)
                trusted_counts.append(count)
        
        if not trusted_weights:
            print("[WARNING] All client updates failed validation. Keeping current global model.")
            aggregated_weights = self.get_global_weights()
        else:
            aggregated_weights = self.aggregate(trusted_weights, trusted_counts)
        
        security_report = SecurityReport(
            clients_evaluated=client_ids,
            clients_accepted=[cid for cid in client_ids if cid not in clients_blocked],
            clients_blocked=clients_blocked,
            validation_accuracies=validation_accuracies,
            threat_detected=len(clients_blocked) > 0,
            details=f"Blocked {len(clients_blocked)} node(s) below {self.MALICIOUS_THRESHOLD:.0%} threshold"
        )
        self.last_security_report = security_report
        
        return aggregated_weights, security_report

    def aggregate(
        self, 
        client_weights: List[Dict[str, torch.Tensor]], 
        sample_counts: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Perform sample-weighted Federated Averaging (FedAvg)."""
        total_samples = sum(sample_counts)
        if total_samples == 0:
            return self.get_global_weights()
            
        aggregated_weights = {}
        first_client = client_weights[0]
        
        for key in first_client.keys():
            weighted_sum = torch.zeros_like(first_client[key], dtype=torch.float32)
            
            for client_w, n_samples in zip(client_weights, sample_counts):
                weight = n_samples / total_samples
                weighted_sum += weight * client_w[key].float()
            
            if self.privacy_noise > 0.0:
                noise = torch.randn_like(weighted_sum) * self.privacy_noise
                weighted_sum += noise
            
            aggregated_weights[key] = weighted_sum
        
        self.global_model.load_state_dict(aggregated_weights)
        return aggregated_weights
    
    def evaluate_on_test_set(
        self,
        test_images: torch.Tensor,
        test_labels: torch.Tensor
    ) -> EvaluationMetrics:
        """Evaluate global model on hold-out validation set."""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        n_samples = test_images.size(0)
        
        with torch.no_grad():
            batch_size = 64
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
    total_rounds: int = 10,
    test_images: Optional[torch.Tensor] = None,
    test_labels: Optional[torch.Tensor] = None,
    use_defense: bool = False,
    algorithm: str = "FedAvg",
    loss_fn_name: str = "CE",
    enable_prototypes: bool = False,
    proto_weight: float = 0.1,
    prox_mu: float = 0.0
) -> Tuple[Dict[str, Any], List[TrainingMetrics], Optional[EvaluationMetrics], Optional[SecurityReport]]:
    """Execute one federated learning round with optional prototype metric alignment."""
    global_weights = server.get_global_weights()
    global_prototypes = server.get_global_prototypes() if enable_prototypes else None
    
    client_updates: List[Dict[str, torch.Tensor]] = []
    client_metrics: List[TrainingMetrics] = []
    sample_counts: List[int] = []
    client_ids: List[int] = []
    
    client_protos: List[Dict[int, torch.Tensor]] = []
    client_traces: List[Dict[int, float]] = []
    client_counts: List[Dict[int, int]] = []
    
    for client in clients:
        updated_weights, metrics = client.train(
            global_weights=global_weights,
            loss_fn_name=loss_fn_name,
            current_round=round_num,
            total_rounds=total_rounds,
            global_prototypes=global_prototypes,
            proto_weight=proto_weight if enable_prototypes else 0.0,
            prox_mu=prox_mu if algorithm == "FedProx" else 0.0
        )
        client_updates.append(updated_weights)
        client_metrics.append(metrics)
        sample_counts.append(client.get_num_samples())
        client_ids.append(client.client_id)
        
        if enable_prototypes:
            p_dict, t_dict, c_dict = client.compute_prototypes_and_dispersion()
            client_protos.append(p_dict)
            client_traces.append(t_dict)
            client_counts.append(c_dict)
    
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
    
    # Aggregate prototypes
    if enable_prototypes and client_protos:
        synthesized_protos = aggregate_prototypes_dispersion_weighted(
            client_prototypes=client_protos,
            client_traces=client_traces,
            client_counts=client_counts,
            num_classes=3
        )
        server.set_global_prototypes(synthesized_protos)
    
    total_samples = sum(sample_counts)
    if total_samples > 0:
        avg_loss = sum(m.loss * m.samples_trained for m in client_metrics) / total_samples
        avg_accuracy = sum(m.accuracy * m.samples_trained for m in client_metrics) / total_samples
    else:
        avg_loss = 0.0
        avg_accuracy = 0.0
        
    aggregated_metrics = {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'total_samples': total_samples,
        'round': round_num,
        'prototypes_enabled': enable_prototypes,
        'loss_fn': loss_fn_name,
        'algorithm': algorithm
    }
    
    test_metrics: Optional[EvaluationMetrics] = None
    if test_images is not None and test_labels is not None:
        test_metrics = server.evaluate_on_test_set(test_images, test_labels)
        
    return aggregated_metrics, client_metrics, test_metrics, security_report
