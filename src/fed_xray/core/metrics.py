"""
Fed-XRay Core Metrics & Security Reporting Data Structures
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


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
    Security report from Byzantine defense mechanism.
    
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
            return f"Blocked {len(self.clients_blocked)} malicious node(s): {self.clients_blocked}"
        return "All nodes validated - no threats detected"
