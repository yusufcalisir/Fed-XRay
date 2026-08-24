"""
Fed-XRay Core Metrics & Security Reporting Data Structures
"""

import numpy as np
from dataclasses import dataclass, field
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
    total_clients: int = 0
    malicious_detected: List[int] = field(default_factory=list)
    clients_accepted: List[int] = field(default_factory=list)
    clients_blocked: List[int] = field(default_factory=list)
    validation_accuracies: Dict[int, float] = field(default_factory=dict)
    defense_active: bool = False
    threat_detected: bool = False
    details: str = ""
    clients_evaluated: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.total_clients and self.clients_evaluated:
            self.total_clients = len(self.clients_evaluated)
        if not self.clients_evaluated and self.total_clients:
            self.clients_evaluated = list(self.validation_accuracies.keys())
        if not self.threat_detected:
            self.threat_detected = len(self.clients_blocked) > 0
        if not self.malicious_detected:
            self.malicious_detected = list(self.clients_blocked)
            
    def get_summary(self) -> str:
        if not self.defense_active:
            return "Defense Shield: OFF"
        if not self.clients_blocked:
            return f"Defense Shield: ACTIVE ({self.total_clients} clients verified)"
        return f"ALERT: {len(self.clients_blocked)} malicious client(s) blocked: {self.clients_blocked}"
