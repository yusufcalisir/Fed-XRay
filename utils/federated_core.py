"""
Fed-XRay Federated Core (Backward-Compatibility Shim)
=====================================================
Re-exports client, server, and metrics from `src.fed_xray.core`.
"""

from src.fed_xray.core.metrics import (
    TrainingMetrics,
    EvaluationMetrics,
    SecurityReport
)
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round

__all__ = [
    "TrainingMetrics",
    "EvaluationMetrics",
    "SecurityReport",
    "HospitalClient",
    "CentralServer",
    "run_federated_round"
]
