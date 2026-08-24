"""
Fed-XRay Federated Core Engine (Backward-Compatibility Shim)
============================================================
Re-exports client, server, evaluation, and distributed optimization functions.
"""

from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round
from src.fed_xray.core.metrics import TrainingMetrics, EvaluationMetrics, SecurityReport
from src.fed_xray.core.algorithms import (
    compute_fedprox_loss,
    compute_feddyn_loss,
    compute_moon_contrastive_loss,
    ScaffoldController
)

__all__ = [
    "HospitalClient",
    "CentralServer",
    "run_federated_round",
    "TrainingMetrics",
    "EvaluationMetrics",
    "SecurityReport",
    "compute_fedprox_loss",
    "compute_feddyn_loss",
    "compute_moon_contrastive_loss",
    "ScaffoldController"
]
