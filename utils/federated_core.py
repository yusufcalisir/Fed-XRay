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


def format_byte_size(num_bytes: float) -> str:
    """Format byte counts into clean human-readable metric strings."""
    if num_bytes >= 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{int(num_bytes)} B"


__all__ = [
    "TrainingMetrics",
    "EvaluationMetrics",
    "SecurityReport",
    "HospitalClient",
    "CentralServer",
    "run_federated_round",
    "format_byte_size",
]
