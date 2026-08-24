"""
Fed-XRay Medical Data Generator (Backward-Compatibility Shim)
=============================================================
Re-exports data generator and dataset loaders from `src.fed_xray.data.generator`.
"""

from src.fed_xray.data.generator import (
    MedicalDataGenerator,
    XRayDataset,
    create_global_test_set,
    create_hospital_dataloaders,
    get_distribution_info
)

__all__ = [
    "MedicalDataGenerator",
    "XRayDataset",
    "create_global_test_set",
    "create_hospital_dataloaders",
    "get_distribution_info"
]
