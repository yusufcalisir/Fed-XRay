"""Data package for Federated Medical Imaging."""

from src.fed_xray.data.generator import (
    MedicalDataGenerator,
    get_distribution_info,
    create_hospital_dataloaders,
    XRayDataset,
)
from src.fed_xray.data.real_world import (
    DiskMedicalDataset,
    RealWorldPatientRecord,
    StrategyEDatasetEcosystem,
)
from src.fed_xray.data.ingest_real_world import (
    verify_cohort_authenticity,
    ensure_strategy_e_dataset,
)

__all__ = [
    "MedicalDataGenerator",
    "get_distribution_info",
    "create_hospital_dataloaders",
    "XRayDataset",
    "DiskMedicalDataset",
    "RealWorldPatientRecord",
    "StrategyEDatasetEcosystem",
    "verify_cohort_authenticity",
    "ensure_strategy_e_dataset",
]
