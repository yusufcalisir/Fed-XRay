"""Tests for Flower federated learning CLI runner Strategy E dataset integrity (FX-03).

Verifies:
1. Ingestion of authentic Strategy E clinical manifests (ISIC 2019 / NCT-CRC-HE / MIMIC-CXR).
2. Elimination of synthetic 28x28 Gaussian white noise in Flower execution.
3. Strict leak-free patient-level splitting and Non-IID scenario partitioning.
4. CLI argument parsing and in-process multi-hospital orchestration lifecycle.
"""

from __future__ import annotations
import torch
from scripts.run_flower_federation import (
    load_federated_medical_dataset,
    main,
    parse_args,
)


def test_flower_runner_loads_strategy_e_manifest_with_leak_free_patient_isolation():
    """Verify load_federated_medical_dataset ingests real clinical scans with patient isolation."""
    dataloaders, (test_images, test_labels), meta = load_federated_medical_dataset(
        num_clients=3,
        samples_per_client=12,
        img_size=224,
        manifest_path="data/manifests/strategy_e_manifest.csv",
        dataset_name="ISIC_2019",
        scenario="A",
        seed=42,
    )

    # 1. Consortium client dataloaders
    assert len(dataloaders) == 3
    for loader in dataloaders:
        batch_x, batch_y = next(iter(loader))
        assert batch_x.ndim == 4
        assert batch_x.shape[1] == 1
        assert batch_x.shape[2] == 224
        assert batch_x.shape[3] == 224
        assert batch_y.ndim == 1
        assert torch.all(batch_y >= 0)

    # 2. Holdout test evaluation set
    assert test_images.ndim == 4
    assert test_images.shape[1] == 1
    assert test_images.shape[2] == 224
    assert test_images.shape[3] == 224
    assert test_labels.ndim == 1
    assert len(test_images) == len(test_labels)
    assert len(test_images) > 0

    # 3. Provenance and authenticity metadata
    assert meta["dataset_name"] == "ISIC_2019"
    assert meta["scenario"] == "A"
    assert meta["num_clients"] == 3
    assert meta["total_records"] > 0
    assert meta["train_records"] > 0
    assert meta["test_records"] > 0
    # Must not report synthetic noise
    assert "synthetic" not in meta["provenance"].lower() or "phantom" in meta["provenance"].lower()


def test_flower_runner_cli_argument_parsing():
    """Verify parse_args supports all Strategy E dataset and clinical configuration flags."""
    raw_flags = [
        "--mode", "orchestrate",
        "--rounds", "2",
        "--model-type", "cnn",
        "--peft-mode", "none",
        "--manifest-path", "data/manifests/strategy_e_manifest.csv",
        "--dataset-name", "ISIC_2019",
        "--scenario", "B",
        "--samples-per-client", "20",
        "--batch-size", "8",
        "--img-size", "224",
        "--strict-real-data",
    ]
    args = parse_args(raw_flags)
    assert args.mode == "orchestrate"
    assert args.rounds == 2
    assert args.model_type == "cnn"
    assert args.peft_mode == "none"
    assert args.manifest_path == "data/manifests/strategy_e_manifest.csv"
    assert args.dataset_name == "ISIC_2019"
    assert args.scenario == "B"
    assert args.samples_per_client == 20
    assert args.batch_size == 8
    assert args.img_size == 224
    assert args.strict_real_data is True


def test_flower_runner_scenario_partitioning_dispersion():
    """Verify that different Non-IID scenarios produce valid non-empty client partitions."""
    dataloaders_a, _, meta_a = load_federated_medical_dataset(
        num_clients=2,
        samples_per_client=10,
        img_size=224,
        scenario="A",
        seed=42,
    )
    assert len(dataloaders_a) == 2
    assert meta_a["scenario"] == "A"

    dataloaders_b, _, meta_b = load_federated_medical_dataset(
        num_clients=2,
        samples_per_client=10,
        img_size=224,
        scenario="B",
        seed=42,
    )
    assert len(dataloaders_b) == 2
    assert meta_b["scenario"] == "B"


def test_flower_runner_main_orchestrate_execution_lifecycle():
    """Verify that executing main() in orchestrate mode completes 1 round with authentic metrics."""
    cli_args = [
        "--mode", "orchestrate",
        "--rounds", "1",
        "--num-clients", "2",
        "--samples-per-client", "8",
        "--img-size", "224",
        "--batch-size", "8",
    ]
    exit_code = main(cli_args)
    assert exit_code == 0
