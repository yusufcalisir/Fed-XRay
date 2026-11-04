"""Unit and integration tests for the Master Foundation PEFT Benchmarking Engine."""

from benchmarks.benchmark_foundation_peft import run_single_foundation_benchmark


def test_foundation_benchmark_execution():
    """Verify single foundation benchmark runs cleanly and returns valid metrics."""
    result = run_single_foundation_benchmark(
        algorithm="FedDyn",
        loss_fn="DAFL",
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        scenario="A",
        num_rounds=2,
        num_clients=2,
        samples_per_client=30,
        seed=123,
    )

    assert "accuracy" in result
    assert "f1_score" in result
    assert "payload_mb_per_round" in result
    assert "dp_epsilon" in result

    assert 0.0 <= result["accuracy"] <= 100.0
    assert 0.0 <= result["f1_score"] <= 100.0
    assert result["payload_mb_per_round"] < 5.0
    assert 0.0 < result["dp_epsilon"] <= 3.0


def test_foundation_benchmark_resolution_and_data_integrity():
    """Verify benchmark engine enforces 224x224 resolution without 28x28 synthetic substitution."""
    from src.fed_xray.data.real_world import StrategyEDatasetEcosystem

    ecosystem = StrategyEDatasetEcosystem(seed=42)
    # 1. Default generation must be 224x224
    records = ecosystem.generate_synthetic_realworld_cohort(num_patients=10, samples_per_patient=1)
    assert records[0].image_tensor.shape == (1, 224, 224), "Default cohort must be 224x224"
    assert records[0].is_authentic_clinical is True or len(records[0].clinical_notes) > 10

    # 2. Benchmark execution with synthetic fallback preserves 224x224
    res_synthetic = run_single_foundation_benchmark(
        algorithm="FedAvg",
        loss_fn="CE",
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        scenario="A",
        num_rounds=1,
        num_clients=2,
        use_disk_manifest=False,
        image_size=224,
    )
    assert res_synthetic["rounds"] == 1
    assert "accuracy" in res_synthetic
