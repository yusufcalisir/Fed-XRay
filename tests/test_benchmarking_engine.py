"""Unit and integration tests for the Master Foundation PEFT Benchmarking Engine."""

import pytest
import os
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
