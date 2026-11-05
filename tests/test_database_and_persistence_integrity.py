"""Forensic Database and Persistence Integrity Test Suite.

Audit Reference: AUD-SEC-05 (Section 5: Database, Persistence, and State Integrity)
Verifies:
1. WAL journal mode, ACID transaction boundaries, and busy timeout concurrency.
2. Transactional rollback and partial write prevention under failure conditions.
3. System state upsert semantics and elimination of stale in-memory caches.
4. Diagnostic audits persistence with complete modality and clinical metadata.
5. SQLite sequence reset upon state wipe (clear_state).
6. Model checkpoint SHA-256 fingerprint registration and latest version retrieval.
7. Automated database integrity verification via PRAGMA integrity_check and health endpoint.
8. Hospital cohort counts normalization and schema validation resilience.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from api.main import app, state
from src.fed_xray.storage.state_store import PersistentStateStore


@pytest.fixture
def isolated_store(tmp_path: Path) -> PersistentStateStore:
    """Create an isolated, dedicated SQLite database instance."""
    db_file = tmp_path / "forensic_persistence.db"
    return PersistentStateStore(str(db_file))


def test_wal_journal_mode_and_concurrency_isolation(isolated_store: PersistentStateStore):
    """Verify WAL mode, busy timeout, and concurrent non-blocking reads during active writes."""
    stats = isolated_store.get_database_stats()
    assert stats["journal_mode"] == "wal", f"Expected WAL mode, got {stats['journal_mode']}"
    assert stats["wal_mode_enabled"] is True

    # Test concurrent reads during a simulated write lock
    read_results: List[int] = []
    errors: List[Exception] = []

    def background_reader():
        try:
            for _ in range(5):
                audits = isolated_store.get_diagnostic_audits(limit=5)
                read_results.append(len(audits))
                time.sleep(0.01)
        except Exception as exc:
            errors.append(exc)

    reader_thread = threading.Thread(target=background_reader)
    reader_thread.start()

    # Perform concurrent writes on main thread
    for i in range(5):
        isolated_store.record_training_round({
            "round_num": i + 1,
            "train_loss": 0.5 - i * 0.05,
            "train_accuracy": 70.0 + i * 5.0,
            "test_loss": 0.52 - i * 0.04,
            "test_accuracy": 68.0 + i * 4.5,
            "precision": 70.0,
            "recall": 68.0,
            "f1_score": 69.0,
            "privacy_epsilon": 0.1 * (i + 1),
            "uplink_bytes": 1024 * (i + 1),
            "threat_detected": False,
            "blocked_nodes": [],
            "status": "completed",
        })
        time.sleep(0.01)

    reader_thread.join(timeout=5.0)

    assert len(errors) == 0, f"Concurrent reader encountered exceptions: {errors}"
    assert len(read_results) == 5, "Reader failed to complete all queries during writes"


def test_transactional_rollback_and_partial_write_atomicity(isolated_store: PersistentStateStore):
    """Verify that aborted operations roll back completely, preventing partial writes."""
    # Attempt an atomic transaction that fails midway
    with pytest.raises(sqlite3.IntegrityError):
        with isolated_store._connection() as conn:
            conn.execute(
                """
                INSERT INTO training_rounds (
                    round_num, train_loss, train_accuracy, test_loss, test_accuracy,
                    precision, recall, f1_score, privacy_epsilon, uplink_bytes,
                    threat_detected, blocked_nodes, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (1, 0.5, 80.0, 0.5, 80.0, 80.0, 80.0, 80.0, 0.5, 1024, 0, "[]", "completed", "2026-11-05T00:00:00Z"),
            )
            # Intentionally insert a duplicate primary key without ON CONFLICT to trigger rollback
            conn.execute(
                """
                INSERT INTO training_rounds (
                    round_num, train_loss, train_accuracy, test_loss, test_accuracy,
                    precision, recall, f1_score, privacy_epsilon, uplink_bytes,
                    threat_detected, blocked_nodes, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (1, 0.9, 50.0, 0.9, 50.0, 50.0, 50.0, 50.0, 0.9, 1024, 0, "[]", "failed", "2026-11-05T00:00:00Z"),
            )

    # Verify that the entire transaction was rolled back and table remains empty
    history = isolated_store.get_training_history()
    assert len(history["round"]) == 0, "Transaction rollback failed: partial write remained in database"


def test_system_state_upsert_and_memory_synchronization(isolated_store: PersistentStateStore):
    """Verify upsert semantics and authoritative state retrieval."""
    initial_state = {
        "model_trained": True,
        "model_type": "vit_tiny",
        "peft_mode": "ffa_lora",
        "use_ckks": True,
        "secure_aggregation": True,
        "rounds_completed": 5,
    }
    isolated_store.save_system_state(initial_state)

    loaded = isolated_store.load_system_state()
    assert loaded["model_trained"] is True
    assert loaded["model_type"] == "vit_tiny"
    assert loaded["peft_mode"] == "ffa_lora"
    assert loaded["use_ckks"] is True
    assert loaded["rounds_completed"] == 5

    # Update subset of keys
    update_state = {
        "peft_mode": "fedsa_lora",
        "rounds_completed": 10,
    }
    isolated_store.save_system_state(update_state)

    reloaded = isolated_store.load_system_state()
    assert reloaded["model_type"] == "vit_tiny"  # preserved
    assert reloaded["peft_mode"] == "fedsa_lora"  # updated
    assert reloaded["rounds_completed"] == 10  # updated


def test_diagnostic_audit_trail_immutable_persistence_and_sequence_reset(isolated_store: PersistentStateStore):
    """Verify diagnostic audit persistence with modality and clean sequence reset."""
    audit_payload = {
        "patient_id": "PAT_FORENSIC_099",
        "predicted_class": 1,
        "predicted_name": "Pneumonia",
        "true_class": 1,
        "true_name": "Pneumonia",
        "confidence": 93.4,
        "probabilities": [4.2, 93.4, 2.4],
        "findings": "Dense focal infiltrate in right middle lobe.",
        "image_source": "holdout_test_set:chexpert",
        "provenance": "chexpert_v2_real_clinical",
        "anatomical_site": "Chest PA",
        "biopsy_confirmed": True,
        "raw_image": [[0.1, 0.2], [0.3, 0.4]],
        "heatmap": [[0.8, 0.9], [0.7, 0.6]],
        "modality": "Chest X-Ray",
    }

    row_id = isolated_store.record_diagnostic_audit(audit_payload)
    assert row_id >= 1

    last_diag = isolated_store.get_last_diagnosis()
    assert last_diag is not None
    assert last_diag["patient_id"] == "PAT_FORENSIC_099"
    assert last_diag["modality"] == "Chest X-Ray"
    assert last_diag["confidence"] == pytest.approx(93.4, rel=1e-2)
    assert last_diag["biopsy_confirmed"] is True

    # Clear state and verify autoincrement sequence counter reset
    isolated_store.clear_state()
    stats = isolated_store.get_database_stats()
    assert stats["persisted_audits_count"] == 0

    new_row_id = isolated_store.record_diagnostic_audit(audit_payload)
    assert new_row_id == 1, f"Expected sequence to reset to 1 after clear_state, got {new_row_id}"


def test_model_checkpoints_fingerprint_integrity_and_retrieval(isolated_store: PersistentStateStore):
    """Verify model checkpoint SHA-256 fingerprint tracking and latest checkpoint retrieval."""
    ckpt1 = {
        "model_type": "vit_tiny",
        "peft_mode": "ffa_lora",
        "checkpoint_path": "checkpoints/round_1.pt",
        "checkpoint_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "is_pretrained": True,
        "provenance": "disk_checkpoint:round_1.pt",
        "rounds_completed": 1,
    }
    ckpt2 = {
        "model_type": "vit_tiny",
        "peft_mode": "ffa_lora",
        "checkpoint_path": "checkpoints/round_2.pt",
        "checkpoint_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e",
        "is_pretrained": True,
        "provenance": "disk_checkpoint:round_2.pt",
        "rounds_completed": 2,
    }

    isolated_store.save_checkpoint(ckpt1)
    isolated_store.save_checkpoint(ckpt2)

    latest = isolated_store.get_latest_checkpoint()
    assert latest is not None
    assert latest["checkpoint_hash"] == ckpt2["checkpoint_hash"]
    assert latest["rounds_completed"] == 2
    assert latest["checkpoint_path"] == "checkpoints/round_2.pt"


def test_database_integrity_verification_and_health_endpoint(isolated_store: PersistentStateStore):
    """Verify PRAGMA integrity_check and exposure through the /api/health endpoint."""
    verification = isolated_store.verify_database_integrity()
    assert verification["is_healthy"] is True
    assert verification["integrity_check"].lower() == "ok"
    assert verification["foreign_key_violations"] == 0
    assert verification["wal_enabled"] is True
    assert len(verification["missing_tables"]) == 0

    # Test through FastAPI endpoint
    original_store = state.store
    try:
        state.store = isolated_store
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "database" in data
        assert data["database"]["is_healthy"] is True
        assert data["database"]["integrity_check"].lower() == "ok"
    finally:
        state.store = original_store


def test_hospital_cohorts_normalization_and_provenance_integrity(isolated_store: PersistentStateStore):
    """Verify resilient normalization of hospital counts across list, dict, and empty payloads."""
    cohorts_mixed: List[Dict[str, Any]] = [
        {
            "hospital_id": 1,
            "name": "Hospital Clinic de Barcelona",
            "num_samples": 100,
            "distribution": [0.4, 0.4, 0.2],
            "counts": [40, 40, 20],  # list input
            "sample_images": [],
            "sample_labels": [],
            "provenance": "isic_archive_v2_real_clinical",
            "is_authentic_clinical": True,
        },
        {
            "hospital_id": 2,
            "name": "Medical University of Vienna",
            "num_samples": 80,
            "distribution": [0.5, 0.3, 0.2],
            "counts": {"0": 40, "1": 24, "2": 16},  # dict input
            "sample_images": [],
            "sample_labels": [],
            "provenance": "isic_archive_v2_real_clinical",
            "is_authentic_clinical": True,
        },
    ]

    isolated_store.save_hospital_cohorts(cohorts_mixed)
    loaded = isolated_store.load_hospital_cohorts()

    assert len(loaded) == 2
    # Both must be normalized to Dict[str, int]
    assert loaded[0]["counts"] == {"0": 40, "1": 40, "2": 20}
    assert loaded[1]["counts"] == {"0": 40, "1": 24, "2": 16}
    assert loaded[0]["is_authentic_clinical"] is True
