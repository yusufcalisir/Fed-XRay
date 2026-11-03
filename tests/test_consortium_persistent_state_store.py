"""ACID Consortium Persistent State Store Verification Suite.

Verifies SQLite WAL mode, write-through persistence across cohort generation,
federated training rounds, diagnostic audits, and restart state hydration.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.fed_xray.storage.state_store import PersistentStateStore
from api.main import app, AppState


@pytest.fixture
def temp_store(tmp_path: Path) -> PersistentStateStore:
    db_file = tmp_path / "test_consortium.db"
    return PersistentStateStore(db_path=db_file)


def test_sqlite_wal_mode_and_tables(temp_store: PersistentStateStore):
    """Verify WAL journal mode and schema integrity."""
    stats = temp_store.get_database_stats()
    assert stats["wal_mode_enabled"] is True
    assert stats["rounds_count"] == 0
    assert stats["cohorts_count"] == 0
    assert stats["audits_count"] == 0
    assert stats["checkpoints_count"] == 0

    with temp_store._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        expected_tables = {
            "system_state",
            "training_rounds",
            "hospital_cohorts",
            "diagnostic_audits",
            "model_checkpoints",
        }
        assert expected_tables.issubset(tables)


def test_system_state_save_and_load(temp_store: PersistentStateStore):
    """Verify system state upsert and retrieval."""
    test_state = {
        "model_type": "uni",
        "peft_mode": "ffa_lora",
        "defense_mode": "trimmed_mean",
        "image_size": 224,
        "data_source": "isic_archive_v2_real_clinical",
        "data_provenance": "isic_2019_real_clinical_archive",
        "is_authentic_clinical": True,
        "strict_real_data": True,
        "model_trained": True,
        "rounds_completed": 5,
        "privacy_spent": 1.42,
    }
    temp_store.save_system_state(test_state)

    loaded = temp_store.load_system_state()
    assert loaded is not None
    assert loaded["model_type"] == "uni"
    assert loaded["peft_mode"] == "ffa_lora"
    assert loaded["image_size"] == 224
    assert loaded["is_authentic_clinical"] is True
    assert loaded["strict_real_data"] is True
    assert loaded["rounds_completed"] == 5
    assert loaded["privacy_spent"] == pytest.approx(1.42, rel=1e-3)


def test_training_rounds_persistence(temp_store: PersistentStateStore):
    """Verify training rounds persistence and history reconstruction."""
    rounds = [
        {
            "round_num": 1,
            "train_loss": 0.85,
            "train_accuracy": 72.5,
            "test_loss": 0.82,
            "test_accuracy": 74.0,
            "precision": 73.0,
            "recall": 72.0,
            "f1_score": 72.5,
            "privacy_epsilon": 0.45,
            "status": "completed",
        },
        {
            "round_num": 2,
            "train_loss": 0.55,
            "train_accuracy": 83.2,
            "test_loss": 0.52,
            "test_accuracy": 84.5,
            "precision": 84.0,
            "recall": 83.0,
            "f1_score": 83.5,
            "privacy_epsilon": 0.90,
            "status": "completed",
        },
    ]

    for r in rounds:
        temp_store.record_training_round(r)

    history = temp_store.get_training_history()
    assert history["round"] == [1, 2]
    assert history["loss"] == [0.85, 0.55]
    assert history["test_loss"] == [0.82, 0.52]
    assert history["accuracy"] == [72.5, 83.2]
    assert history["test_accuracy"] == [74.0, 84.5]
    assert history["f1_score"] == [72.5, 83.5]
    assert history["privacy_spent"] == pytest.approx(0.90, rel=1e-3)


def test_hospital_cohorts_persistence(temp_store: PersistentStateStore):
    """Verify hospital cohorts persistence and retrieval."""
    cohorts = [
        {
            "hospital_id": 1,
            "name": "Hospital Clinic de Barcelona",
            "num_samples": 120,
            "distribution": [0.3, 0.4, 0.3],
            "counts": [36, 48, 36],
            "sample_images": [[[0.1, 0.2], [0.3, 0.4]]],
            "sample_labels": [0],
            "provenance": "isic_archive_v2_real_clinical",
            "is_authentic_clinical": True,
        },
        {
            "hospital_id": 2,
            "name": "Medical University of Vienna",
            "num_samples": 95,
            "distribution": [0.5, 0.2, 0.3],
            "counts": [47, 19, 29],
            "sample_images": [[[0.5, 0.6], [0.7, 0.8]]],
            "sample_labels": [1],
            "provenance": "isic_archive_v2_real_clinical",
            "is_authentic_clinical": True,
        },
    ]

    temp_store.save_hospital_cohorts(cohorts)
    loaded = temp_store.load_hospital_cohorts()
    assert len(loaded) == 2
    assert loaded[0]["hospital_id"] == 1
    assert loaded[0]["name"] == "Hospital Clinic de Barcelona"
    assert loaded[0]["num_samples"] == 120
    assert loaded[0]["is_authentic_clinical"] is True
    assert loaded[1]["hospital_id"] == 2
    assert loaded[1]["num_samples"] == 95


def test_diagnostic_audits_persistence(temp_store: PersistentStateStore):
    """Verify diagnostic audit trail persistence and retrieval."""
    audit_entry = {
        "patient_id": "PAT_ISIC_00124",
        "predicted_class": 0,
        "predicted_name": "Melanoma",
        "true_class": 0,
        "true_name": "Melanoma",
        "confidence": 94.6,
        "probabilities": [94.6, 3.2, 2.2],
        "findings": "Asymmetric pigmented lesion with atypical network.",
        "raw_image": [[0.1, 0.2], [0.3, 0.4]],
        "heatmap": [[0.8, 0.9], [0.7, 0.6]],
        "image_source": "holdout_test_set:isic_archive_v2_real_clinical",
        "provenance": "isic_archive_v2_real_clinical",
        "anatomical_site": "anterior torso",
        "biopsy_confirmed": True,
    }

    temp_store.record_diagnostic_audit(audit_entry)

    last = temp_store.get_last_diagnosis()
    assert last is not None
    assert last["patient_id"] == "PAT_ISIC_00124"
    assert last["predicted_name"] == "Melanoma"
    assert last["confidence"] == pytest.approx(94.6, rel=1e-2)
    assert last["biopsy_confirmed"] is True

    audits = temp_store.get_diagnostic_audits(limit=10)
    assert len(audits) == 1
    assert audits[0]["patient_id"] == "PAT_ISIC_00124"


def test_checkpoints_persistence(temp_store: PersistentStateStore):
    """Verify model checkpoint metadata persistence."""
    ckpt = {
        "model_type": "uni",
        "peft_mode": "ffa_lora",
        "checkpoint_path": "checkpoints/federated_global_model.pt",
        "checkpoint_hash": "a1b2c3d4e5f67890",
        "is_pretrained": True,
        "provenance": "disk_checkpoint:federated_global_model.pt",
        "rounds_completed": 10,
    }
    temp_store.record_checkpoint(ckpt)

    latest = temp_store.get_latest_checkpoint()
    assert latest is not None
    assert latest["model_type"] == "uni"
    assert latest["checkpoint_hash"] == "a1b2c3d4e5f67890"
    assert latest["rounds_completed"] == 10


def test_state_store_clear(temp_store: PersistentStateStore):
    """Verify state store clear operation wipes tables cleanly."""
    temp_store.save_system_state({"model_trained": True})
    temp_store.record_training_round({"round_num": 1, "train_loss": 0.5})

    stats_before = temp_store.get_database_stats()
    assert stats_before["rounds_count"] == 1

    temp_store.clear_state()

    stats_after = temp_store.get_database_stats()
    assert stats_after["rounds_count"] == 0
    assert temp_store.load_system_state() == {}


def test_app_state_hydration_across_restart(tmp_path: Path):
    """Verify AppState hydration restores consortium state from SQLite after restart."""
    db_file = tmp_path / "hydration_consortium.db"
    store = PersistentStateStore(db_path=db_file)

    # 1. Simulate active session saving cohorts and training history
    cohorts_data = [
        {
            "hospital_id": 1,
            "name": "Barcelona Clinic",
            "num_samples": 50,
            "distribution": [0.4, 0.3, 0.3],
            "counts": [20, 15, 15],
            "sample_images": [[[0.1, 0.2], [0.3, 0.4]]],
            "sample_labels": [0],
            "provenance": "isic_archive_v2_real_clinical",
            "is_authentic_clinical": True,
        }
    ]
    store.save_hospital_cohorts(cohorts_data)
    store.save_system_state({
        "model_type": "vit_tiny",
        "peft_mode": "lora",
        "defense_mode": "none",
        "image_size": 224,
        "data_source": "isic_archive_v2_real_clinical",
        "data_provenance": "isic_archive_v2_real_clinical",
        "is_authentic_clinical": True,
        "strict_real_data": True,
        "model_trained": True,
        "rounds_completed": 3,
        "privacy_spent": 0.85,
    })
    store.record_training_round({
        "round_num": 1,
        "train_loss": 0.7,
        "train_accuracy": 75.0,
        "test_loss": 0.65,
        "test_accuracy": 78.0,
        "precision": 78.0,
        "recall": 78.0,
        "f1_score": 78.0,
        "privacy_epsilon": 0.85,
        "status": "completed",
    })

    # 2. Simulate process restart: instantiate fresh AppState pointing to the same db
    fresh_app_state = AppState()
    fresh_app_state.store = store
    assert len(fresh_app_state.hospital_cohort_info) == 0

    # 3. Call restore_from_disk() and verify hydration
    restored = fresh_app_state.restore_from_disk(checkpoint_dir=str(tmp_path))
    assert restored is True
    assert fresh_app_state.model_type == "vit_tiny"
    assert fresh_app_state.peft_mode == "lora"
    assert fresh_app_state.is_authentic_clinical is True
    assert len(fresh_app_state.hospital_cohort_info) == 1
    assert fresh_app_state.hospital_cohort_info[0].name == "Barcelona Clinic"
    assert fresh_app_state.training_history["round"] == [1]
    assert fresh_app_state.training_history["loss"] == [0.7]
    assert fresh_app_state.training_history["test_loss"] == [0.65]


def test_api_state_and_history_endpoints():
    """Verify FastAPI state, history, audit trail, and reset endpoints."""
    client = TestClient(app)

    # 1. GET /api/state returns persistence telemetry
    resp = client.get("/api/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "persistence" in data
    assert data["persistence"]["backend"] == "sqlite_wal"
    assert "stats" in data["persistence"]
    assert data["persistence"]["stats"]["wal_mode_enabled"] is True

    # 2. GET /api/state/history returns history structure
    hist_resp = client.get("/api/state/history")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert "history" in hist_data
    assert "total_rounds" in hist_data

    # 3. GET /api/diagnose/audit-trail returns audit trail
    audit_resp = client.get("/api/diagnose/audit-trail")
    assert audit_resp.status_code == 200
    audit_data = audit_resp.json()
    assert "audits" in audit_data
    assert "total_records" in audit_data

    # 4. POST /api/state/reset clears persistent state
    reset_resp = client.post("/api/state/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "reset"
