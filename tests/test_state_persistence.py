"""Tests for AppState checkpoint persistence, recovery, and /api/state endpoints.

Adheres strictly to Rule 12 (Zero Fake Results) and Rule 18 (Contract Consistency).
"""

from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

from api.main import app, state, AppState
from src.fed_xray.core.server import CentralServer


client = TestClient(app)


def test_app_state_save_fails_when_untrained(tmp_path: Path):
    """Saving untrained AppState returns False and creates no files."""
    s = AppState()
    s.model_trained = False
    s.server = None
    save_dir = str(tmp_path / "untrained_ckpt")
    assert not s.save_to_disk(checkpoint_dir=save_dir)
    assert not (Path(save_dir) / "federated_global_model.pt").exists()


def test_app_state_save_and_restore_roundtrip(tmp_path: Path):
    """Verify weights, metadata, and history survive a fresh process state."""
    save_dir = str(tmp_path / "test_checkpoints")

    # 1. Setup trained state
    s1 = AppState()
    server1 = CentralServer(
        device=torch.device("cpu"),
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        img_size=224,
    )
    s1.server = server1
    s1.model_trained = True
    s1.model_type = "vit_tiny"
    s1.peft_mode = "ffa_lora"
    s1.image_size = 224
    s1.data_source = "strategy_e_test_cohort"
    s1.training_history = {
        "round": [1, 2],
        "loss": [0.85, 0.45],
        "accuracy": [75.0, 92.5],
        "test_loss": [0.80, 0.40],
        "test_accuracy": [78.0, 93.0],
    }

    # 2. Save
    assert s1.save_to_disk(checkpoint_dir=save_dir)
    assert (Path(save_dir) / "federated_global_model.pt").exists()
    assert (Path(save_dir) / "federated_state_metadata.json").exists()

    # 3. Restore in fresh AppState instance
    s2 = AppState()
    assert s2.restore_from_disk(checkpoint_dir=save_dir)

    assert s2.model_trained is True
    assert s2.model_type == "vit_tiny"
    assert s2.peft_mode == "ffa_lora"
    assert s2.image_size == 224
    assert s2.data_source == "strategy_e_test_cohort"
    assert s2.training_history["round"] == [1, 2]
    assert s2.server is not None

    # Verify weight tensor equality
    w1 = server1.global_model.state_dict()
    w2 = s2.server.global_model.state_dict()
    assert set(w1.keys()) == set(w2.keys())
    for k in w1.keys():
        assert torch.allclose(w1[k].cpu(), w2[k].cpu(), atol=1e-5)


def test_api_state_endpoint_fields():
    """Verify /api/state returns persistence and data source contract."""
    res = client.get("/api/state")
    assert res.status_code == 200
    data = res.json()
    assert "model_trained" in data
    assert "hospitals_loaded" in data
    assert "has_test_set" in data
    assert "history_rounds" in data
    assert "model_type" in data
    assert "peft_mode" in data
    assert "persisted_on_disk" in data
    assert "data_source" in data


def test_api_state_save_and_restore_endpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify POST /api/state/save and POST /api/state/restore API contracts."""
    ckpt_dir = str(tmp_path / "api_checkpoints")
    monkeypatch.setattr(state, "model_trained", False)
    monkeypatch.setattr(state, "server", None)

    # 1. Save fails when untrained
    res_save_fail = client.post("/api/state/save")
    assert res_save_fail.status_code == 400

    # 2. Train and save via endpoint
    server = CentralServer(device=torch.device("cpu"), model_type="vit_tiny", peft_mode="ffa_lora", img_size=224)
    monkeypatch.setattr(state, "server", server)
    monkeypatch.setattr(state, "model_trained", True)
    orig_save = state.save_to_disk
    orig_restore = state.restore_from_disk
    monkeypatch.setattr(state, "save_to_disk", lambda **kwargs: orig_save(checkpoint_dir=ckpt_dir))
    monkeypatch.setattr(state, "restore_from_disk", lambda **kwargs: orig_restore(checkpoint_dir=ckpt_dir))

    res_save = client.post("/api/state/save")
    assert res_save.status_code == 200
    assert res_save.json()["status"] == "saved"

    # 3. Restore via endpoint
    res_restore = client.post("/api/state/restore")
    assert res_restore.status_code == 200
    data = res_restore.json()
    assert data["status"] == "restored"
    assert data["model_type"] == "vit_tiny"
