"""Persistent state and ACID database storage engine for Fed-XRay.

Replaces volatile in-process state with a persistent SQLite storage engine in Write-Ahead
Logging (WAL) mode. Guarantees state survivability across process restarts, multi-worker
concurrency, and container redeployments.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger("fed_xray.storage")


class PersistentStateStore:
    """ACID-compliant persistent storage engine for federated oncology state."""

    def __init__(self, db_path: Path | str = "data/state/consortium_state.db") -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextlib.contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a managed, thread-safe SQLite connection with WAL mode."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=15.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize relational schema with indexes if not already created."""
        with self._connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS training_rounds (
                round_num INTEGER PRIMARY KEY,
                train_loss REAL NOT NULL,
                train_accuracy REAL NOT NULL,
                test_loss REAL NOT NULL,
                test_accuracy REAL NOT NULL,
                precision REAL NOT NULL,
                recall REAL NOT NULL,
                f1_score REAL NOT NULL,
                privacy_epsilon REAL NOT NULL,
                uplink_bytes INTEGER NOT NULL DEFAULT 0,
                threat_detected INTEGER NOT NULL DEFAULT 0,
                blocked_nodes TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'completed',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hospital_cohorts (
                hospital_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                num_samples INTEGER NOT NULL,
                distribution_json TEXT NOT NULL,
                counts_json TEXT NOT NULL,
                sample_images_json TEXT NOT NULL,
                sample_labels_json TEXT NOT NULL,
                provenance TEXT NOT NULL,
                is_authentic_clinical INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS diagnostic_audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                predicted_class INTEGER NOT NULL,
                predicted_name TEXT NOT NULL,
                true_class INTEGER NOT NULL,
                true_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities_json TEXT NOT NULL,
                findings TEXT NOT NULL,
                image_source TEXT NOT NULL,
                provenance TEXT NOT NULL,
                biopsy_confirmed INTEGER NOT NULL DEFAULT 1,
                anatomical_site TEXT,
                raw_image_json TEXT,
                heatmap_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT NOT NULL,
                peft_mode TEXT,
                checkpoint_path TEXT NOT NULL,
                checkpoint_hash TEXT,
                is_pretrained INTEGER NOT NULL DEFAULT 1,
                provenance TEXT NOT NULL,
                rounds_completed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_training_rounds_round ON training_rounds(round_num);
            CREATE INDEX IF NOT EXISTS idx_diagnostic_audits_patient ON diagnostic_audits(patient_id);
            CREATE INDEX IF NOT EXISTS idx_checkpoints_created ON model_checkpoints(created_at);
            """)
        logger.info("Persistent state database initialized at %s", self.db_path)

    def save_system_state(self, state_dict: Dict[str, Any]) -> None:
        """Upsert system-level configuration and runtime flags into SQLite."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            for k, v in state_dict.items():
                val_str = json.dumps(v) if isinstance(v, (dict, list, bool)) else str(v)
                conn.execute(
                    """
                    INSERT INTO system_state (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    (k, val_str, now),
                )

    def load_system_state(self) -> Dict[str, Any]:
        """Load all system-level configuration parameters from SQLite."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT key, value FROM system_state")
            rows = cursor.fetchall()

        result: Dict[str, Any] = {}
        for row in rows:
            k = row["key"]
            val_raw = row["value"]
            try:
                result[k] = json.loads(val_raw)
            except (json.JSONDecodeError, TypeError):
                result[k] = val_raw
        return result

    def record_training_round(self, round_data: Dict[str, Any]) -> None:
        """Persist metrics of a completed federated round transactionally."""
        now = datetime.now(timezone.utc).isoformat()
        round_num = int(round_data["round_num"])
        blocked = round_data.get("blocked_nodes", [])
        blocked_json = json.dumps(blocked if isinstance(blocked, list) else [])

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO training_rounds (
                    round_num, train_loss, train_accuracy, test_loss, test_accuracy,
                    precision, recall, f1_score, privacy_epsilon, uplink_bytes,
                    threat_detected, blocked_nodes, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(round_num) DO UPDATE SET
                    train_loss = excluded.train_loss,
                    train_accuracy = excluded.train_accuracy,
                    test_loss = excluded.test_loss,
                    test_accuracy = excluded.test_accuracy,
                    precision = excluded.precision,
                    recall = excluded.recall,
                    f1_score = excluded.f1_score,
                    privacy_epsilon = excluded.privacy_epsilon,
                    uplink_bytes = excluded.uplink_bytes,
                    threat_detected = excluded.threat_detected,
                    blocked_nodes = excluded.blocked_nodes,
                    status = excluded.status,
                    created_at = excluded.created_at
                """,
                (
                    round_num,
                    float(round_data.get("train_loss", 0.0)),
                    float(round_data.get("train_accuracy", 0.0)),
                    float(round_data.get("test_loss", 0.0)),
                    float(round_data.get("test_accuracy", 0.0)),
                    float(round_data.get("precision", 0.0)),
                    float(round_data.get("recall", 0.0)),
                    float(round_data.get("f1_score", 0.0)),
                    float(round_data.get("privacy_epsilon", 0.0)),
                    int(round_data.get("uplink_bytes", 0)),
                    1 if round_data.get("threat_detected") else 0,
                    blocked_json,
                    str(round_data.get("status", "completed")),
                    now,
                ),
            )

    def get_training_history(self) -> Dict[str, Any]:
        """Reconstruct the canonical training_history dictionary from SQLite records."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT round_num, train_loss, train_accuracy, test_loss, test_accuracy,
                       precision, recall, f1_score, privacy_epsilon, threat_detected, blocked_nodes
                FROM training_rounds
                ORDER BY round_num ASC
                """
            )
            rows = cursor.fetchall()

        history: Dict[str, Any] = {
            "round": [],
            "loss": [],
            "accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "privacy_epsilon": [],
            "threat_detected": [],
            "blocked_nodes": [],
            "blocked_count": 0,
        }

        for r in rows:
            history["round"].append(int(r["round_num"]))
            history["loss"].append(float(r["train_loss"]))
            history["accuracy"].append(float(r["train_accuracy"]))
            history["test_loss"].append(float(r["test_loss"]))
            history["test_accuracy"].append(float(r["test_accuracy"]))
            history["precision"].append(float(r["precision"]))
            history["recall"].append(float(r["recall"]))
            history["f1_score"].append(float(r["f1_score"]))
            history["privacy_epsilon"].append(float(r["privacy_epsilon"]))
            history["threat_detected"].append(bool(r["threat_detected"]))
            try:
                blocked = json.loads(r["blocked_nodes"])
                blocked_parsed = blocked if isinstance(blocked, list) else []
                history["blocked_nodes"].append(blocked_parsed)
                history["blocked_count"] += len(blocked_parsed)
            except Exception:
                history["blocked_nodes"].append([])

        history["privacy_spent"] = history["privacy_epsilon"][-1] if history["privacy_epsilon"] else 0.0
        return history

    def save_hospital_cohorts(self, cohorts_info: List[Dict[str, Any]]) -> None:
        """Persist hospital cohort allocations and preview samples to SQLite."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            for c in cohorts_info:
                hid = int(c["hospital_id"])
                name = str(c["name"])
                num_s = int(c["num_samples"])
                dist_json = json.dumps(c.get("distribution", []))

                # Normalize counts to Dict[str, int] for Pydantic schema validation
                raw_counts = c.get("counts", {})
                if isinstance(raw_counts, list):
                    normalized_counts = {str(i): int(cnt) for i, cnt in enumerate(raw_counts)}
                elif isinstance(raw_counts, dict):
                    normalized_counts = {str(k): int(v) for k, v in raw_counts.items()}
                else:
                    normalized_counts = {}
                counts_json = json.dumps(normalized_counts)

                sample_imgs_json = json.dumps(c.get("sample_images", []))
                sample_lbls_json = json.dumps(c.get("sample_labels", []))
                provenance = str(c.get("provenance", "strategy_e_disk_manifest"))
                is_auth = 1 if c.get("is_authentic_clinical", True) else 0

                conn.execute(
                    """
                    INSERT INTO hospital_cohorts (
                        hospital_id, name, num_samples, distribution_json, counts_json,
                        sample_images_json, sample_labels_json, provenance,
                        is_authentic_clinical, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(hospital_id) DO UPDATE SET
                        name = excluded.name,
                        num_samples = excluded.num_samples,
                        distribution_json = excluded.distribution_json,
                        counts_json = excluded.counts_json,
                        sample_images_json = excluded.sample_images_json,
                        sample_labels_json = excluded.sample_labels_json,
                        provenance = excluded.provenance,
                        is_authentic_clinical = excluded.is_authentic_clinical,
                        updated_at = excluded.updated_at
                    """,
                    (
                        hid,
                        name,
                        num_s,
                        dist_json,
                        counts_json,
                        sample_imgs_json,
                        sample_lbls_json,
                        provenance,
                        is_auth,
                        now,
                    ),
                )

    def load_hospital_cohorts(self) -> List[Dict[str, Any]]:
        """Retrieve persisted hospital cohort records from SQLite."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT hospital_id, name, num_samples, distribution_json, counts_json,
                       sample_images_json, sample_labels_json, provenance, is_authentic_clinical
                FROM hospital_cohorts
                ORDER BY hospital_id ASC
                """
            )
            rows = cursor.fetchall()

        result = []
        for r in rows:
            parsed_counts = json.loads(r["counts_json"]) if r["counts_json"] else {}
            if isinstance(parsed_counts, list):
                counts_dict = {str(i): int(c) for i, c in enumerate(parsed_counts)}
            elif isinstance(parsed_counts, dict):
                counts_dict = {str(k): int(v) for k, v in parsed_counts.items()}
            else:
                counts_dict = {}

            result.append({
                "hospital_id": r["hospital_id"],
                "name": r["name"],
                "num_samples": r["num_samples"],
                "distribution": json.loads(r["distribution_json"]),
                "counts": counts_dict,
                "sample_images": json.loads(r["sample_images_json"]),
                "sample_labels": json.loads(r["sample_labels_json"]),
                "provenance": r["provenance"],
                "is_authentic_clinical": bool(r["is_authentic_clinical"]),
            })
        return result

    def record_diagnostic_audit(self, audit: Dict[str, Any]) -> int:
        """Persist clinical decision support diagnosis, XAI explanation, and patient record."""
        now = datetime.now(timezone.utc).isoformat()
        probs_json = json.dumps(audit.get("probabilities", []))
        raw_json = json.dumps(audit.get("raw_image")) if audit.get("raw_image") is not None else None
        heatmap_json = json.dumps(audit.get("heatmap")) if audit.get("heatmap") is not None else None
        is_biopsy = 1 if audit.get("biopsy_confirmed", True) else 0

        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO diagnostic_audits (
                    patient_id, predicted_class, predicted_name, true_class, true_name,
                    confidence, probabilities_json, findings, image_source, provenance,
                    biopsy_confirmed, anatomical_site, raw_image_json, heatmap_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(audit.get("patient_id", "PAT_UNKNOWN")),
                    int(audit.get("predicted_class", 0)),
                    str(audit.get("predicted_name", "")),
                    int(audit.get("true_class", 0)),
                    str(audit.get("true_name", "")),
                    float(audit.get("confidence", 0.0)),
                    probs_json,
                    str(audit.get("findings", "")),
                    str(audit.get("image_source", "uploaded")),
                    str(audit.get("provenance", "isic_archive_v2_real_clinical")),
                    is_biopsy,
                    str(audit.get("anatomical_site", "Torso")),
                    raw_json,
                    heatmap_json,
                    now,
                ),
            )
            return cursor.lastrowid or 0

    def get_last_diagnosis(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent clinical diagnosis record."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, patient_id, predicted_class, predicted_name, true_class, true_name,
                       confidence, probabilities_json, findings, image_source, provenance,
                       biopsy_confirmed, anatomical_site, raw_image_json, heatmap_json, created_at
                FROM diagnostic_audits
                ORDER BY id DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "patient_id": row["patient_id"],
            "predicted_class": row["predicted_class"],
            "predicted_name": row["predicted_name"],
            "true_class": row["true_class"],
            "true_name": row["true_name"],
            "confidence": row["confidence"],
            "probabilities": json.loads(row["probabilities_json"]),
            "findings": row["findings"],
            "image_source": row["image_source"],
            "provenance": row["provenance"],
            "biopsy_confirmed": bool(row["biopsy_confirmed"]),
            "anatomical_site": row["anatomical_site"],
            "raw_image": json.loads(row["raw_image_json"]) if row["raw_image_json"] else None,
            "heatmap": json.loads(row["heatmap_json"]) if row["heatmap_json"] else None,
            "created_at": row["created_at"],
        }

    def get_diagnostic_audits(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve historical clinical audit trail sorted chronologically descending."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, patient_id, predicted_class, predicted_name, true_class, true_name,
                       confidence, findings, image_source, provenance, biopsy_confirmed,
                       anatomical_site, created_at
                FROM diagnostic_audits
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        result = []
        for r in rows:
            result.append({
                "id": r["id"],
                "patient_id": r["patient_id"],
                "predicted_class": r["predicted_class"],
                "predicted_name": r["predicted_name"],
                "true_class": r["true_class"],
                "true_name": r["true_name"],
                "confidence": r["confidence"],
                "findings": r["findings"],
                "image_source": r["image_source"],
                "provenance": r["provenance"],
                "biopsy_confirmed": bool(r["biopsy_confirmed"]),
                "anatomical_site": r["anatomical_site"],
                "created_at": r["created_at"],
            })
        return result

    def record_checkpoint(self, checkpoint_info: Dict[str, Any]) -> None:
        """Register a saved model weights checkpoint and SHA-256 fingerprint."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO model_checkpoints (
                    model_type, peft_mode, checkpoint_path, checkpoint_hash,
                    is_pretrained, provenance, rounds_completed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(checkpoint_info.get("model_type", "vit_tiny")),
                    str(checkpoint_info.get("peft_mode", "ffa_lora")),
                    str(checkpoint_info.get("checkpoint_path", "")),
                    str(checkpoint_info.get("checkpoint_hash", "")),
                    1 if checkpoint_info.get("is_pretrained", True) else 0,
                    str(checkpoint_info.get("provenance", "local_checkpoint")),
                    int(checkpoint_info.get("rounds_completed", 0)),
                    now,
                ),
            )

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Retrieve the latest registered model checkpoint record."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, model_type, peft_mode, checkpoint_path, checkpoint_hash,
                       is_pretrained, provenance, rounds_completed, created_at
                FROM model_checkpoints
                ORDER BY id DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "model_type": row["model_type"],
            "peft_mode": row["peft_mode"],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_hash": row["checkpoint_hash"],
            "is_pretrained": bool(row["is_pretrained"]),
            "provenance": row["provenance"],
            "rounds_completed": row["rounds_completed"],
            "created_at": row["created_at"],
        }

    def clear_state(self) -> None:
        """Clear all tables in SQLite (used for clean resets)."""
        with self._connection() as conn:
            conn.executescript("""
            DELETE FROM system_state;
            DELETE FROM training_rounds;
            DELETE FROM hospital_cohorts;
            DELETE FROM diagnostic_audits;
            DELETE FROM model_checkpoints;
            """)
        logger.info("Persistent state database cleared at %s", self.db_path)

    def get_database_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics of the SQLite database."""
        with self._connection() as conn:
            mode_row = conn.execute("PRAGMA journal_mode;").fetchone()
            mode = str(mode_row[0]).lower() if mode_row else "unknown"
            n_rounds = conn.execute("SELECT COUNT(*) FROM training_rounds").fetchone()[0]
            n_cohorts = conn.execute("SELECT COUNT(*) FROM hospital_cohorts").fetchone()[0]
            n_audits = conn.execute("SELECT COUNT(*) FROM diagnostic_audits").fetchone()[0]
            n_checkpoints = conn.execute("SELECT COUNT(*) FROM model_checkpoints").fetchone()[0]

        file_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "database_path": str(self.db_path),
            "file_size_bytes": file_size,
            "journal_mode": mode,
            "wal_mode_enabled": mode == "wal",
            "persisted_rounds_count": n_rounds,
            "persisted_cohorts_count": n_cohorts,
            "persisted_audits_count": n_audits,
            "persisted_checkpoints_count": n_checkpoints,
            "rounds_count": n_rounds,
            "cohorts_count": n_cohorts,
            "audits_count": n_audits,
            "checkpoints_count": n_checkpoints,
            "persistence_backend": "sqlite_wal",
        }
