import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLitePipelineStateStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self._lock:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS pipeline_journal (
                    job_id TEXT PRIMARY KEY,
                    phase TEXT NOT NULL,
                    payload TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pipeline_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            self.connection.commit()

    def create_job(self, job_id: str, payload: Dict[str, Any]) -> None:
        now = _utcnow()
        with self._lock:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO pipeline_journal (job_id, phase, payload, last_error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, "prepare", json.dumps(payload, default=str), None, now, now),
            )
            self.connection.commit()

    def update_phase(
        self,
        job_id: str,
        phase: str,
        payload_patch: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        record = self.get_job(job_id) or {"payload": {}}
        payload = record.get("payload", {})
        if payload_patch:
            payload.update(payload_patch)
        with self._lock:
            self.connection.execute(
                """
                UPDATE pipeline_journal
                SET phase = ?, payload = ?, last_error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (phase, json.dumps(payload, default=str), error, _utcnow(), job_id),
            )
            self.connection.commit()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.connection.execute(
                "SELECT job_id, phase, payload, last_error, created_at, updated_at FROM pipeline_journal WHERE job_id = ?",
                (job_id,),
            ).fetchone()

        if not row:
            return None
        payload = json.loads(row[2]) if row[2] else {}
        return {
            "job_id": row[0],
            "phase": row[1],
            "payload": payload,
            "last_error": row[3],
            "created_at": row[4],
            "updated_at": row[5],
        }

    def append_event(self, job_id: str, stage_name: str, event_type: str, payload: Optional[Dict[str, Any]]) -> None:
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO pipeline_events (job_id, stage_name, event_type, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, stage_name, event_type, json.dumps(payload or {}, default=str), _utcnow()),
            )
            self.connection.commit()

    def list_events(self, job_id: str) -> list[Dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT stage_name, event_type, payload, created_at
                FROM pipeline_events
                WHERE job_id = ?
                ORDER BY id ASC
                """,
                (job_id,),
            ).fetchall()
        return [
            {
                "stage_name": stage_name,
                "event_type": event_type,
                "payload": json.loads(payload) if payload else {},
                "created_at": created_at,
            }
            for stage_name, event_type, payload, created_at in rows
        ]
