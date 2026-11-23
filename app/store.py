"""SQLite WAL-backed task persistence store for long-audio async transcription jobs."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class TaskStore:
    """Thread-safe SQLite store for tracking transcription tasks and results."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    filename TEXT,
                    duration_sec REAL,
                    total_chunks INTEGER NOT NULL DEFAULT 0,
                    completed_chunks INTEGER NOT NULL DEFAULT 0,
                    progress REAL NOT NULL DEFAULT 0.0,
                    language_requested TEXT,
                    context TEXT NOT NULL DEFAULT '',
                    return_timestamps INTEGER NOT NULL DEFAULT 1,
                    include_words INTEGER NOT NULL DEFAULT 1,
                    webhook_url TEXT,
                    result_json TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks (created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status)"
            )

    def mark_interrupted_tasks_failed(self) -> int:
        """Mark any leftover in-flight tasks from a previous crash/restart as failed."""
        now = time.time()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE tasks
                SET status = 'failed',
                    error = 'Server restarted while task was in progress.',
                    updated_at = ?,
                    completed_at = ?
                WHERE status IN ('queued', 'decoding', 'processing')
                """,
                (now, now),
            )
            return cur.rowcount

    def create_task(
        self,
        *,
        task_id: str,
        filename: str,
        language_requested: Optional[str],
        context: str,
        return_timestamps: bool,
        include_words: bool,
        webhook_url: Optional[str],
    ) -> Dict[str, Any]:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    task_id, status, filename, language_requested, context,
                    return_timestamps, include_words, webhook_url,
                    created_at, updated_at
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    filename,
                    language_requested,
                    context,
                    1 if return_timestamps else 0,
                    1 if include_words else 0,
                    webhook_url,
                    now,
                    now,
                ),
            )
        return self.get_task(task_id)  # type: ignore[return-value]

    def update_status(
        self,
        task_id: str,
        *,
        status: str,
        duration_sec: Optional[float] = None,
        total_chunks: Optional[int] = None,
        completed_chunks: Optional[int] = None,
        progress: Optional[float] = None,
    ) -> None:
        now = time.time()
        fields = ["status = ?", "updated_at = ?"]
        params: List[Any] = [status, now]

        if duration_sec is not None:
            fields.append("duration_sec = ?")
            params.append(duration_sec)
        if total_chunks is not None:
            fields.append("total_chunks = ?")
            params.append(total_chunks)
        if completed_chunks is not None:
            fields.append("completed_chunks = ?")
            params.append(completed_chunks)
        if progress is not None:
            fields.append("progress = ?")
            params.append(max(0.0, min(1.0, float(progress))))

        params.append(task_id)
        with self._lock, self._connect() as conn:
            conn.execute(
                f"UPDATE tasks SET {', '.join(fields)} WHERE task_id = ?",
                params,
            )

    def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = 'completed',
                    progress = 1.0,
                    completed_chunks = total_chunks,
                    result_json = ?,
                    error = NULL,
                    updated_at = ?,
                    completed_at = ?
                WHERE task_id = ?
                """,
                (json.dumps(result, ensure_ascii=False), now, now, task_id),
            )

    def fail_task(self, task_id: str, error: str) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = 'failed',
                    error = ?,
                    updated_at = ?,
                    completed_at = ?
                WHERE task_id = ?
                """,
                (error, now, now, task_id),
            )

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def list_tasks(
        self, limit: int = 50, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._row_to_dict(r, include_result=False) for r in rows]

    def delete_task(self, task_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            return cur.rowcount > 0

    def cleanup_expired_tasks(self, retention_hours: int) -> int:
        cutoff = time.time() - retention_hours * 3600.0
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM tasks
                WHERE status IN ('completed', 'failed')
                  AND completed_at IS NOT NULL
                  AND completed_at < ?
                """,
                (cutoff,),
            )
            return cur.rowcount

    @staticmethod
    def _row_to_dict(
        row: sqlite3.Row, include_result: bool = True
    ) -> Dict[str, Any]:
        result = None
        if include_result and row["result_json"]:
            result = json.loads(row["result_json"])
        return {
            "task_id": row["task_id"],
            "status": row["status"],
            "filename": row["filename"],
            "duration_sec": row["duration_sec"],
            "total_chunks": row["total_chunks"],
            "completed_chunks": row["completed_chunks"],
            "progress": round(float(row["progress"]), 4),
            "language_requested": row["language_requested"],
            "return_timestamps": bool(row["return_timestamps"]),
            "include_words": bool(row["include_words"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
            "error": row["error"],
            "result": result,
        }
