"""SQLite task persistence, serialized on one dedicated database thread."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from functools import partial
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, TypeVar


_T = TypeVar("_T")
_ACTIVE = "'queued', 'decoding', 'processing'"
_IN_FLIGHT = f"{_ACTIVE}, 'cancelling'"
_SCHEMA_VERSION = 1
_METADATA_COLUMNS = """task_id, status, filename, duration_sec, total_chunks,
    completed_chunks, progress, language_requested, context, return_timestamps,
    include_words, webhook_url, error, created_at, updated_at, completed_at,
    error_code, cancel_requested_at, deadline_at"""


class TaskStore:
    """Await start before use and close at shutdown; SQLite never runs on the loop.

    Terminal transitions return whether they won the conditional update. A cancel
    request prevents completion; failure may still terminate a cancelling task
    when its worker or cleanup fails.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="task-store")
        self._lifecycle_lock = asyncio.Lock()
        self._started = False
        self._closed = False

    async def _submit(self, operation: Callable[[], _T]) -> _T:
        return await asyncio.get_running_loop().run_in_executor(self._executor, operation)

    async def _run(self, operation: Callable[[], _T]) -> _T:
        if not self._started or self._closed:
            raise RuntimeError("TaskStore must be started and open")
        return await self._submit(operation)

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("TaskStore is closed")
            if not self._started:
                await self._submit(self._init_db)
                self._started = True

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            # This barrier follows all submitted operations on the only DB thread.
            # Connections close within each operation before the barrier completes.
            try:
                await self._submit(lambda: None)
            finally:
                self._executor.shutdown(wait=False)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(str(self.db_path), timeout=30.0)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            with conn:
                yield conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            # BEGIN makes DDL and user_version one atomic migration, including
            # rollback if a later ALTER fails. SQLite DDL alone does not begin it.
            conn.execute("BEGIN IMMEDIATE")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > _SCHEMA_VERSION:
                raise RuntimeError(f"Unsupported task database schema version {version}")
            if version == 0:
                conn.execute("""
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
                """)
                for name, sql_type in (
                    ("error_code", "TEXT"),
                    ("cancel_requested_at", "REAL"),
                    ("deadline_at", "REAL"),
                ):
                    conn.execute(f"ALTER TABLE tasks ADD COLUMN {name} {sql_type}")
                conn.execute("PRAGMA user_version = 1")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks (created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status)")

    def _execute_update(self, sql: str, params: Any) -> int:
        with self._connection() as conn:
            return conn.execute(sql, params).rowcount

    async def mark_interrupted_tasks_failed(self) -> int:
        now = time.time()
        return await self._run(partial(self._execute_update, f"""
            UPDATE tasks SET status = 'failed',
                error = 'Server restarted while task was in progress.',
                error_code = 'SERVER_RESTARTED', updated_at = ?, completed_at = ?
            WHERE status IN ({_IN_FLIGHT})
        """, (now, now)))

    async def create_task(
        self, *, task_id: str, filename: str, language_requested: Optional[str],
        context: str, return_timestamps: bool, include_words: bool,
        webhook_url: Optional[str], deadline_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        def create() -> Dict[str, Any]:
            now = time.time()
            with self._connection() as conn:
                conn.execute("""
                    INSERT INTO tasks (task_id, status, filename,
                        language_requested, context, return_timestamps,
                        include_words, webhook_url, deadline_at, created_at, updated_at)
                    VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (task_id, filename, language_requested, context,
                      int(return_timestamps), int(include_words), webhook_url,
                      deadline_at, now, now))
                row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
                return self._row_to_dict(row)
        return await self._run(create)

    async def update_status(
        self, task_id: str, *, status: str,
        duration_sec: Optional[float] = None, total_chunks: Optional[int] = None,
        completed_chunks: Optional[int] = None, progress: Optional[float] = None,
    ) -> bool:
        if status not in {"queued", "decoding", "processing"}:
            raise ValueError("Use complete_task, fail_task or cancellation methods for terminal transitions")
        fields = ["status = ?", "updated_at = ?"]
        params: List[Any] = [status, time.time()]
        for name, value in (("duration_sec", duration_sec), ("total_chunks", total_chunks),
                            ("completed_chunks", completed_chunks), ("progress", progress)):
            if value is not None:
                fields.append(f"{name} = ?")
                params.append(max(0.0, min(1.0, float(value))) if name == "progress" else value)
        params.append(task_id)
        return bool(await self._run(partial(self._execute_update,
            f"UPDATE tasks SET {', '.join(fields)} WHERE task_id = ? AND status IN ({_ACTIVE})", params)))

    async def request_cancel(self, task_id: str) -> bool:
        now = time.time()
        return bool(await self._run(partial(self._execute_update, f"""
            UPDATE tasks SET status = 'cancelling', cancel_requested_at = ?, updated_at = ?
            WHERE task_id = ? AND status IN ({_ACTIVE})
        """, (now, now, task_id))))

    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        def complete() -> int:
            now = time.time()
            return self._execute_update(f"""
                UPDATE tasks SET status = 'completed', progress = 1.0,
                    completed_chunks = total_chunks, result_json = ?, error = NULL,
                    error_code = NULL, updated_at = ?, completed_at = ?
                WHERE task_id = ? AND status IN ({_ACTIVE})
            """, (json.dumps(result, ensure_ascii=False), now, now, task_id))
        return bool(await self._run(complete))

    async def fail_task(self, task_id: str, error: str, error_code: str = "TASK_FAILED") -> bool:
        now = time.time()
        return bool(await self._run(partial(self._execute_update, f"""
            UPDATE tasks SET status = 'failed', error = ?, error_code = ?,
                updated_at = ?, completed_at = ?
            WHERE task_id = ? AND status IN ({_IN_FLIGHT})
        """, (error, error_code, now, now, task_id))))

    async def cancel_task(self, task_id: str) -> bool:
        now = time.time()
        return bool(await self._run(partial(self._execute_update, """
            UPDATE tasks SET status = 'cancelled', updated_at = ?, completed_at = ?
            WHERE task_id = ? AND status = 'cancelling'
        """, (now, now, task_id))))

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        def get() -> Optional[Dict[str, Any]]:
            with self._connection() as conn:
                row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
                return self._row_to_dict(row) if row else None
        return await self._run(get)

    async def list_tasks(self, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        def list_rows() -> List[Dict[str, Any]]:
            sql = f"SELECT {_METADATA_COLUMNS} FROM tasks"
            params: List[Any] = []
            if status:
                sql += " WHERE status = ?"
                params.append(status)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            with self._connection() as conn:
                return [self._row_to_dict(row, include_result=False)
                        for row in conn.execute(sql, params).fetchall()]
        return await self._run(list_rows)

    async def active_task_ids(self) -> Set[str]:
        def get_ids() -> Set[str]:
            with self._connection() as conn:
                return {row[0] for row in conn.execute(
                    f"SELECT task_id FROM tasks WHERE status IN ({_IN_FLIGHT})")}
        return await self._run(get_ids)

    async def delete_task(self, task_id: str) -> bool:
        return bool(await self._run(partial(self._execute_update,
            "DELETE FROM tasks WHERE task_id = ?", (task_id,))))

    async def cleanup_expired_tasks(self, retention_hours: int) -> int:
        cutoff = time.time() - retention_hours * 3600.0
        return await self._run(partial(self._execute_update, """
            DELETE FROM tasks WHERE status IN ('completed', 'failed', 'cancelled')
                AND completed_at IS NOT NULL AND completed_at < ?
        """, (cutoff,)))

    @staticmethod
    def _row_to_dict(row: sqlite3.Row, include_result: bool = True) -> Dict[str, Any]:
        result = json.loads(row["result_json"]) if include_result and row["result_json"] else None
        return {
            "task_id": row["task_id"], "status": row["status"],
            "filename": row["filename"], "duration_sec": row["duration_sec"],
            "total_chunks": row["total_chunks"], "completed_chunks": row["completed_chunks"],
            "progress": round(float(row["progress"]), 4),
            "language_requested": row["language_requested"],
            "return_timestamps": bool(row["return_timestamps"]),
            "include_words": bool(row["include_words"]),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "completed_at": row["completed_at"], "error": row["error"],
            "error_code": row["error_code"], "cancel_requested_at": row["cancel_requested_at"],
            "deadline_at": row["deadline_at"], "result": result,
        }
