"""Persistence tests exercise the same async interface used by job coordination."""

import asyncio
from contextlib import closing
import sqlite3
import threading
import time

import pytest

from app.store import TaskStore


async def create(store, task_id="task", **kwargs):
    return await store.create_task(
        task_id=task_id, filename="audio.wav", language_requested=None,
        context="", return_timestamps=True, include_words=True,
        webhook_url=None, **kwargs,
    )


def make_v0(path):
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("""
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY, status TEXT NOT NULL, filename TEXT,
                duration_sec REAL, total_chunks INTEGER NOT NULL DEFAULT 0,
                completed_chunks INTEGER NOT NULL DEFAULT 0,
                progress REAL NOT NULL DEFAULT 0.0, language_requested TEXT,
                context TEXT NOT NULL DEFAULT '', return_timestamps INTEGER NOT NULL DEFAULT 1,
                include_words INTEGER NOT NULL DEFAULT 1, webhook_url TEXT,
                result_json TEXT, error TEXT, created_at REAL NOT NULL,
                updated_at REAL NOT NULL, completed_at REAL)
        """)
        conn.execute("""INSERT INTO tasks (task_id,status,result_json,created_at,updated_at)
                        VALUES ('old','completed','{"text":"existing"}',1,2)""")


def test_v0_migration_preserves_existing_data_and_version(tmp_path):
    path = tmp_path / "tasks.db"
    make_v0(path)

    async def scenario():
        store = TaskStore(path)
        try:
            await store.start()
            row = await store.get_task("old")
            assert row["result"] == {"text": "existing"}
            assert row["created_at"] == 1
            assert row["error_code"] is None
            assert row["cancel_requested_at"] is None
            assert row["deadline_at"] is None
            fresh = await create(store, deadline_at=1234.5)
            assert fresh["deadline_at"] == 1234.5
            await store.start()  # Repeated startup does not run ALTER again.
        finally:
            await store.close()
        reopened = TaskStore(path)
        try:
            await reopened.start()
            assert (await reopened.get_task("old"))["result"] == {"text": "existing"}
        finally:
            await reopened.close()

    asyncio.run(scenario())
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        columns = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(tasks)")}
        assert columns["error_code"] == "TEXT"
        assert columns["cancel_requested_at"] == "REAL"
        assert columns["deadline_at"] == "REAL"


def test_newer_schema_is_rejected_without_version_change(tmp_path):
    path = tmp_path / "newer.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA user_version = 2")

    async def scenario():
        store = TaskStore(path)
        try:
            with pytest.raises(RuntimeError, match="schema version 2"):
                await store.start()
        finally:
            await store.close()

    asyncio.run(scenario())
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'tasks'").fetchone() is None


def test_migration_rolls_back_partial_alters(tmp_path):
    path = tmp_path / "partial.db"
    make_v0(path)
    # The second ALTER fails, so the first ALTER and version change must roll back.
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("ALTER TABLE tasks ADD COLUMN cancel_requested_at REAL")

    async def scenario():
        store = TaskStore(path)
        try:
            with pytest.raises(sqlite3.OperationalError, match="duplicate column"):
                await store.start()
        finally:
            await store.close()

    asyncio.run(scenario())
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
        columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
        assert "error_code" not in columns
        assert conn.execute("SELECT result_json FROM tasks").fetchone()[0] == '{"text":"existing"}'


def test_terminal_race_has_one_winner_and_late_updates_are_ignored(tmp_path):
    async def scenario():
        store = TaskStore(tmp_path / "tasks.db")
        await store.start()
        try:
            for i in range(10):
                task_id = str(i)
                await create(store, task_id)
                assert await store.update_status(task_id, status="processing", total_chunks=3, progress=0.4)
                transitions = [store.complete_task(task_id, {"text": "done"}),
                               store.fail_task(task_id, "failed", "WORKER_EXITED")]
                if i % 2:
                    transitions.reverse()
                outcomes = await asyncio.gather(*transitions)
                assert sum(outcomes) == 1
                row = await store.get_task(task_id)
                assert row["status"] in {"completed", "failed"}
                assert not await store.update_status(task_id, status="processing", progress=0.9)
                assert not await store.request_cancel(task_id)
                assert not await store.cancel_task(task_id)
                assert not await store.fail_task(task_id, "late")
                assert not await store.complete_task(task_id, {"text": "late"})
                assert await store.get_task(task_id) == row
        finally:
            await store.close()
    asyncio.run(scenario())


def test_cancel_completion_rules_and_failure_during_cancellation(tmp_path):
    async def scenario():
        store = TaskStore(tmp_path / "tasks.db")
        await store.start()
        try:
            await create(store)
            assert not await store.cancel_task("task")
            before = time.time()
            assert await store.request_cancel("task")
            row = await store.get_task("task")
            assert before <= row["cancel_requested_at"] <= time.time()
            assert not await store.request_cancel("task")
            assert not await store.complete_task("task", {})
            assert not await store.update_status("task", status="processing")
            assert await store.cancel_task("task")
            assert not await store.cancel_task("task")
            assert (await store.get_task("task"))["status"] == "cancelled"
            await create(store, "worker-failed")
            assert await store.request_cancel("worker-failed")
            assert await store.fail_task("worker-failed", "worker gone", "WORKER_EXITED")
            failed = await store.get_task("worker-failed")
            assert failed["error_code"] == "WORKER_EXITED"
            assert failed["status"] == "failed"
            assert not await store.cancel_task("worker-failed")
            await create(store, "default-error")
            assert await store.fail_task("default-error", "error")
            assert (await store.get_task("default-error"))["error_code"] == "TASK_FAILED"
            assert not await store.request_cancel("missing")
            assert not await store.complete_task("missing", {})
            with pytest.raises(ValueError):
                await store.update_status("task", status="completed")
        finally:
            await store.close()
    asyncio.run(scenario())


def test_completion_and_cancel_request_race(tmp_path):
    async def scenario():
        store = TaskStore(tmp_path / "tasks.db")
        await store.start()
        try:
            for index in range(10):
                task_id = str(index)
                await create(store, task_id)
                calls = [store.complete_task(task_id, {}), store.request_cancel(task_id)]
                if index % 2:
                    calls.reverse()
                assert sum(await asyncio.gather(*calls)) == 1
                row = await store.get_task(task_id)
                if row["status"] == "cancelling":
                    assert await store.cancel_task(task_id)
                else:
                    assert row["status"] == "completed"
        finally:
            await store.close()
    asyncio.run(scenario())


def test_restart_marks_all_inflight_and_leaves_terminal_records(tmp_path):
    async def scenario():
        store = TaskStore(tmp_path / "tasks.db")
        await store.start()
        try:
            for status in ("queued", "decoding", "processing", "cancelling", "completed", "failed", "cancelled"):
                await create(store, status)
                if status in {"decoding", "processing"}:
                    await store.update_status(status, status=status)
                elif status in {"cancelling", "cancelled"}:
                    await store.request_cancel(status)
                    if status == "cancelled":
                        await store.cancel_task(status)
                elif status == "completed":
                    await store.complete_task(status, {})
                elif status == "failed":
                    await store.fail_task(status, "original")
            assert await store.active_task_ids() == {"queued", "decoding", "processing", "cancelling"}
            assert await store.mark_interrupted_tasks_failed() == 4
            assert await store.active_task_ids() == set()
            assert await store.mark_interrupted_tasks_failed() == 0
            for status in ("queued", "decoding", "processing", "cancelling"):
                row = await store.get_task(status)
                assert row["status"] == "failed"
                assert row["error_code"] == "SERVER_RESTARTED"
                assert row["completed_at"] is not None
            assert (await store.get_task("completed"))["status"] == "completed"
            assert (await store.get_task("failed"))["error"] == "original"
            assert (await store.get_task("cancelled"))["status"] == "cancelled"
        finally:
            await store.close()
    asyncio.run(scenario())


def test_list_omits_result_column_and_cleanup_includes_cancelled(tmp_path, monkeypatch):
    statements = []
    original_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn
    monkeypatch.setattr(sqlite3, "connect", traced_connect)

    async def scenario():
        store = TaskStore(tmp_path / "tasks.db")
        await store.start()
        try:
            for name in ("completed", "failed", "cancelled", "active"):
                await create(store, name)
            await store.complete_task("completed", {"text": "large result"})
            await store.fail_task("failed", "failure")
            await store.request_cancel("cancelled")
            await store.cancel_task("cancelled")
            statements.clear()
            rows = await store.list_tasks(limit=3)
            assert len(rows) == 3
            assert all(row["result"] is None for row in rows)
            select = [sql for sql in statements if sql.startswith("SELECT")]
            assert len(select) == 1
            assert "result_json" not in select[0]
            assert "SELECT *" not in select[0]
            assert len(await store.list_tasks(status="failed")) == 1
            assert (await store.get_task("completed"))["result"] == {"text": "large result"}
            assert await store.cleanup_expired_tasks(24) == 0
            assert await store.cleanup_expired_tasks(0) == 3
            assert await store.active_task_ids() == {"active"}
            assert await store.delete_task("active")
            assert not await store.delete_task("active")
            assert await store.get_task("active") is None
        finally:
            await store.close()
    asyncio.run(scenario())


def test_db_worker_wait_does_not_block_event_loop(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    async def scenario():
        store = TaskStore(tmp_path / "tasks.db")
        await store.start()
        await create(store)
        original = store._execute_update

        def blocked(*args):
            entered.set()
            assert release.wait(timeout=5), "test did not release DB worker"
            return original(*args)
        monkeypatch.setattr(store, "_execute_update", blocked)
        pending = asyncio.create_task(store.update_status("task", status="processing"))
        try:
            assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), timeout=3)
            # A callback on the loop must run while the DB thread remains blocked.
            tick = asyncio.Event()
            asyncio.get_running_loop().call_soon(tick.set)
            await asyncio.wait_for(tick.wait(), timeout=0.5)
            assert not pending.done()
        finally:
            release.set()
            await pending
            await store.close()
    asyncio.run(scenario())


def test_connections_are_created_used_and_closed_on_one_thread(tmp_path, monkeypatch):
    events = []
    original_connect = sqlite3.connect
    main_thread = threading.get_ident()

    class ObservedConnection(sqlite3.Connection):
        def execute(self, *args, **kwargs):
            events.append(("execute", threading.get_ident()))
            return super().execute(*args, **kwargs)

        def close(self):
            events.append(("close", threading.get_ident()))
            return super().close()

    def connect(*args, **kwargs):
        events.append(("connect", threading.get_ident()))
        kwargs["factory"] = ObservedConnection
        return original_connect(*args, **kwargs)
    monkeypatch.setattr(sqlite3, "connect", connect)

    async def scenario():
        path = tmp_path / "nested" / "tasks.db"
        store = TaskStore(path)
        assert not path.parent.exists()
        assert not events
        with pytest.raises(RuntimeError, match="started"):
            await store.get_task("missing")
        await store.start()
        await create(store)
        await store.get_task("task")
        await store.list_tasks()
        await store.close()
        await store.close()
        with pytest.raises(RuntimeError, match="closed"):
            await store.start()
        with pytest.raises(RuntimeError, match="open"):
            await store.list_tasks()

    asyncio.run(scenario())
    thread_ids = {thread_id for _, thread_id in events}
    assert len(thread_ids) == 1
    assert main_thread not in thread_ids
    assert sum(action == "connect" for action, _ in events) == sum(action == "close" for action, _ in events)
