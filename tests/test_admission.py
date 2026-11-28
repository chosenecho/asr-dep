"""Admission invariants tested without pytest-asyncio."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.admission import AdmissionController


def controller(**overrides):
    values = dict(max_concurrent_jobs=1, max_queued_jobs=1, max_temp_storage_mb=1)
    values.update(overrides)
    return AdmissionController(SimpleNamespace(**values))


def test_initial_queued_capacity_activation_and_repeated_release():
    async def run():
        admission = controller()
        first, second = admission.reserve(), admission.reserve()
        assert admission.stats()["waiting_jobs"] == 2
        with pytest.raises(HTTPException) as error:
            admission.reserve()
        assert error.value.status_code == 503
        assert error.value.headers["Retry-After"] == "1"
        await first.activate()
        pending = asyncio.create_task(second.activate())
        await asyncio.sleep(0)
        assert not pending.done()
        assert admission.active_jobs == admission.waiting_jobs == 1
        first.release()
        first.release()
        await asyncio.wait_for(pending, 1)
        assert admission.active_jobs == 1 and admission.waiting_jobs == 0
        second.release()
        assert admission.stats() == dict(active_jobs=0, waiting_jobs=0, reserved_temp_bytes=0)
    asyncio.run(run())


def test_cancel_waiting_and_stop_do_not_leak():
    async def run():
        admission = controller()
        first, second = admission.reserve(), admission.reserve()
        await first.activate()
        second.reserve_bytes(100)
        waiting = asyncio.create_task(second.activate())
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert admission.waiting_jobs == 1
        assert admission.reserved_temp_bytes == 100
        # The owner still has files to delete before it returns the ticket.
        second.release()
        assert admission.waiting_jobs == admission.reserved_temp_bytes == 0
        admission.stop()
        with pytest.raises(HTTPException):
            admission.reserve()
        first.release()
        assert admission.active_jobs == 0
    asyncio.run(run())


def test_zero_queue_still_reserves_active_capacity_and_storage_is_atomic():
    admission = controller(max_queued_jobs=0)
    ticket = admission.reserve()
    with pytest.raises(HTTPException):
        admission.reserve()
    ticket.reserve_bytes(100)
    with pytest.raises(HTTPException):
        ticket.reserve_bytes(1024 * 1024)
    assert admission.reserved_temp_bytes == 100
    ticket.release_bytes(40)
    assert ticket.reserved_bytes == 60
    ticket.release()
    ticket.release()
    assert admission.reserved_temp_bytes == admission.waiting_jobs == 0
