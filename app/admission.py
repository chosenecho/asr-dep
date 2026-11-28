"""Single-event-loop admission and temporary-storage ownership."""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from fastapi import HTTPException


def _busy(detail: str) -> HTTPException:
    return HTTPException(503, detail=detail, headers={"Retry-After": "1"})


class AdmissionController:
    """Reserve synchronously before any await; tickets live until readers drain."""

    def __init__(self, settings: Any):
        self.max_active = settings.max_concurrent_jobs
        self.capacity = self.max_active + settings.max_queued_jobs
        self.max_bytes = int(settings.max_temp_storage_mb * 1024 * 1024)
        self.active_jobs = 0
        self.waiting_jobs = 0
        self.reserved_temp_bytes = 0
        self.stopped = False
        self._waiters: deque[AdmissionTicket] = deque()

    def reserve(self) -> AdmissionTicket:
        if self.stopped or self.active_jobs + self.waiting_jobs >= self.capacity:
            raise _busy("Service is stopping or job capacity is exhausted.")
        self.waiting_jobs += 1
        return AdmissionTicket(self)

    def stop(self) -> None:
        """Stop new reservations; existing tickets may finish."""
        self.stopped = True

    def stats(self) -> dict[str, int]:
        return {"active_jobs": self.active_jobs, "waiting_jobs": self.waiting_jobs,
                "reserved_temp_bytes": self.reserved_temp_bytes}

    def _promote(self) -> None:
        while self._waiters and self.active_jobs < self.max_active:
            ticket = self._waiters.popleft()
            if ticket.released:
                continue
            ticket.active = True
            self.waiting_jobs -= 1
            self.active_jobs += 1
            ticket._ready.set()


class AdmissionTicket:
    def __init__(self, controller: AdmissionController):
        self._controller = controller
        self.active = False
        self.released = False
        self.reserved_bytes = 0
        self._ready = asyncio.Event()
        self._activating = False

    async def activate(self) -> None:
        if self.released:
            raise RuntimeError("Ticket has already been released.")
        if self.active:
            return
        if self._activating:
            raise RuntimeError("Ticket activation is already pending.")
        self._activating = True
        self._controller._waiters.append(self)
        self._controller._promote()
        try:
            await self._ready.wait()
            if self.released:
                raise RuntimeError("Ticket was released while waiting.")
        except BaseException:
            # The job still owns staged files. Its coordinator releases the
            # ticket only after file/reader cleanup, even when activation stops.
            if self in self._controller._waiters:
                self._controller._waiters.remove(self)
            raise
        finally:
            self._activating = False

    def reserve_bytes(self, delta: int) -> None:
        if delta < 0:
            raise ValueError("Reservation must be nonnegative.")
        if self.released:
            raise RuntimeError("Ticket has already been released.")
        owner = self._controller
        if owner.reserved_temp_bytes + delta > owner.max_bytes:
            raise _busy("Temporary storage capacity is exhausted.")
        owner.reserved_temp_bytes += delta
        self.reserved_bytes += delta

    def release_bytes(self, delta: int) -> None:
        if delta < 0:
            raise ValueError("Release must be nonnegative.")
        amount = min(delta, self.reserved_bytes)
        self.reserved_bytes -= amount
        self._controller.reserved_temp_bytes -= amount

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.release_bytes(self.reserved_bytes)
        owner = self._controller
        if self.active:
            owner.active_jobs -= 1
            self.active = False
        else:
            owner.waiting_jobs -= 1
        if self in owner._waiters:
            owner._waiters.remove(self)
        self._ready.set()
        owner._promote()
