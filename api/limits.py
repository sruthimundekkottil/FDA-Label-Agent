"""In-memory request limits: per-client sliding window and a global daily cap.

Both live in the process, so they reset on restart and are per worker. That is
deliberate for a single-container demo (run uvicorn with one worker); a shared
store such as Redis would be needed to enforce them across workers.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable


class SlidingWindowLimiter:
    """At most ``limit`` hits per ``window_s`` seconds for each key (client IP)."""

    def __init__(self, limit: int, window_s: float = 60.0, *, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Record a hit. Returns ``None`` if allowed, else seconds until the next free slot."""

        if self.limit <= 0:  # 0 disables the limit
            return None
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(0.0, self.window_s - (now - hits[0]))
            hits.append(now)
            if len(self._hits) > 10_000:  # forget idle clients so memory stays bounded
                for idle in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window_s]:
                    del self._hits[idle]
            return None


class DailyCap:
    """At most ``cap`` calls per UTC day across all clients."""

    def __init__(self, cap: int, *, today: Callable[[], str] | None = None):
        self.cap = cap
        self._today = today or (lambda: datetime.now(timezone.utc).date().isoformat())
        self._day = self._today()
        self._count = 0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        day = self._today()
        if day != self._day:
            self._day, self._count = day, 0

    def try_take(self) -> bool:
        if self.cap <= 0:  # 0 disables the cap
            return True
        with self._lock:
            self._roll()
            if self._count >= self.cap:
                return False
            self._count += 1
            return True

    def used(self) -> int:
        with self._lock:
            self._roll()
            return self._count

    @staticmethod
    def seconds_until_reset() -> int:
        now = datetime.now(timezone.utc)
        midnight = datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp() + 86_400
        return math.ceil(midnight - now.timestamp())
