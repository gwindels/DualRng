"""The one clock shared by every DualRng process.

Block times are written by the main program and read timestamps by the
recorder process, so both must read the same clock. time.monotonic() is not
guaranteed to: on some macOS Python builds each process's time.monotonic()
has its own offset (seen as seconds of disagreement between a parent and a
spawned child). clock_gettime(CLOCK_MONOTONIC) is the system-wide monotonic
clock on Linux and macOS. Recorder start-up checks that the two processes
agree (see recorder.Recorder.start).
"""
from __future__ import annotations

import time

if hasattr(time, "clock_gettime") and hasattr(time, "CLOCK_MONOTONIC"):
    def now() -> float:
        return time.clock_gettime(time.CLOCK_MONOTONIC)
else:  # pragma: no cover - platforms without clock_gettime (Windows)
    now = time.monotonic
