"""Recorder process: reads the source continuously and appends to reads.bin.

It runs in its own process (spawned, not forked, so it never inherits pygame or
matplotlib state). Nothing the GUI or analysis does can block it; they only
read the file it writes.
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import os
import time
import traceback

from .records import ReadsWriter


def _log(path: str, event: str, detail: str = "") -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["t_mono", "wall", "event", "detail"])
        w.writerow([f"{time.monotonic():.6f}", time.strftime("%Y-%m-%d %H:%M:%S"), event, detail])
        f.flush()
        os.fsync(f.fileno())


def recorder_main(session_dir: str, source_spec: dict, bits_per_read: int,
                  stop_event, ready_event, error_event, parent_pid: int | None = None) -> None:
    from .sources import build_source

    log_path = os.path.join(session_dir, "acq_log.csv")
    writer = None
    source = None
    try:
        source = build_source(source_spec, bits_per_read, session_dir=session_dir)
        source.open()
        writer = ReadsWriter(os.path.join(session_dir, "reads.bin"), bits_per_read)
        _log(log_path, "start", repr(source.describe()))
        ready_event.set()
        seq = 0
        last_stats = last_parent_check = time.monotonic()
        while not stop_event.is_set():
            t_start, t_end, b1, b2 = source.read_pair()
            writer.write_pair(seq, t_start, t_end, b1, b2)
            seq += 1
            now = time.monotonic()
            if parent_pid and now - last_parent_check > 1.0:
                last_parent_check = now
                if os.getppid() != parent_pid:
                    # The program that started this session died without stopping us.
                    # Release the device; the data so far is on disk and will be
                    # finalized the next time the app starts.
                    _log(log_path, "parent_gone", f"reads={seq}")
                    break
            if time.monotonic() - last_stats > 10:
                st = source.stats()
                if st:
                    _log(log_path, "stats", repr(st))
                last_stats = time.monotonic()
        st = source.stats()
        _log(log_path, "stop", f"reads={seq} {st!r}" if st else f"reads={seq}")
    except Exception as e:  # recorded, and surfaced to the controller
        _log(log_path, "error", f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)!r}")
        error_event.set()
        ready_event.set()
    finally:
        if writer is not None:
            writer.close()
        if source is not None:
            try:
                source.close()
            except Exception:
                pass


class Recorder:
    """Handle on a running recorder process."""

    def __init__(self, session_dir: str, source_spec: dict, bits_per_read: int):
        ctx = mp.get_context("spawn")
        self.stop_event = ctx.Event()
        self.ready_event = ctx.Event()
        self.error_event = ctx.Event()
        self.process = ctx.Process(
            target=recorder_main,
            args=(session_dir, source_spec, bits_per_read,
                  self.stop_event, self.ready_event, self.error_event, os.getpid()),
            name="dualrng-recorder",
            daemon=False,
        )

    def start(self, timeout: float = 30.0) -> None:
        self.process.start()
        if not self.ready_event.wait(timeout):
            self.stop()
            raise RuntimeError("Recorder did not start in time")
        if self.error_event.is_set():
            self.stop()
            raise RuntimeError("Recorder failed to start; see acq_log.csv")

    @property
    def failed(self) -> bool:
        return self.error_event.is_set() or (self.process.exitcode not in (None, 0))

    def alive(self) -> bool:
        return self.process.is_alive()

    def stop(self, timeout: float = 10.0) -> None:
        self.stop_event.set()
        self.process.join(timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(5)
