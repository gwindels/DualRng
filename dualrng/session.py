"""Running one session: freeze the config, start the recorder, write the
schedule, follow the clock, stop, and finalize.

All timing uses dualrng.clock.now() (the system-wide monotonic clock), which
the recorder process also uses, so block boundaries and read timestamps are on
the same clock. Recorder start-up verifies this.
"""
from __future__ import annotations

import os
import platform
import socket
import sys
import time
from datetime import datetime

from .analysis import finalize_session, write_json
from . import clock
from .protocol import CUES, ProtocolConfig, make_schedule, plan_blocks, write_schedule
from .recorder import Recorder


class SessionRunner:
    def __init__(self, config: ProtocolConfig, base_dir: str, source_spec: dict,
                 ledger_path: str | None, lead_seconds: float = 3.0):
        self.config = config
        self.base_dir = base_dir
        self.source_spec = dict(source_spec)
        self.ledger_path = ledger_path
        self.lead_seconds = lead_seconds
        self.session_id = None
        self.dir = None
        self.plan = []
        self.meta = {}
        self.recorder = None
        self.stopped = False

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        kind = self.config.session_type + ("_sham" if self.config.sham else "")
        self.session_id = f"{stamp}_{kind}"
        self.dir = os.path.join(self.base_dir, self.session_id)
        os.makedirs(self.dir, exist_ok=False)
        self.meta = {
            "session_id": self.session_id,
            "status": "running",
            "started_wall": datetime.now().isoformat(timespec="seconds"),
            "config": self.config.to_dict(),
            "config_hash": self.config.config_hash(),
            "source": self.source_spec,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        self._write_meta()  # config frozen before any data exists
        self.recorder = Recorder(self.dir, self.source_spec, self.config.bits_per_read)
        self.recorder.start()
        labels = make_schedule(self.config)
        t0 = clock.now() + self.lead_seconds
        self.plan = plan_blocks(labels, t0, self.config.block_seconds, self.config.gap_seconds)
        write_schedule(self.dir, self.plan, self.meta["started_wall"], t0)
        self.meta.update({"t0_mono": t0, "wall_minus_mono": time.time() - clock.now(),
                          "recorder_clock_offset_s": self.recorder.clock_offset,
                          "planned_end_mono": self.plan[-1]["end"]})
        self._write_meta()

    def _write_meta(self) -> None:
        write_json(os.path.join(self.dir, "meta.json"), self.meta)

    def stop(self, early: bool = False) -> None:
        if self.stopped:
            return
        self.stopped = True
        now = clock.now()
        failed = self.recorder is not None and self.recorder.failed
        if self.recorder is not None:
            self.recorder.stop()
        if failed:
            self.meta["status"] = "recorder_failed"
        elif early or now < self.plan[-1]["end"]:
            self.meta["status"] = "stopped"
            self.meta["stopped_mono"] = now
        else:
            self.meta["status"] = "complete"
        self.meta["ended_wall"] = datetime.now().isoformat(timespec="seconds")
        self._write_meta()

    def finalize(self, **kw) -> dict:
        return finalize_session(self.dir, self.ledger_path, **kw)

    # -- clock ---------------------------------------------------------
    def done(self, now: float | None = None) -> bool:
        now = clock.now() if now is None else now
        return bool(self.plan) and now >= self.plan[-1]["end"]

    def state(self, now: float | None = None) -> dict:
        """Where the session is right now: lead-in, gap before a block, in a block, or done."""
        now = clock.now() if now is None else now
        n = len(self.plan)
        if not n:
            return {"phase": "starting"}
        if now >= self.plan[-1]["end"]:
            return {"phase": "done", "n_blocks": n}
        for b in self.plan:
            if now < b["start"]:
                return {"phase": "gap", "index": b["index"], "label": b["label"],
                        "cue": CUES[b["label"]], "n_blocks": n, "until": b["start"] - now}
            if now < b["end"]:
                return {"phase": "block", "index": b["index"], "label": b["label"],
                        "cue": CUES[b["label"]], "n_blocks": n, "elapsed": now - b["start"],
                        "remaining": b["end"] - now}
        return {"phase": "done", "n_blocks": n}

    def recorder_ok(self) -> bool:
        return self.recorder is not None and self.recorder.alive() and not self.recorder.failed
