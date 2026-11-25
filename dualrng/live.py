"""Live feedback from a running session's reads file.

Reads only; the recorder never waits on it. The value shown is the session's
pre-specified measure computed on the current block:

* intention:  aim-signed z of the block's pooled p(1) against the lean
  estimated from REST blocks so far (before any REST data: all other reads so
  far, never the block being shown). block_trace() gives the same measure
  read by read, as a cumulative deviation for the trace display.
* connection: sum of z1*z2 over the block / sqrt(reads), with z standardized
  from REST blocks so far (all reads so far before enough REST data).

This is display only. The confirmatory result comes from the permutation test
after the session.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .records import ReadsTailer

FIELDS = ("count", "bits", "O1", "O2", "S1", "S2", "Q1", "Q2", "C12")


class LiveFeedback:
    def __init__(self, session_dir: str, session_type: str, plan: list):
        self.session_type = session_type
        self.plan = plan
        self.starts = np.array([b["start"] for b in plan])
        self.ends = np.array([b["end"] for b in plan])
        self.labels = [b["label"] for b in plan]
        self.tailer = ReadsTailer(os.path.join(session_dir, "reads.bin"))
        nb = len(plan)
        self.blk = {f: np.zeros(nb) for f in FIELDS}
        self.all = {f: 0.0 for f in FIELDS}
        self._pending = {}
        self._block_reads = {}   # block index -> list of (pooled ones, pooled bits)
        self.last_read_mono = None
        self.reads = 0

    def poll(self) -> int:
        recs = self.tailer.poll()
        if recs is None:
            return 0
        added = 0
        for r in recs:
            seq = int(r["seq"])
            other = self._pending.pop(seq, None)
            if other is None:
                self._pending[seq] = r
                continue
            g1, g2 = (r, other) if r["gen"] == 1 else (other, r)
            self._add(float(min(g1["t_start"], g2["t_start"])), float(max(g1["t_end"], g2["t_end"])),
                      int(g1["n"]), int(g1["ones"]), int(g2["ones"]))
            added += 1
        return added

    def _add(self, t0: float, t1: float, n: int, o1: int, o2: int) -> None:
        x1, x2 = o1 / n, o2 / n
        vals = {"count": 1.0, "bits": n, "O1": o1, "O2": o2, "S1": x1, "S2": x2,
                "Q1": x1 * x1, "Q2": x2 * x2, "C12": x1 * x2}
        for f, v in vals.items():
            self.all[f] += v
        i = int(np.searchsorted(self.starts, t0, side="right")) - 1
        if 0 <= i < len(self.plan) and t1 <= self.ends[i]:
            for f, v in vals.items():
                self.blk[f][i] += v
            self._block_reads.setdefault(i, []).append((o1 + o2, 2 * n))
        self.last_read_mono = t1
        self.reads += 1

    def _rest(self, exclude: int | None = None) -> dict:
        m = np.array([lab == "REST" for lab in self.labels])
        rest = {f: float(self.blk[f][m].sum()) for f in FIELDS}
        if rest["count"] >= 10:
            return rest
        # No REST data yet: everything recorded so far except the block being shown.
        if exclude is None:
            return dict(self.all)
        return {f: self.all[f] - float(self.blk[f][exclude]) for f in FIELDS}

    def pooled_lean(self, exclude: int | None = None) -> float | None:
        base = self._rest(exclude)
        if base["bits"] <= 0:
            return None
        return (base["O1"] + base["O2"]) / (2 * base["bits"])

    def block_trace(self, index: int) -> np.ndarray | None:
        """Cumulative deviation of the block, read by read, in z units:
        trace[k] = sum over the first k+1 reads of (ones - bits*p0) / sqrt(bits*p0*q0),
        with p0 the pooled lean from REST blocks. Up means more 1s than usual.
        Under chance, trace[k] has standard deviation sqrt(k+1)."""
        reads = self._block_reads.get(index)
        p0 = self.pooled_lean(exclude=index)
        if not reads or p0 is None or not 0 < p0 < 1:
            return None
        r = np.asarray(reads, dtype=float)
        return np.cumsum((r[:, 0] - r[:, 1] * p0) / np.sqrt(r[:, 1] * p0 * (1 - p0)))

    def block_z(self, index: int) -> float | None:
        """The pre-specified measure on one block, as a z-like number (aim-signed for intention)."""
        b = {f: float(self.blk[f][index]) for f in FIELDS}
        if b["count"] < 2:
            return None
        base = self._rest(exclude=index)
        if base["count"] < 2:
            return None
        if self.session_type == "intention":
            aim = {"HIGH": 1.0, "LOW": -1.0}.get(self.labels[index])
            if aim is None:
                return None
            p0 = (base["O1"] + base["O2"]) / (2 * base["bits"])
            N = 2 * b["bits"]
            return aim * ((b["O1"] + b["O2"]) - N * p0) / math.sqrt(N * p0 * (1 - p0))
        if self.labels[index] != "CONNECT":
            return None
        c = base["count"]
        m1, m2 = base["S1"] / c, base["S2"] / c
        v1 = (base["Q1"] - c * m1 * m1) / (c - 1)
        v2 = (base["Q2"] - c * m2 * m2) / (c - 1)
        if v1 <= 0 or v2 <= 0:
            return None
        prod = (b["C12"] - m2 * b["S1"] - m1 * b["S2"] + b["count"] * m1 * m2) / math.sqrt(v1 * v2)
        return prod / math.sqrt(b["count"])
