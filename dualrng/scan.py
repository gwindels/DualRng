"""EXPLORATORY multi-scale scan. Never written to the ledger.

Statistic: the largest standardized windowed sum of the session's measure over
all window lengths and positions (full windows only):

    M = max over w, i of |sum(u[i:i+w])| / (sqrt(w) * sd(u))

The p-value comes from a null built by circularly shifting one stream against
the other, which keeps each stream's own structure (lean, drift,
autocorrelation) and breaks only their alignment:

* connection: u = z1 * z2; generator 2 is shifted against generator 1.
* intention:  u = aim * z, with aim = +1 (HIGH), -1 (LOW), 0 otherwise and z the
  pooled per-read deviation; the aim sequence is shifted against the data.
"""
from __future__ import annotations

import csv
import glob
import math
import os
from datetime import datetime

import numpy as np

LABEL = "EXPLORATORY - not a pre-specified test; not part of the ledger"


def _standardize(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def _scan(u: np.ndarray, windows: list) -> tuple[float, int, int]:
    """Return (M, best window, start index of best window)."""
    sd = u.std()
    if sd == 0:
        return 0.0, windows[0], 0
    c = np.concatenate([[0.0], np.cumsum(u)])
    best = (-1.0, windows[0], 0)
    for w in windows:
        s = np.abs(c[w:] - c[:-w]) / (math.sqrt(w) * sd)
        i = int(np.argmax(s))
        if s[i] > best[0]:
            best = (float(s[i]), w, i)
    return best


def _windows_in_reads(window_seconds, read_seconds: float, n: int) -> list:
    ws = sorted({max(2, int(round(s / read_seconds))) for s in window_seconds})
    return [w for w in ws if w <= n // 2]


def run_scan(kind: str, a: np.ndarray, b: np.ndarray, window_seconds, read_seconds: float,
             n_shifts: int = 1000, seed: int | None = None, t: np.ndarray | None = None) -> dict:
    """kind 'connection': a, b = per-read proportions of generator 1 and 2.
    kind 'intention': a = pooled per-read proportion, b = aim (+1/-1/0)."""
    n = len(a)
    windows = _windows_in_reads(window_seconds, read_seconds, n)
    if not windows:
        return {"label": LABEL, "error": "session too short for the scan windows"}
    rng = np.random.default_rng(seed)
    if kind == "connection":
        z1, z2 = _standardize(a), _standardize(b)
        obs = _scan(z1 * z2, windows)
    elif kind == "intention":
        z = _standardize(a)
        aim = b.astype(float)
        obs = _scan(aim * z, windows)
    else:
        raise ValueError(kind)
    min_shift = max(windows)
    if n - 2 * min_shift < 1:
        min_shift = max(1, n // 4)
    null = np.empty(n_shifts)
    for j in range(n_shifts):
        k = int(rng.integers(min_shift, n - min_shift + 1))
        if kind == "connection":
            null[j] = _scan(z1 * np.roll(z2, k), windows)[0]
        else:
            null[j] = _scan(np.roll(aim, k) * z, windows)[0]
    M, w, i = obs
    out = {
        "label": LABEL, "kind": kind, "M": M,
        "p_scan": float((1 + np.sum(null >= M)) / (1 + n_shifts)),
        "best_window_reads": int(w), "best_window_seconds": float(w * read_seconds),
        "best_start_index": int(i), "windows_reads": windows,
        "window_seconds_nominal": [float(x * read_seconds) for x in windows],
        "n_shifts": n_shifts, "null_median": float(np.median(null)),
        "null_95th": float(np.percentile(null, 95)), "n_reads": n,
    }
    if t is not None and len(t) == n:
        out["best_start_time"] = float(t[i])
    return out


# --------------------------------------------------------------------------
# Legacy (pre-2.0) session CSVs in data/
# --------------------------------------------------------------------------

LEGACY_BITS = 8192  # each legacy z was computed from 1024 bytes = 8192 LSBs at p = 0.5


def load_legacy_session(path: str) -> dict:
    ts, z1, z2 = [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            ts.append(datetime.fromisoformat(r["timestamps"]).timestamp())
            z1.append(float(r["z_scores_1"]))
            z2.append(float(r["z_scores_2"]))
    n = LEGACY_BITS
    half_sd = math.sqrt(n) / 2
    ones1 = np.rint(n / 2 + np.array(z1) * half_sd)
    ones2 = np.rint(n / 2 + np.array(z2) * half_sd)
    t = np.array(ts)
    return {"t": t, "x1": ones1 / n, "x2": ones2 / n, "n": n}


def legacy_scan(data_dir: str, window_seconds=(5, 10, 20, 40, 80, 160), n_shifts: int = 1000,
                seed: int = 0) -> list:
    out = []
    for path in sorted(glob.glob(os.path.join(data_dir, "session_*.csv"))):
        s = load_legacy_session(path)
        t = s["t"]
        dt = np.diff(t)
        read_seconds = float(np.median(dt))
        wall = float(t[-1] - t[0])
        res = run_scan("connection", s["x1"], s["x2"], window_seconds, read_seconds,
                       n_shifts=n_shifts, seed=seed, t=t)
        res.update({
            "file": os.path.basename(path), "reads": len(t), "wall_seconds": wall,
            "sampled_fraction": float(np.sum(np.minimum(dt, 1.5 * read_seconds)) / wall) if wall else float("nan"),
            "lean_p1": float(s["x1"].mean()), "lean_p2": float(s["x2"].mean()),
            "corr_z1_z2": float(np.corrcoef(s["x1"], s["x2"])[0, 1]),
        })
        out.append(res)
    return out


def format_legacy(results: list) -> str:
    lines = [LABEL, "Legacy sessions re-run through the connection scan (z1*z2, circular-shift null).",
             "Windows are counted in reads; legacy reads are ~0.57 s apart with long unsampled gaps,",
             "so a window spans more wall time than its nominal length. no-stall = share of wall",
             "time not inside a stall (an interval longer than 1.5x the median read interval).", ""]
    hdr = f"{'file':<46} {'reads':>5} {'no-stall':>8} {'p1':>8} {'p2':>8} {'r':>7} {'M':>6} {'win':>5} {'p_scan':>7}"
    lines += [hdr, "-" * len(hdr)]
    for r in results:
        lines.append(f"{r['file']:<46} {r['reads']:>5} {r['sampled_fraction']:>8.0%} {r['lean_p1']:>8.5f} "
                     f"{r['lean_p2']:>8.5f} {r['corr_z1_z2']:>7.3f} {r['M']:>6.2f} {r['best_window_reads']:>5} "
                     f"{r['p_scan']:>7.3f}")
    ps = [r["p_scan"] for r in results]
    lines += ["", f"{sum(p < 0.05 for p in ps)} of {len(ps)} sessions have p_scan < 0.05 "
              f"(about {0.05 * len(ps):.1f} expected by chance)."]
    return "\n".join(lines)
