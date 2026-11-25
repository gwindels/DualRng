"""Session analysis and finalization.

finalize_session() is idempotent and works from the files on disk only, so a
session whose program crashed, was killed, or lost power is analysed the next
time the app starts (finalize_pending) with whatever complete blocks it has.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

from .ledger import append_row
from .protocol import ProtocolConfig, read_schedule
from .records import load_records, pair_records
from .stats import block_sums, run_primary


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def assign_blocks(t_start: np.ndarray, t_end: np.ndarray, plan: list,
                  stop_time: float | None = None) -> tuple[np.ndarray, list]:
    """Block index for each read (-1 if the read is not wholly inside a block).

    A block is used only if it ran to completion before the session stopped.
    Returns (block_of_read, used) where used[i] says whether block i is usable.
    """
    block_of_read = np.full(len(t_start), -1, dtype=np.int64)
    used = []
    for b in plan:
        ok = stop_time is None or b["end"] <= stop_time
        used.append(ok)
        if not ok:
            continue
        m = (t_start >= b["start"]) & (t_end <= b["end"])
        block_of_read[m] = b["index"]
    return block_of_read, used


def analyze_arrays(config: ProtocolConfig, reads: dict, plan: list,
                   stop_time: float | None = None, seed: int | None = None) -> dict:
    """Pre-specified analysis of one session from in-memory arrays."""
    block_of_read, used = assign_blocks(reads["t_start"], reads["t_end"], plan, stop_time)
    labels = [b["label"] for b in plan]
    bs = block_sums(block_of_read, reads["n"], reads["ones1"], reads["ones2"], labels)
    for i, ok in enumerate(used):
        if not ok:
            bs.count[i] = 0
    dur = reads["t_end"] - reads["t_start"]
    read_seconds = float(np.median(dur)) if len(dur) else float("nan")
    expected = config.block_seconds / read_seconds if read_seconds > 0 else float("nan")
    blocks = [{"index": b["index"], "label": b["label"], "reads": int(bs.count[b["index"]]),
               "coverage": float(bs.count[b["index"]] / expected) if expected == expected else None,
               "p1": float(bs.O1[b["index"]] / bs.bits[b["index"]]) if bs.bits[b["index"]] else None,
               "p2": float(bs.O2[b["index"]] / bs.bits[b["index"]]) if bs.bits[b["index"]] else None,
               "used": bool(used[b["index"]] and bs.count[b["index"]] > 0)}
              for b in plan]
    result = {
        "session_type": config.session_type,
        "primary_statistic": config.primary_statistic,
        "bits_collected": float(2 * reads["n"].sum()),
        "reads": int(len(reads["n"])),
        "median_read_seconds": read_seconds,
        "blocks": blocks,
    }
    try:
        result["primary"] = run_primary(config.session_type, bs, config.tail,
                                        config.exact_limit, config.n_permutations, seed)
    except ValueError as e:
        result["primary"] = None
        result["error"] = str(e)
    return result


def load_session_arrays(session_dir: str) -> tuple[dict, list]:
    recs = load_records(os.path.join(session_dir, "reads.bin"))
    return pair_records(recs), read_schedule(session_dir)


def _status(meta: dict) -> str:
    st = meta.get("status", "running")
    return "interrupted" if st == "running" else st


def ledger_row(session_id: str, meta: dict, config: ProtocolConfig, result: dict, status: str) -> dict:
    p = result.get("primary") or {}
    sec = p.get("secondary_netvar") or {}
    lean = p.get("lean") or {}
    return {
        "session_id": session_id,
        "date": meta.get("started_wall", ""),
        "session_type": config.session_type,
        "condition": "sham" if config.sham else "real",
        "status": status if p else f"{status}-no-result",
        "statistic_name": config.primary_statistic,
        "statistic": p.get("statistic"),
        "z": p.get("z"),
        "p": p.get("p"),
        "tail": config.tail,
        "z_parametric": p.get("z_parametric"),
        "secondary_name": "netvar_connect_minus_rest" if sec else "",
        "secondary_statistic": sec.get("statistic"),
        "secondary_z": sec.get("z"),
        "secondary_p": sec.get("p"),
        "lean_p1": lean.get("p1"),
        "lean_p2": lean.get("p2"),
        "blocks_used": sum(1 for b in result.get("blocks", []) if b["used"]),
        "bits_collected": result.get("bits_collected"),
        "bits_analyzed": p.get("bits_analyzed"),
        "method": p.get("method"),
        "config_hash": meta.get("config_hash"),
        "analysis_version": config.analysis_version,
        "source": (meta.get("source") or {}).get("kind", ""),
    }


def exploratory(config: ProtocolConfig, reads: dict, plan: list, read_seconds: float) -> dict:
    from .scan import run_scan

    x1 = reads["ones1"] / reads["n"]
    x2 = reads["ones2"] / reads["n"]
    if config.session_type == "connection":
        return run_scan("connection", x1, x2, config.scan_windows_seconds, read_seconds,
                        n_shifts=config.scan_shifts, t=reads["t_start"])
    aim = np.zeros(len(x1))
    for b in plan:
        m = (reads["t_start"] >= b["start"]) & (reads["t_end"] <= b["end"])
        aim[m] = 1.0 if b["label"] == "HIGH" else (-1.0 if b["label"] == "LOW" else 0.0)
    return run_scan("intention", (x1 + x2) / 2, aim, config.scan_windows_seconds, read_seconds,
                    n_shifts=config.scan_shifts, t=reads["t_start"])


def finalize_session(session_dir: str, ledger_path: str | None, run_exploratory: bool = True,
                     make_plots: bool = True, force: bool = False) -> dict:
    """Analyse a session from disk, write results.json, and append it to the ledger once."""
    results_path = os.path.join(session_dir, "results.json")
    if os.path.exists(results_path) and not force:
        return read_json(results_path)
    meta = read_json(os.path.join(session_dir, "meta.json"))
    config = ProtocolConfig.from_dict(meta["config"])
    session_id = meta["session_id"]
    status = _status(meta)
    stop_time = meta.get("stopped_mono")
    try:
        reads, plan = load_session_arrays(session_dir)
    except (FileNotFoundError, ValueError) as e:
        reads, plan = None, []
        result = {"primary": None, "error": f"no usable data: {e}", "blocks": [], "bits_collected": 0}
    if reads is not None:
        if stop_time is None and status != "complete" and len(reads["t_end"]):
            stop_time = float(reads["t_end"].max())
        result = analyze_arrays(config, reads, plan, stop_time)
    result.update({"session_id": session_id, "status": status, "config_hash": meta.get("config_hash"),
                   "condition": "sham" if config.sham else "real",
                   "analyzed_wall": datetime.now().isoformat(timespec="seconds")})
    write_json(results_path, result)
    if ledger_path:
        append_row(ledger_path, ledger_row(session_id, meta, config, result, status))
    if meta.get("status") == "running":
        meta["status"] = status
        write_json(os.path.join(session_dir, "meta.json"), meta)
    if run_exploratory and reads is not None and len(reads["n"]) > 10:
        try:
            ex = exploratory(config, reads, plan, result["median_read_seconds"])
            write_json(os.path.join(session_dir, "exploratory.json"), ex)
        except Exception as e:  # exploratory output must never block finalization
            write_json(os.path.join(session_dir, "exploratory.json"), {"error": str(e)})
    if make_plots and reads is not None:
        try:
            from .plots import plot_session
            plot_session(session_dir, config, result, reads, plan)
        except Exception as e:
            print(f"Plotting failed: {e}")
    return result


def find_sessions(base_dir: str) -> list:
    if not os.path.isdir(base_dir):
        return []
    return sorted(os.path.join(base_dir, d) for d in os.listdir(base_dir)
                  if os.path.exists(os.path.join(base_dir, d, "meta.json")))


def finalize_pending(base_dir: str, ledger_path: str, **kw) -> list:
    """Finalize every session that has no results yet (e.g. after a crash)."""
    done = []
    for d in find_sessions(base_dir):
        if not os.path.exists(os.path.join(d, "results.json")):
            meta = read_json(os.path.join(d, "meta.json"))
            if meta.get("status") == "running" and _pid_alive(meta.get("pid")):
                continue  # still running in another process
            done.append((d, finalize_session(d, ledger_path, **kw)))
    return done


def _pid_alive(pid) -> bool:
    if not pid or pid == os.getpid():
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def summarize(result: dict) -> str:
    p = result.get("primary")
    head = f"{result.get('session_type', '?')} ({result.get('condition', '?')}, {result.get('status', '?')})"
    if not p:
        return f"{head}: no result ({result.get('error', 'insufficient data')})"
    s = (f"{head}: statistic {p['statistic']:+.6g}, z {p['z']:+.2f}, "
         f"p {p['p']:.4f} ({p['tail']}, {p['method']})")
    if "secondary_netvar" in p:
        nv = p["secondary_netvar"]
        s += f"; NetVar {nv['statistic']:+.4f}, p {nv['p']:.4f}"
    return s
