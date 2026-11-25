"""Confirmatory ledger: one row per session, and the combined report.

Only the pre-specified analysis writes here; exploratory output never does.
Every session that started is recorded, including sessions stopped early or
interrupted, so that stopping decisions cannot quietly remove sessions.
"""
from __future__ import annotations

import csv
import io
import math
import os

from scipy import stats as sps

from .protocol import DEFAULT_TAIL
from .stats import stouffer

FIELDS = [
    "session_id", "date", "session_type", "condition", "status", "feedback",
    "statistic_name", "statistic", "z", "p", "tail", "z_parametric",
    "secondary_name", "secondary_statistic", "secondary_z", "secondary_p",
    "lean_p1", "lean_p2", "blocks_used", "bits_collected", "bits_analyzed",
    "method", "config_hash", "analysis_version", "source",
]


def _lock(f):
    try:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    except ImportError:  # non-POSIX
        pass


def read_ledger(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_row(path: str, row: dict) -> bool:
    """Append a session's row unless that session is already in the ledger."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a+", newline="") as f:
        _lock(f)
        f.seek(0)
        content = f.read()
        existing = list(csv.DictReader(io.StringIO(content))) if content else []
        if any(r.get("session_id") == row["session_id"] for r in existing):
            return False
        header = next(csv.reader(io.StringIO(content)), None) if content else None
        if header is not None and header != FIELDS:
            # Ledger written by an older version: rewrite it with the current
            # columns, keeping every existing value (new columns left blank).
            f.seek(0)
            f.truncate()
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in existing:
                w.writerow({k: r.get(k, "") or "" for k in FIELDS})
        f.seek(0, os.SEEK_END)
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if f.tell() == 0:
            w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in FIELDS})
        f.flush()
        os.fsync(f.fileno())
    return True


def _float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _p_for(z: float, tail: str) -> float:
    if tail == "greater":
        return float(sps.norm.sf(z))
    if tail == "less":
        return float(sps.norm.cdf(z))
    return float(2 * sps.norm.sf(abs(z)))


def combine(rows: list, complete_only: bool = False) -> list:
    """Stouffer-combine z per (session type, condition)."""
    out = []
    for stype in ("intention", "connection"):
        for cond in ("real", "sham"):
            grp = [r for r in rows if r["session_type"] == stype and r["condition"] == cond]
            if complete_only:
                grp = [r for r in grp if r["status"] == "complete"]
            zs = [_float(r["z"]) for r in grp]
            Z, k = stouffer(zs)
            tail = DEFAULT_TAIL[stype]
            sec = [_float(r["secondary_z"]) for r in grp]
            Zs, ks = stouffer(sec)
            out.append({
                "session_type": stype, "condition": cond,
                "sessions": len(grp), "with_result": k,
                "incomplete": sum(1 for r in grp if r["status"] != "complete"),
                "stouffer_z": Z, "p": _p_for(Z, tail) if k else float("nan"), "tail": tail,
                "secondary_z": Zs if ks else float("nan"),
                "secondary_p": _p_for(Zs, "greater") if ks else float("nan"),
                "bits": sum(_float(r["bits_analyzed"]) or 0 for r in grp),
                "z_values": [z for z in zs if z is not None],
            })
    return out


def real_vs_sham(groups: list) -> list:
    """Difference in mean z between real and sham sessions of each type."""
    res = []
    for stype in ("intention", "connection"):
        g = {x["condition"]: x for x in groups if x["session_type"] == stype}
        r, s = g.get("real"), g.get("sham")
        if not r or not s or not r["z_values"] or not s["z_values"]:
            continue
        kr, ks = len(r["z_values"]), len(s["z_values"])
        mr = sum(r["z_values"]) / kr
        ms = sum(s["z_values"]) / ks
        zd = (mr - ms) / math.sqrt(1 / kr + 1 / ks)
        res.append({"session_type": stype, "mean_z_real": mr, "mean_z_sham": ms,
                    "z_difference": zd, "p": _p_for(zd, DEFAULT_TAIL[stype])})
    return res


def by_feedback(rows: list, complete_only: bool = False) -> list:
    """Stouffer Z per (type, condition, display), for groups that used more than one display."""
    out = []
    for stype in ("intention", "connection"):
        for cond in ("real", "sham"):
            grp = [r for r in rows if r["session_type"] == stype and r["condition"] == cond
                   and (not complete_only or r["status"] == "complete")]
            displays = sorted({r.get("feedback") or "unrecorded" for r in grp})
            if len(displays) < 2:
                continue
            for fb in displays:
                sub = [r for r in grp if (r.get("feedback") or "unrecorded") == fb]
                Z, k = stouffer([_float(r["z"]) for r in sub])
                out.append({"session_type": stype, "condition": cond, "feedback": fb, "k": k,
                            "stouffer_z": Z, "p": _p_for(Z, DEFAULT_TAIL[stype]) if k else float("nan")})
    return out


def format_report(path: str, complete_only: bool = False) -> str:
    rows = read_ledger(path)
    if not rows:
        return f"No sessions in {path}."
    groups = combine(rows, complete_only)
    lines = [f"Confirmatory report: {path}",
             "Each session contributes one z (normal equivalent of its permutation p).",
             "Sessions are combined with Stouffer's method: Z = sum(z) / sqrt(k).",
             ""]
    hdr = f"{'type':<11} {'cond':<5} {'k':>4} {'incompl':>7} {'Stouffer Z':>11} {'p':>9} {'tail':<10} {'NetVar Z':>9} {'bits':>12}"
    lines += [hdr, "-" * len(hdr)]
    for g in groups:
        if not g["sessions"]:
            continue
        nv = "" if math.isnan(g["secondary_z"]) else f"{g['secondary_z']:9.3f}"
        z = "" if not g["with_result"] else f"{g['stouffer_z']:11.3f}"
        p = "" if not g["with_result"] else f"{g['p']:9.4f}"
        lines.append(f"{g['session_type']:<11} {g['condition']:<5} {g['with_result']:>4} "
                     f"{g['incomplete']:>7} {z:>11} {p:>9} {g['tail']:<10} {nv:>9} {g['bits']:>12.3g}")
    fb = by_feedback(rows, complete_only)
    if fb:
        lines += ["", "By display (only where more than one display was used):"]
        for g in fb:
            z = f"{g['stouffer_z']:+.3f}" if g["k"] else "   n/a"
            p = f"{g['p']:.4f}" if g["k"] else "n/a"
            lines.append(f"  {g['session_type']:<11} {g['condition']:<5} {g['feedback']:<10} k {g['k']:>3}  Z {z}  p {p}")
    diffs = real_vs_sham(groups)
    if diffs:
        lines += ["", "Real vs sham (difference in mean session z):"]
        for d in diffs:
            lines.append(f"  {d['session_type']:<11} real {d['mean_z_real']:+.3f}  sham {d['mean_z_sham']:+.3f}  "
                         f"z_diff {d['z_difference']:+.3f}  p {d['p']:.4f} ({DEFAULT_TAIL[d['session_type']]})")
    lines += ["", "Sham sessions run the same schedule with no cues; their combined Z should look like chance.",
              "incompl = sessions stopped early or interrupted (included unless --complete-only)."]
    return "\n".join(lines)
