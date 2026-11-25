"""Session protocol: configuration, pre-specified statistic, and block schedule.

Everything that determines how a session is analysed lives in ProtocolConfig
and is frozen (with a hash) in meta.json before the first block starts.
The block order is drawn from the operating system's CSPRNG (secrets), never
from the generators under test.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass, field

from . import ANALYSIS_VERSION

SESSION_TYPES = ("intention", "connection")

LABELS = {
    "intention": ("HIGH", "LOW", "REST"),
    "connection": ("CONNECT", "REST"),
}

# What the participant is shown for each label (real sessions only).
CUES = {
    "HIGH": "AIM HIGH",
    "LOW": "AIM LOW",
    "REST": "REST",
    "CONNECT": "CONNECT",
}

PRIMARY_STATISTIC = {
    "intention": "p1(HIGH) - p1(LOW), both generators pooled",
    "connection": "mean z1*z2 in CONNECT - mean z1*z2 in REST (z from REST blocks)",
}

SECONDARY_STATISTIC = {
    "intention": None,
    "connection": "mean NetVar in CONNECT - mean NetVar in REST (NetVar = Stouffer Z^2)",
}

# Pre-specified alternative for the primary statistic.
DEFAULT_TAIL = {
    "intention": "greater",     # HIGH blocks have more ones than LOW blocks
    "connection": "two-sided",  # coherence could appear as positive or negative coupling
}

DEFAULT_BLOCKS_PER_LABEL = {"intention": 8, "connection": 12}

# What the participant sees during blocks. Part of the protocol (and its hash),
# so sessions run with different displays are never pooled unknowingly.
#   trace  cumulative deviation line for the current block, with the aimed
#          direction and a chance envelope (intention only)
#   wave   the sine-wave visual, calmer and brighter as the block goes the
#          intended way
#   text   headless runs: cues and block z printed in a terminal
FEEDBACK_CHOICES = {
    "intention": ("trace", "wave", "text"),
    "connection": ("wave", "text"),
}
DEFAULT_FEEDBACK = {"intention": "trace", "connection": "wave"}


@dataclass
class ProtocolConfig:
    session_type: str = "intention"
    sham: bool = False
    block_seconds: float = 30.0
    gap_seconds: float = 3.0
    blocks_per_label: int = 0          # 0 = default for the session type
    bits_per_read: int = 1024
    tail: str = ""                     # "" = default for the session type
    feedback: str = ""                 # "" = default display for the session type
    n_permutations: int = 20000        # used when exact enumeration is too large
    exact_limit: int = 200000          # enumerate all label assignments up to this many
    scan_windows_seconds: tuple = (5, 10, 20, 40, 80, 160)
    scan_shifts: int = 1000
    analysis_version: str = ANALYSIS_VERSION

    def __post_init__(self):
        if self.session_type not in SESSION_TYPES:
            raise ValueError(f"session_type must be one of {SESSION_TYPES}")
        if not self.blocks_per_label:
            self.blocks_per_label = DEFAULT_BLOCKS_PER_LABEL[self.session_type]
        if not self.tail:
            self.tail = DEFAULT_TAIL[self.session_type]
        if not self.feedback:
            self.feedback = DEFAULT_FEEDBACK[self.session_type]
        if self.feedback not in FEEDBACK_CHOICES[self.session_type]:
            raise ValueError(f"feedback for {self.session_type} must be one of "
                             f"{FEEDBACK_CHOICES[self.session_type]}")
        if self.tail not in ("greater", "less", "two-sided"):
            raise ValueError("tail must be greater, less or two-sided")
        if self.blocks_per_label < 2:
            raise ValueError("blocks_per_label must be at least 2")
        if self.block_seconds <= 0 or self.gap_seconds < 0:
            raise ValueError("block_seconds must be > 0 and gap_seconds >= 0")
        self.scan_windows_seconds = tuple(self.scan_windows_seconds)

    @property
    def labels(self) -> tuple:
        return LABELS[self.session_type]

    @property
    def n_blocks(self) -> int:
        return self.blocks_per_label * len(self.labels)

    @property
    def duration_seconds(self) -> float:
        return self.n_blocks * (self.block_seconds + self.gap_seconds)

    @property
    def primary_statistic(self) -> str:
        return PRIMARY_STATISTIC[self.session_type]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scan_windows_seconds"] = list(self.scan_windows_seconds)
        d["primary_statistic"] = self.primary_statistic
        d["secondary_statistic"] = SECONDARY_STATISTIC[self.session_type]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProtocolConfig":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})

    def config_hash(self) -> str:
        """Hash of everything that defines the analysis (not the session's identity)."""
        canon = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()[:16]


def make_schedule(config: ProtocolConfig) -> list:
    """Balanced block labels in an order drawn from the OS CSPRNG."""
    labels = [lab for lab in config.labels for _ in range(config.blocks_per_label)]
    secrets.SystemRandom().shuffle(labels)
    return labels


def plan_blocks(labels: list, t0: float, block_seconds: float, gap_seconds: float) -> list:
    """Absolute monotonic times for each block. Each block is preceded by a gap."""
    plan = []
    period = block_seconds + gap_seconds
    for i, lab in enumerate(labels):
        start = t0 + i * period + gap_seconds
        plan.append({"index": i, "label": lab, "start": start, "end": start + block_seconds})
    return plan


def write_schedule(session_dir: str, plan: list, wall_t0: str, mono_t0: float) -> str:
    """Write the whole plan before the first block starts, and fsync it."""
    path = os.path.join(session_dir, "schedule.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "label", "start_mono", "end_mono", "start_offset_s"])
        for b in plan:
            w.writerow([b["index"], b["label"], f"{b['start']:.6f}", f"{b['end']:.6f}",
                        f"{b['start'] - mono_t0:.3f}"])
        f.flush()
        os.fsync(f.fileno())
    return path


def read_schedule(session_dir: str) -> list:
    path = os.path.join(session_dir, "schedule.csv")
    with open(path, newline="") as f:
        return [{"index": int(r["index"]), "label": r["label"],
                 "start": float(r["start_mono"]), "end": float(r["end_mono"])}
                for r in csv.DictReader(f)]
