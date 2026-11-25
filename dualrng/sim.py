"""Fast count-level session simulation for validating the statistics.

Produces the same paired-read arrays and block plan the analysis reads from
disk, using the same SimModel as the bit-level SimSource, but draws ones
counts directly from the binomial distribution instead of generating bits.
"""
from __future__ import annotations

import numpy as np

from .protocol import ProtocolConfig, make_schedule, plan_blocks
from .sources import SimModel


def simulate_session(config: ProtocolConfig, model: SimModel | None = None,
                     reads_per_second: float = 50.0, rng: np.random.Generator | None = None,
                     labels: list | None = None) -> tuple[dict, list]:
    """Return (reads, plan) for one simulated session.

    reads: dict of arrays t_start, t_end, n, ones1, ones2 (one entry per read).
    plan:  list of blocks with label, start, end (same time base as reads).
    """
    model = model or SimModel()
    rng = rng or np.random.default_rng()
    labels = labels or make_schedule(config)
    plan = plan_blocks(labels, 0.0, config.block_seconds, config.gap_seconds)
    period = 1.0 / reads_per_second
    t_start = np.arange(0.0, config.duration_seconds, period)
    t_end = t_start + period
    # Label of each read (reads that straddle a boundary get no label).
    read_labels = np.full(len(t_start), "", dtype=object)
    for b in plan:
        m = (t_start >= b["start"]) & (t_end <= b["end"])
        read_labels[m] = b["label"]
    n = config.bits_per_read
    p1, p2 = model.p_for_reads(read_labels, n, rng)
    reads = {
        "t_start": t_start, "t_end": t_end,
        "n": np.full(len(t_start), n, dtype=np.int64),
        "ones1": rng.binomial(n, p1).astype(np.int64),
        "ones2": rng.binomial(n, p2).astype(np.int64),
    }
    return reads, plan
