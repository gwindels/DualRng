"""Per-session chart: block-level view of the pre-specified measure, plus the
exploratory cumulative trace (labelled as such)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABEL_COLORS = {
    "HIGH": "#e8851a",
    "LOW": "#5f81eb",
    "CONNECT": "#aa5feb",
    "REST": "#8a8a8a",
}


def _block_measure(config, reads, plan, result):
    n = reads["n"].astype(float)
    x1 = reads["ones1"] / n
    x2 = reads["ones2"] / n
    rest = np.zeros(len(n), bool)
    in_block = np.full(len(n), -1)
    for b in plan:
        m = (reads["t_start"] >= b["start"]) & (reads["t_end"] <= b["end"])
        in_block[m] = b["index"]
        if b["label"] == "REST":
            rest |= m
    if not rest.any():
        rest[:] = True
    vals = []
    if config.session_type == "intention":
        p0 = (x1[rest].mean() + x2[rest].mean()) / 2
        ylabel = "p(1) - REST lean (pooled)"
        for b in plan:
            m = in_block == b["index"]
            vals.append(((x1[m] + x2[m]) / 2).mean() - p0 if m.any() else np.nan)
        trace = np.cumsum(np.where(in_block >= 0, (x1 + x2) / 2 - p0, 0.0) *
                          np.array([{"HIGH": 1, "LOW": -1}.get(plan[i]["label"], 0) if i >= 0 else 0
                                    for i in in_block]))
        tlabel = "cumulative aim-signed deviation (HIGH +, LOW -)"
    else:
        z1 = (x1 - x1[rest].mean()) / x1[rest].std(ddof=1)
        z2 = (x2 - x2[rest].mean()) / x2[rest].std(ddof=1)
        ylabel = "mean z1*z2"
        for b in plan:
            m = in_block == b["index"]
            vals.append((z1[m] * z2[m]).mean() if m.any() else np.nan)
        trace = np.cumsum(z1 * z2)
        tlabel = "cumulative sum of z1*z2 (all reads)"
    return np.array(vals), ylabel, trace, tlabel


def plot_session(session_dir, config, result, reads, plan):
    if not len(reads["n"]):
        return None
    vals, ylabel, trace, tlabel = _block_measure(config, reads, plan, result)
    t0 = plan[0]["start"] - config.gap_seconds if plan else reads["t_start"][0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    for b, v in zip(plan, vals):
        ax1.bar((b["start"] - t0) / 60, v, width=config.block_seconds / 60, align="edge",
                color=LABEL_COLORS.get(b["label"], "#444444"), alpha=0.85)
    for lab in config.labels:
        ax1.bar(0, 0, color=LABEL_COLORS.get(lab), label=lab)
    ax1.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel("minutes")
    p = result.get("primary")
    cond = "sham" if config.sham else "real"
    if p:
        title = (f"{config.session_type.capitalize()} ({cond}): {config.primary_statistic}\n"
                 f"statistic {p['statistic']:+.4g}   z {p['z']:+.2f}   p {p['p']:.4f} ({p['tail']}, permutation)")
    else:
        title = f"{config.session_type.capitalize()} ({cond}): no result ({result.get('error', '')})"
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize="small")

    t = (reads["t_start"] - t0) / 60
    ax2.plot(t, trace, color="#8a8a8a")
    ax2.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("minutes")
    ax2.set_ylabel(tlabel)
    ax2.set_title("EXPLORATORY view (not part of the pre-specified test)")
    plt.tight_layout()
    path = os.path.join(session_dir, "session.png")
    plt.savefig(path)
    plt.close(fig)
    return path
