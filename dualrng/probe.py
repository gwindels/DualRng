"""Inspect what the device actually sends in MODE_UNWHITENED and MODE_RAW_BIN.

The vendor documents MODE_RAW_BIN's packet layout but not MODE_UNWHITENED's
output, so this captures a sample of each, saves the raw bytes, and prints
what the bytes look like. Run it once on the Pi with the device attached:

    python app.py probe
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime

import numpy as np

from .sources import RawBinParser, find_port, knock


def _capture(port: str, mode: str, n_bytes: int) -> tuple[bytes, float]:
    import serial

    knock(port, mode)
    with serial.Serial(port, timeout=2) as s:
        s.setDTR(True)
        time.sleep(0.2)
        s.reset_input_buffer()
        t0 = time.monotonic()
        data = b""
        while len(data) < n_bytes:
            chunk = s.read(n_bytes - len(data))
            if not chunk:
                break
            data += chunk
        return data, time.monotonic() - t0


def describe_unwhitened(data: bytes) -> list:
    lines = [f"  bytes captured: {len(data)}"]
    if not data:
        return lines + ["  no data"]
    printable = np.mean([32 <= b < 127 or b in (10, 13) for b in data])
    lines.append(f"  printable ASCII fraction: {printable:.3f}")
    lines.append(f"  first 120 bytes: {data[:120]!r}")
    if printable > 0.95:
        text = data.decode("ascii", "replace")
        lines.append(f"  contains '-' signs: {'-' in text}")
        seps = sorted({c for c in text if not c.isdigit() and c != '-'})
        lines.append(f"  separators seen: {seps!r}")
        vals = np.array([int(v) for v in re.findall(r"-?\d+", text)[1:-1]])  # drop partial ends
        if len(vals) > 10:
            ev, od = vals[0::2], vals[1::2]
            m = min(len(ev), len(od))
            lines += [
                f"  values: {len(vals)}  min {vals.min()}  max {vals.max()}  mean {vals.mean():.1f}  sd {vals.std():.1f}",
                f"  even-position mean {ev.mean():.1f} sd {ev.std():.1f} | odd-position mean {od.mean():.1f} sd {od.std():.1f}",
                f"  LSB p(1): all {np.mean(vals & 1):.4f}  even {np.mean(ev & 1):.4f}  odd {np.mean(od & 1):.4f}",
                f"  corr(even, odd neighbours): {np.corrcoef(ev[:m], od[:m])[0, 1]:+.3f}",
                f"  lag-1 autocorrelation: {np.corrcoef(vals[:-1], vals[1:])[0, 1]:+.3f}",
                "  If even and odd positions differ in mean or sd, the stream likely alternates two",
                "  generators. Values near 0 with '-' signs suggest a difference (RNG1 - RNG2).",
            ]
    else:
        b = np.frombuffer(data, dtype=np.uint8)
        tags = np.bincount(b >> 6, minlength=4) / len(b)
        lines += [f"  binary; top-2-bit histogram {np.round(tags, 3).tolist()}",
                  f"  byte LSB p(1): even {np.mean(b[0::2] & 1):.4f} odd {np.mean(b[1::2] & 1):.4f}"]
    return lines


def describe_raw_bin(data: bytes, seconds: float) -> list:
    p = RawBinParser()
    s1, s2 = p.feed(data)
    lines = [f"  bytes captured: {len(data)} in {seconds:.2f} s",
             f"  packets: {p.packets}  resyncs: {p.resyncs}  bytes discarded: {p.bytes_discarded}"]
    if p.packets:
        lines += [
            f"  packet rate: {p.packets / seconds:,.0f} per second (each packet = 1 sample per generator)",
            f"  gen1 10-bit mean {s1.mean():.1f} sd {s1.std():.1f} | gen2 mean {s2.mean():.1f} sd {s2.std():.1f}"
            "  (vendor: mean 400-600, sd 30-100)",
            f"  LSB p(1): gen1 {np.mean(s1 & 1):.4f}  gen2 {np.mean(s2 & 1):.4f}",
            f"  corr(gen1, gen2 samples): {np.corrcoef(s1, s2)[0, 1]:+.4f}",
            f"  corr(gen1 LSB, gen2 LSB): {np.corrcoef(s1 & 1, s2 & 1)[0, 1]:+.4f}",
            f"  lag-1 autocorrelation of gen1 LSB: {np.corrcoef((s1 & 1)[:-1], (s1 & 1)[1:])[0, 1]:+.4f}",
        ]
    return lines


def run_probe(out_dir: str = "probe", n_bytes: int = 400000) -> str:
    port, model = find_port()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(out_dir, stamp)
    os.makedirs(d, exist_ok=True)
    report = [f"Device: {model} on {port}", ""]
    for mode, describe in (("MODE_UNWHITENED", None), ("MODE_RAW_BIN", None)):
        report.append(f"{mode}:")
        try:
            data, secs = _capture(port, mode, n_bytes)
            with open(os.path.join(d, f"{mode}.bin"), "wb") as f:
                f.write(data)
            report += describe_unwhitened(data) if mode == "MODE_UNWHITENED" else describe_raw_bin(data, secs)
        except Exception as e:
            report.append(f"  failed: {type(e).__name__}: {e}")
        report.append("")
    knock(port, "MODE_NORMAL")
    text = "\n".join(report)
    with open(os.path.join(d, "report.txt"), "w") as f:
        f.write(text + "\n")
    return text + f"\nRaw captures and this report saved in {d}/"
