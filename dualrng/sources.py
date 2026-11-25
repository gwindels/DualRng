"""Bit sources: the TrueRNGpro in RAW_BIN mode, and a simulator.

Hardware notes (see README "Hardware documentation"):

* Mode selection is by a baud-rate "knock" (110, 300, 110, then the mode's
  baud). This is documented by ubld.it.
* MODE_RAW_BIN (19200) streams 4-byte packets. Bits 7:6 of every byte are a
  sequence code: 00 = generator 1 high, 01 = generator 1 low, 10 = generator 2
  high, 11 = generator 2 low. The high byte carries ADC bits 9:6 in bits 3:0;
  the low byte carries ADC bits 5:0 in bits 5:0. One packet therefore holds one
  10-bit sample from each generator, and pairing by packet is guaranteed by
  the format. The documentation does not say the two samples are taken at the
  same instant.
* MODE_UNWHITENED (57600, TrueRNGproV2 only) appears in the vendor's example
  scripts with no description of its output, and the per-generator command
  bytes (0xC4/0xC5) used by the original DualRng code are not documented, so
  this module does not use them. `python app.py probe` inspects both modes.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import clock

TRUERNG_IDS = {
    "04D8:F5FE": "TrueRNG",
    "16D0:0AA0": "TrueRNGpro",
    "04D8:EBB5": "TrueRNGproV2",
}

MODE_BAUD = {
    "MODE_NORMAL": 300,
    "MODE_PSDEBUG": 1200,
    "MODE_RNGDEBUG": 2400,
    "MODE_RNG1WHITE": 4800,
    "MODE_RNG2WHITE": 9600,
    "MODE_RAW_BIN": 19200,
    "MODE_RAW_ASC": 38400,
    "MODE_UNWHITENED": 57600,
    "MODE_NORMAL_ASC": 115200,
    "MODE_NORMAL_ASC_SLOW": 230400,
}


def find_port() -> tuple[str, str]:
    """Return (port, model) of the first TrueRNG device found."""
    from serial.tools import list_ports

    for p in list_ports.comports():
        for vidpid, model in TRUERNG_IDS.items():
            if vidpid in (p.hwid or ""):
                return p.device, model
    raise RuntimeError("TrueRNG device not found")


def knock(port: str, mode: str) -> None:
    """Select a device mode with the documented baud-rate knock sequence."""
    import serial

    if mode not in MODE_BAUD:
        raise ValueError(f"Unknown mode {mode}")
    for baud in (110, 300, 110, MODE_BAUD[mode]):
        s = serial.Serial(port=port, baudrate=baud, timeout=1)
        s.close()
    time.sleep(0.5)


# --------------------------------------------------------------------------
# RAW_BIN parsing
# --------------------------------------------------------------------------

class RawBinParser:
    """Incremental parser for MODE_RAW_BIN packets.

    feed() accepts arbitrary byte chunks and returns the 10-bit samples of every
    complete, correctly tagged packet. Bytes that do not fit the expected tag
    sequence are discarded until the next aligned packet, and counted.
    """

    EXPECTED = np.array([0, 1, 2, 3], dtype=np.uint8)

    def __init__(self):
        self._buf = np.zeros(0, dtype=np.uint8)
        self.bytes_discarded = 0
        self.resyncs = 0
        self.packets = 0

    def feed(self, data: bytes) -> tuple[np.ndarray, np.ndarray]:
        buf = np.concatenate([self._buf, np.frombuffer(data, dtype=np.uint8)])
        out1, out2 = [], []
        pos = 0
        while len(buf) - pos >= 4:
            # Vectorised: take as many aligned packets as possible from pos.
            n = (len(buf) - pos) // 4
            block = buf[pos: pos + 4 * n].reshape(n, 4)
            ok = np.all((block >> 6) == self.EXPECTED, axis=1)
            bad = np.flatnonzero(~ok)
            good_n = n if bad.size == 0 else int(bad[0])
            if good_n:
                good = block[:good_n]
                out1.append(((good[:, 0] & 0x0F).astype(np.uint16) << 6) | (good[:, 1] & 0x3F))
                out2.append(((good[:, 2] & 0x0F).astype(np.uint16) << 6) | (good[:, 3] & 0x3F))
                self.packets += good_n
                pos += 4 * good_n
            if good_n == n:
                break
            # Misaligned at pos: skip one byte at a time until a full aligned packet.
            self.resyncs += 1
            start = pos
            pos += 1
            while len(buf) - pos >= 4 and not np.all((buf[pos: pos + 4] >> 6) == self.EXPECTED):
                pos += 1
            self.bytes_discarded += pos - start
        self._buf = buf[pos:].copy()
        if out1:
            return np.concatenate(out1), np.concatenate(out2)
        return np.zeros(0, np.uint16), np.zeros(0, np.uint16)


class HardwareSource:
    """Continuous paired reads from a TrueRNGpro in MODE_RAW_BIN.

    Each call to read_pair() returns exactly `bits_per_read` LSBs per
    generator, both taken from the same packets.
    """

    kind = "hardware"

    def __init__(self, bits_per_read: int, port: str | None = None, chunk_bytes: int = 4096):
        self.bits_per_read = bits_per_read
        self.port = port
        self.model = None
        self.chunk_bytes = chunk_bytes
        self.parser = RawBinParser()
        self._ser = None
        self._pending1 = np.zeros(0, np.uint8)
        self._pending2 = np.zeros(0, np.uint8)

    def open(self) -> None:
        import serial

        if self.port is None:
            self.port, self.model = find_port()
        if self.model == "TrueRNG":
            raise RuntimeError("A plain TrueRNG has one generator and no raw mode")
        knock(self.port, "MODE_RAW_BIN")
        self._ser = serial.Serial(self.port, timeout=1)
        self._ser.setDTR(True)
        time.sleep(0.1)
        self._ser.reset_input_buffer()

    def describe(self) -> dict:
        return {"kind": self.kind, "port": self.port, "model": self.model,
                "mode": "MODE_RAW_BIN", "bit": "ADC LSB"}

    def stats(self) -> dict:
        return {"packets": self.parser.packets, "resyncs": self.parser.resyncs,
                "bytes_discarded": self.parser.bytes_discarded}

    def read_pair(self) -> tuple[float, float, np.ndarray, np.ndarray]:
        t_start = clock.now()
        n = self.bits_per_read
        while len(self._pending1) < n:
            data = self._ser.read(self.chunk_bytes)
            if not data:
                raise IOError("No data from TrueRNG (timeout)")
            s1, s2 = self.parser.feed(data)
            self._pending1 = np.concatenate([self._pending1, (s1 & 1).astype(np.uint8)])
            self._pending2 = np.concatenate([self._pending2, (s2 & 1).astype(np.uint8)])
        t_end = clock.now()
        b1, self._pending1 = self._pending1[:n], self._pending1[n:]
        b2, self._pending2 = self._pending2[:n], self._pending2[n:]
        return t_start, t_end, b1, b2

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

@dataclass
class SimModel:
    """Generative model shared by the bit-level source and the fast simulator.

    lean            p(1) of each generator with no effect (hardware sits near 0.4992)
    intention_delta added to p(1) of both generators in HIGH blocks and
                    subtracted in LOW blocks
    shared_rho      target correlation between the two generators' per-read z
                    in the labels listed in shared_labels (default CONNECT),
                    produced by a common random shift of p(1)
    """

    lean: tuple = (0.4992, 0.4992)
    intention_delta: float = 0.0
    shared_rho: float = 0.0
    shared_labels: tuple = ("CONNECT",)

    def p_for_reads(self, labels: np.ndarray, bits_per_read: int, rng: np.random.Generator
                    ) -> tuple[np.ndarray, np.ndarray]:
        labels = np.asarray(labels, dtype=object)
        p1 = np.full(labels.shape, float(self.lean[0]))
        p2 = np.full(labels.shape, float(self.lean[1]))
        if self.intention_delta:
            sign = np.where(labels == "HIGH", 1.0, np.where(labels == "LOW", -1.0, 0.0))
            p1 = p1 + sign * self.intention_delta
            p2 = p2 + sign * self.intention_delta
        if self.shared_rho:
            mask = np.isin(labels, list(self.shared_labels))
            if mask.any():
                pq = float(np.mean(self.lean) * (1 - np.mean(self.lean)))
                rho = self.shared_rho
                c = np.sqrt(abs(rho) * pq / (bits_per_read * (1 - abs(rho))))
                s = rng.standard_normal(int(mask.sum()))
                p1[mask] += c * s
                p2[mask] += np.sign(rho) * c * s
        return np.clip(p1, 0, 1), np.clip(p2, 0, 1)

    def to_dict(self) -> dict:
        return {"lean": list(self.lean), "intention_delta": self.intention_delta,
                "shared_rho": self.shared_rho, "shared_labels": list(self.shared_labels)}


def load_plan(session_dir: str) -> list[dict] | None:
    path = os.path.join(session_dir, "schedule.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [{"label": r["label"], "start": float(r["start_mono"]), "end": float(r["end_mono"])}
            for r in rows]


def label_at(plan: list[dict] | None, t: float) -> str:
    if not plan:
        return ""
    for b in plan:
        if b["start"] <= t < b["end"]:
            return b["label"]
    return ""


class SimSource:
    """Bit-level simulated source paced in real time.

    The effect (if any) follows the session's schedule.csv, read from the
    session directory once it has been written, so injected effects line up
    with the blocks exactly as a real effect would.
    """

    kind = "sim"

    def __init__(self, bits_per_read: int, model: SimModel | None = None,
                 reads_per_second: float = 50.0, session_dir: str | None = None,
                 seed: int | None = None):
        self.bits_per_read = bits_per_read
        self.model = model or SimModel()
        self.reads_per_second = reads_per_second
        self.session_dir = session_dir
        self.rng = np.random.default_rng(seed)
        self._plan = None
        self._next_t = None

    def open(self) -> None:
        self._next_t = clock.now()

    def describe(self) -> dict:
        return {"kind": self.kind, "model": self.model.to_dict(),
                "reads_per_second": self.reads_per_second}

    def stats(self) -> dict:
        return {}

    def read_pair(self):
        period = 1.0 / self.reads_per_second if self.reads_per_second > 0 else 0.0
        t_start = clock.now()
        self._next_t += period
        delay = self._next_t - clock.now()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next_t = clock.now()
        if self._plan is None and self.session_dir:
            self._plan = load_plan(self.session_dir)
        label = label_at(self._plan, (t_start + clock.now()) / 2)
        p1, p2 = self.model.p_for_reads(np.array([label], dtype=object), self.bits_per_read, self.rng)
        b1 = (self.rng.random(self.bits_per_read) < p1[0]).astype(np.uint8)
        b2 = (self.rng.random(self.bits_per_read) < p2[0]).astype(np.uint8)
        return t_start, clock.now(), b1, b2

    def close(self) -> None:
        pass


def build_source(spec: dict, bits_per_read: int, session_dir: str | None = None):
    kind = spec.get("kind", "hardware")
    if kind == "hardware":
        return HardwareSource(bits_per_read, port=spec.get("port"))
    if kind == "sim":
        model = SimModel(
            lean=tuple(spec.get("lean", (0.4992, 0.4992))),
            intention_delta=float(spec.get("intention_delta", 0.0)),
            shared_rho=float(spec.get("shared_rho", 0.0)),
            shared_labels=tuple(spec.get("shared_labels", ("CONNECT",))),
        )
        return SimSource(bits_per_read, model, float(spec.get("reads_per_second", 50.0)),
                         session_dir=session_dir, seed=spec.get("seed"))
    raise ValueError(f"Unknown source kind {kind}")
