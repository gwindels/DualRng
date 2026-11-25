"""Raw session file format.

reads.bin is an append-only binary file:

    header  16 bytes   b"DUALRNG1" | uint32 bits_per_read | uint32 format version
    records fixed size, one per generator per read:
        t_start  float64   time.monotonic() before the read began
        t_end    float64   time.monotonic() when the read completed
        seq      uint32    read number; both generators of one read share it
        gen      uint8     1 or 2
        (pad)    3 bytes
        n        uint32    number of bits in this read
        ones     uint32    number of 1 bits
        bits     n/8 bytes LSBs, packed MSB-first (numpy.packbits)

Every record carries both the raw bits and the ones count, so the file can be
re-analysed at any granularity, and a truncated tail (power loss mid-write)
only loses the last partial record.
"""
from __future__ import annotations

import os
import time

import numpy as np

MAGIC = b"DUALRNG1"
FORMAT_VERSION = 1
HEADER_SIZE = 16


def record_dtype(bits_per_read: int) -> np.dtype:
    if bits_per_read <= 0 or bits_per_read % 8:
        raise ValueError("bits_per_read must be a positive multiple of 8")
    return np.dtype([
        ("t_start", "<f8"),
        ("t_end", "<f8"),
        ("seq", "<u4"),
        ("gen", "u1"),
        ("_pad", "u1", (3,)),
        ("n", "<u4"),
        ("ones", "<u4"),
        ("bits", "u1", (bits_per_read // 8,)),
    ])


def _header(bits_per_read: int) -> bytes:
    return MAGIC + np.array([bits_per_read, FORMAT_VERSION], dtype="<u4").tobytes()


def read_header(path: str) -> int:
    with open(path, "rb") as f:
        head = f.read(HEADER_SIZE)
    if len(head) < HEADER_SIZE or head[:8] != MAGIC:
        raise ValueError(f"{path} is not a DualRng reads file")
    bits_per_read, version = np.frombuffer(head[8:], dtype="<u4")
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported reads file version {version}")
    return int(bits_per_read)


class ReadsWriter:
    """Appends records and makes them durable.

    Each write is flushed to the OS immediately (so readers in other processes
    see it) and fsynced at most every `fsync_interval` seconds.
    """

    def __init__(self, path: str, bits_per_read: int, fsync_interval: float = 1.0):
        self.bits_per_read = bits_per_read
        self.dtype = record_dtype(bits_per_read)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        if not new and read_header(path) != bits_per_read:
            raise ValueError("Existing reads file has a different bits_per_read")
        self._f = open(path, "ab")
        if new:
            self._f.write(_header(bits_per_read))
            self._f.flush()
            os.fsync(self._f.fileno())
        self._fsync_interval = fsync_interval
        self._last_sync = time.monotonic()

    def write_pair(self, seq: int, t_start: float, t_end: float,
                   bits1: np.ndarray, bits2: np.ndarray) -> None:
        recs = np.zeros(2, dtype=self.dtype)
        for i, (gen, bits) in enumerate(((1, bits1), (2, bits2))):
            bits = np.asarray(bits, dtype=np.uint8)
            if bits.size != self.bits_per_read:
                raise ValueError("wrong number of bits")
            recs[i]["t_start"] = t_start
            recs[i]["t_end"] = t_end
            recs[i]["seq"] = seq
            recs[i]["gen"] = gen
            recs[i]["n"] = bits.size
            recs[i]["ones"] = int(bits.sum())
            recs[i]["bits"] = np.packbits(bits)
        self._f.write(recs.tobytes())
        self._f.flush()
        now = time.monotonic()
        if now - self._last_sync >= self._fsync_interval:
            os.fsync(self._f.fileno())
            self._last_sync = now

    def close(self) -> None:
        if self._f.closed:
            return
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()


class ReadsTailer:
    """Reads complete records appended since the last call. Never writes."""

    def __init__(self, path: str):
        self.path = path
        self.dtype = None
        self._offset = HEADER_SIZE

    def poll(self) -> np.ndarray | None:
        if self.dtype is None:
            if not os.path.exists(self.path) or os.path.getsize(self.path) < HEADER_SIZE:
                return None
            self.dtype = record_dtype(read_header(self.path))
        size = os.path.getsize(self.path)
        n_new = (size - self._offset) // self.dtype.itemsize
        if n_new <= 0:
            return None
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            buf = f.read(n_new * self.dtype.itemsize)
        n_new = len(buf) // self.dtype.itemsize
        self._offset += n_new * self.dtype.itemsize
        return np.frombuffer(buf[: n_new * self.dtype.itemsize], dtype=self.dtype)


def load_records(path: str) -> np.ndarray:
    """All complete records in a reads file (a truncated final record is ignored)."""
    bits_per_read = read_header(path)
    dtype = record_dtype(bits_per_read)
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE)
        buf = f.read()
    n = len(buf) // dtype.itemsize
    return np.frombuffer(buf[: n * dtype.itemsize], dtype=dtype)


def pair_records(recs: np.ndarray) -> dict:
    """Join generator 1 and 2 records on seq. Reads missing either side are dropped.

    Returns arrays sorted by seq: seq, t_start, t_end, n, ones1, ones2.
    """
    g1 = recs[recs["gen"] == 1]
    g2 = recs[recs["gen"] == 2]
    common, i1, i2 = np.intersect1d(g1["seq"], g2["seq"], return_indices=True)
    a, b = g1[i1], g2[i2]
    if np.any(a["n"] != b["n"]):
        raise ValueError("Paired reads have different bit counts")
    return {
        "seq": common.astype(np.int64),
        "t_start": np.minimum(a["t_start"], b["t_start"]),
        "t_end": np.maximum(a["t_end"], b["t_end"]),
        "n": a["n"].astype(np.int64),
        "ones1": a["ones"].astype(np.int64),
        "ones2": b["ones"].astype(np.int64),
    }


def unpack_bits(recs: np.ndarray) -> np.ndarray:
    """Unpack the bits field of records into an (n_records, bits_per_read) array."""
    return np.unpackbits(recs["bits"], axis=1)
