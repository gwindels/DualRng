import os
import time

import numpy as np

from dualrng.records import ReadsTailer, ReadsWriter, load_records, pair_records, unpack_bits
from dualrng.recorder import Recorder
from dualrng.sources import RawBinParser, SimModel


def encode_packets(s1, s2):
    """Encode 10-bit sample pairs as RAW_BIN packets (inverse of the documented layout)."""
    s1 = np.asarray(s1, dtype=np.uint16)
    s2 = np.asarray(s2, dtype=np.uint16)
    out = np.empty((len(s1), 4), dtype=np.uint8)
    out[:, 0] = 0x00 | ((s1 >> 6) & 0x0F)
    out[:, 1] = 0x40 | (s1 & 0x3F)
    out[:, 2] = 0x80 | ((s2 >> 6) & 0x0F)
    out[:, 3] = 0xC0 | (s2 & 0x3F)
    return out.tobytes()


def test_raw_bin_roundtrip_with_chunking_and_garbage():
    rng = np.random.default_rng(1)
    s1 = rng.integers(0, 1024, 5000)
    s2 = rng.integers(0, 1024, 5000)
    stream = encode_packets(s1[:2000], s2[:2000])
    # A dropped byte mid-stream (e.g. a USB hiccup) and leading garbage
    stream = b"\xC1\x80" + stream[:4001] + stream[4002:] + encode_packets(s1[2000:], s2[2000:])
    p = RawBinParser()
    got1, got2 = [], []
    for i in range(0, len(stream), 333):  # arbitrary chunking that splits packets
        a, b = p.feed(stream[i:i + 333])
        got1.append(a)
        got2.append(b)
    got1 = np.concatenate(got1)
    got2 = np.concatenate(got2)
    # Packet 1000 was damaged; every other packet must come through, still paired.
    keep = np.r_[0:1000, 1001:5000]
    assert np.array_equal(got1, s1[keep])
    assert np.array_equal(got2, s2[keep])
    assert p.resyncs >= 2
    assert p.bytes_discarded > 0


def test_writer_tailer_and_truncation(tmp_path):
    path = str(tmp_path / "reads.bin")
    w = ReadsWriter(path, 64)
    tail = ReadsTailer(path)
    rng = np.random.default_rng(0)
    bits = []
    for seq in range(5):
        b1 = rng.integers(0, 2, 64).astype(np.uint8)
        b2 = rng.integers(0, 2, 64).astype(np.uint8)
        bits.append((b1, b2))
        w.write_pair(seq, seq * 1.0, seq * 1.0 + 0.5, b1, b2)
        got = tail.poll()
        assert len(got) == 2 and got["seq"][0] == seq
    w.close()
    # Simulate power loss in the middle of the last record.
    with open(path, "ab") as f:
        f.write(b"\x00" * 17)
    recs = load_records(path)
    assert len(recs) == 10
    paired = pair_records(recs)
    assert np.array_equal(paired["ones1"], [b[0].sum() for b in bits])
    assert np.array_equal(paired["ones2"], [b[1].sum() for b in bits])
    unpacked = unpack_bits(recs[recs["gen"] == 1])
    assert np.array_equal(unpacked[3], bits[3][0])


def test_shared_rho_produces_stated_correlation():
    rng = np.random.default_rng(5)
    n = 1024
    m = SimModel(lean=(0.4992, 0.4992), shared_rho=0.2)
    labels = np.array(["CONNECT"] * 40000, dtype=object)
    p1, p2 = m.p_for_reads(labels, n, rng)
    o1 = rng.binomial(n, p1)
    o2 = rng.binomial(n, p2)
    r = np.corrcoef(o1, o2)[0, 1]
    assert abs(r - 0.2) < 0.02


def test_recorder_process_with_sim_source(tmp_path):
    rec = Recorder(str(tmp_path), {"kind": "sim", "reads_per_second": 200, "seed": 3}, 256)
    rec.start()
    time.sleep(1.0)
    rec.stop()
    assert not rec.failed
    recs = load_records(str(tmp_path / "reads.bin"))
    paired = pair_records(recs)
    assert len(paired["seq"]) > 50
    assert np.all(np.diff(paired["seq"]) == 1)          # no gaps
    assert np.all(np.diff(paired["t_start"]) > 0)
    assert os.path.exists(tmp_path / "acq_log.csv")


def test_probe_descriptions_on_synthetic_captures():
    from dualrng.probe import describe_raw_bin, describe_unwhitened

    rng = np.random.default_rng(2)
    s1 = rng.normal(490, 50, 3000).clip(0, 1023).astype(int)
    s2 = rng.normal(530, 40, 3000).clip(0, 1023).astype(int)
    text = "\n".join(describe_raw_bin(encode_packets(s1, s2), 0.5))
    assert "packets: 3000" in text and "resyncs: 0" in text
    # An ASCII stream alternating two generators with different means
    vals = np.ravel(np.column_stack([s1, s2]))
    ascii_stream = (",".join(str(v) for v in vals) + ",").encode()
    text = "\n".join(describe_unwhitened(ascii_stream))
    assert "even-position mean" in text and "contains '-' signs: False" in text
