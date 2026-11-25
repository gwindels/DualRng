import numpy as np
from scipy import stats as sps

from dualrng.analysis import exploratory
from dualrng.protocol import ProtocolConfig
from dualrng.scan import run_scan
from dualrng.sim import simulate_session
from dualrng.sources import SimModel

RPS = 10.0


def cfg(kind):
    return ProtocolConfig(session_type=kind, block_seconds=10.0, gap_seconds=1.0,
                          scan_windows_seconds=(2, 5, 10, 20, 40), scan_shifts=199)


def scan_ps(kind, model, n, seed):
    rng = np.random.default_rng(seed)
    c = cfg(kind)
    ps = []
    for _ in range(n):
        reads, plan = simulate_session(c, model, RPS, rng)
        ex = exploratory(c, reads, plan, 1 / RPS, seed=int(rng.integers(2**62)))
        ps.append(ex["p_scan"])
    return np.array(ps)


def test_connection_scan_null_calibrated():
    ps = scan_ps("connection", SimModel(lean=(0.4992, 0.4994)), 200, seed=31)
    rate = np.mean(ps <= 0.05)
    lo, hi = sps.binom.ppf([0.0005, 0.9995], len(ps), 0.05) / len(ps)
    assert lo <= rate <= hi, rate


def test_intention_scan_null_calibrated():
    ps = scan_ps("intention", SimModel(lean=(0.4992, 0.4994)), 200, seed=32)
    rate = np.mean(ps <= 0.05)
    lo, hi = sps.binom.ppf([0.0005, 0.9995], len(ps), 0.05) / len(ps)
    assert lo <= rate <= hi, rate


def test_scan_finds_a_local_shared_burst():
    rng = np.random.default_rng(7)
    n = 3000
    x1 = rng.binomial(1024, 0.4992, n) / 1024
    x2 = rng.binomial(1024, 0.4992, n) / 1024
    # 200 reads of strongly shared signal in the middle of an otherwise null session
    s = rng.standard_normal(200) * 0.03
    x1[1400:1600] += s
    x2[1400:1600] += s
    res = run_scan("connection", x1, x2, (5, 10, 20, 40, 80), 0.1, n_shifts=199, seed=1)
    assert res["p_scan"] <= 0.01
    assert 1200 <= res["best_start_index"] <= 1600
    assert "EXPLORATORY" in res["label"]


def test_scan_uses_full_windows_only():
    x = np.random.default_rng(0).random(100)
    res = run_scan("connection", x, x[::-1].copy(), (1, 5, 50, 500), 1.0, n_shifts=9, seed=0)
    assert max(res["windows_reads"]) <= 50
    assert min(res["windows_reads"]) >= 2


def test_connection_scan_ignores_constant_coupling():
    """The TrueRNGproV2 probe showed a small steady coupling between the two
    generators' LSBs (r ~ -0.011). A coupling that is the same all session long
    is not a local event, and must not make ordinary sessions look unusual."""
    rng = np.random.default_rng(41)
    model = SimModel(lean=(0.4834, 0.5030), shared_rho=-0.05, shared_labels=("",))
    n, sims = 10000, 150
    labels = np.array([""] * n, dtype=object)
    ps = []
    for i in range(sims):
        p1, p2 = model.p_for_reads(labels, 1024, rng)
        x1 = rng.binomial(1024, p1) / 1024
        x2 = rng.binomial(1024, p2) / 1024
        res = run_scan("connection", x1, x2, (0.5, 1, 2, 4, 8), 0.01, n_shifts=99, seed=i)
        ps.append(res["p_scan"])
    rate = np.mean(np.array(ps) <= 0.05)
    lo, hi = sps.binom.ppf([0.0005, 0.9995], sims, 0.05) / sims
    assert lo <= rate <= hi, rate
