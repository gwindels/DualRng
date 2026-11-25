"""Calibration and power of the pre-specified statistics on simulated sessions.

(a) With no effect, p-values must be uniform, including when the generators
    lean away from 0.5 (as the hardware does, p(1) ~ 0.4992).
(b) With an injected effect of stated size, the detection rate must match
    the rate expected for that effect.
"""
import math

import numpy as np
import pytest
from scipy import stats as sps

from dualrng.analysis import analyze_arrays
from dualrng.protocol import ProtocolConfig
from dualrng.sim import simulate_session
from dualrng.sources import SimModel

READS_PER_SECOND = 20.0
HW_LEAN = (0.4992, 0.4994)


def small_config(kind, **kw):
    # Short blocks keep the tests fast; the statistics do not depend on block length.
    base = dict(session_type=kind, block_seconds=10.0, gap_seconds=1.0, bits_per_read=1024,
                n_permutations=2000)
    base.update(kw)
    return ProtocolConfig(**base)


def run_many(config, model, n_sessions, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_sessions):
        reads, plan = simulate_session(config, model, READS_PER_SECOND, rng)
        res = analyze_arrays(config, reads, plan, seed=int(rng.integers(2**62)))
        out.append(res["primary"])
    return out


def assert_uniform(p, label):
    p = np.asarray(p)
    ks = sps.kstest(p, "uniform")
    rate = np.mean(p < 0.05)
    # 99.9% binomial band for the 5% rejection rate
    lo, hi = sps.binom.ppf([0.0005, 0.9995], len(p), 0.05) / len(p)
    assert ks.pvalue > 0.001, f"{label}: KS p = {ks.pvalue:.4g}"
    assert lo <= rate <= hi, f"{label}: rejection rate {rate:.3f} outside [{lo:.3f}, {hi:.3f}]"


@pytest.mark.parametrize("lean", [(0.5, 0.5), HW_LEAN])
def test_intention_null_is_uniform(lean):
    res = run_many(small_config("intention"), SimModel(lean=lean), 400, seed=11)
    assert all(r["method"] == "exact" for r in res)
    assert_uniform([r["p"] for r in res], f"intention p, lean {lean}")
    z = np.array([r["z"] for r in res])
    assert abs(z.mean()) < 0.2 and 0.8 < z.std() < 1.2


@pytest.mark.parametrize("lean", [(0.5, 0.5), HW_LEAN])
def test_connection_null_is_uniform(lean):
    res = run_many(small_config("connection"), SimModel(lean=lean), 400, seed=12)
    assert all(r["method"] == "monte_carlo" for r in res)
    assert_uniform([r["p"] for r in res], f"connection p, lean {lean}")
    assert_uniform([r["secondary_netvar"]["p"] for r in res], f"NetVar p, lean {lean}")


def test_fixed_half_baseline_would_be_biased():
    """Why the old 0.5 assumption is wrong: with the hardware lean, per-read z
    computed against p = 0.5 is centred well away from zero over a session."""
    rng = np.random.default_rng(3)
    n, reads = 8192, 1200
    ones = rng.binomial(n, 0.4992, reads)
    z_half = (ones - n * 0.5) / math.sqrt(n * 0.25)
    stouffer_half = z_half.sum() / math.sqrt(reads)
    assert stouffer_half < -3  # spurious "effect" from the lean alone


def expected_power(effect, se, df, tail, alpha=0.05):
    ncp = effect / se
    if tail == "greater":
        crit = sps.t.ppf(1 - alpha, df)
        return float(sps.nct.sf(crit, df, ncp))
    crit = sps.t.ppf(1 - alpha / 2, df)
    return float(sps.nct.sf(crit, df, ncp) + sps.nct.cdf(-crit, df, ncp))


def test_intention_power_matches_expected_rate():
    cfg = small_config("intention")
    reads_per_block = cfg.block_seconds * READS_PER_SECOND - 1  # one read straddles each edge
    bits_per_condition = cfg.blocks_per_label * reads_per_block * 2 * cfg.bits_per_read
    p = np.mean(HW_LEAN)
    se = math.sqrt(p * (1 - p) * 2 / bits_per_condition)
    delta = 1.25 * se  # HIGH +delta, LOW -delta => expected difference 2.5 SE
    expect = expected_power(2 * delta, se, df=2 * cfg.blocks_per_label - 2, tail="greater")
    res = run_many(cfg, SimModel(lean=HW_LEAN, intention_delta=delta), 300, seed=21)
    rate = np.mean([r["p"] < 0.05 for r in res])
    print(f"intention: delta={delta:.2e} expected power {expect:.3f}, observed {rate:.3f}")
    assert abs(rate - expect) < 0.09
    est = np.mean([r["statistic"] for r in res])
    assert abs(est - 2 * delta) < 0.2 * 2 * delta


def test_connection_power_matches_expected_rate():
    cfg = small_config("connection")
    reads_per_block = cfg.block_seconds * READS_PER_SECOND - 1
    c = cfg.blocks_per_label * reads_per_block
    rho = 0.06
    effect = rho / (1 - rho)          # E[z1*z2] in CONNECT with z scaled by REST
    se = math.sqrt(1 / c + 1 / c)
    expect = expected_power(effect, se, df=2 * cfg.blocks_per_label - 2, tail="two-sided")
    res = run_many(cfg, SimModel(lean=HW_LEAN, shared_rho=rho), 300, seed=22)
    rate = np.mean([r["p"] < 0.05 for r in res])
    print(f"connection: rho={rho} expected power {expect:.3f}, observed {rate:.3f}")
    assert abs(rate - expect) < 0.09


def test_exact_and_monte_carlo_agree():
    cfg = small_config("intention")
    rng = np.random.default_rng(9)
    reads, plan = simulate_session(cfg, SimModel(lean=HW_LEAN, intention_delta=2e-4), READS_PER_SECOND, rng)
    exact = analyze_arrays(cfg, reads, plan)["primary"]
    cfg_mc = small_config("intention", exact_limit=1, n_permutations=50000)
    mc = analyze_arrays(cfg_mc, reads, plan, seed=1)["primary"]
    assert exact["method"] == "exact" and mc["method"] == "monte_carlo"
    assert exact["statistic"] == pytest.approx(mc["statistic"])
    assert abs(exact["p"] - mc["p"]) < 0.01


def test_stopped_session_uses_only_completed_blocks():
    cfg = small_config("intention")
    rng = np.random.default_rng(4)
    reads, plan = simulate_session(cfg, SimModel(), READS_PER_SECOND, rng)
    stop = plan[9]["start"] + 3.0  # stopped part-way through block 9
    res = analyze_arrays(cfg, reads, plan, stop_time=stop)
    used = [b["index"] for b in res["blocks"] if b["used"]]
    assert used == list(range(9))
