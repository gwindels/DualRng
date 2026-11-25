"""Pre-specified session statistics and their block-label permutation tests.

Both statistics are computed from per-block sufficient statistics, so every
permutation of block labels (including re-estimating the REST baseline used
for standardization) is a matrix product.

Notation per read i and generator k: n bits, ones_k, x_k = ones_k / n.
"""
from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy import stats as sps


@dataclass
class BlockSums:
    labels: np.ndarray      # label per block (object array)
    count: np.ndarray       # reads per block
    bits: np.ndarray        # bits per generator per block (sum of n)
    O1: np.ndarray          # ones, generator 1
    O2: np.ndarray
    S1: np.ndarray          # sum of x1
    S2: np.ndarray
    Q1: np.ndarray          # sum of x1^2
    Q2: np.ndarray
    C12: np.ndarray         # sum of x1*x2

    def subset(self, mask: np.ndarray) -> "BlockSums":
        return BlockSums(*(getattr(self, f)[mask] for f in self.__dataclass_fields__))


def block_sums(block_of_read: np.ndarray, n: np.ndarray, ones1: np.ndarray, ones2: np.ndarray,
               block_labels: list) -> BlockSums:
    """Aggregate reads into blocks. block_of_read is -1 for reads outside any block."""
    nb = len(block_labels)
    keep = block_of_read >= 0
    b = block_of_read[keep]
    n = n[keep].astype(float)
    o1 = ones1[keep].astype(float)
    o2 = ones2[keep].astype(float)
    x1 = o1 / n
    x2 = o2 / n

    def agg(v):
        return np.bincount(b, weights=v, minlength=nb)

    return BlockSums(
        labels=np.asarray(block_labels, dtype=object),
        count=np.bincount(b, minlength=nb).astype(float),
        bits=agg(n), O1=agg(o1), O2=agg(o2),
        S1=agg(x1), S2=agg(x2), Q1=agg(x1 * x1), Q2=agg(x2 * x2), C12=agg(x1 * x2),
    )


# --------------------------------------------------------------------------
# Permutation machinery
# --------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _all_assignments(B: int, k: int) -> np.ndarray:
    total = math.comb(B, k)
    A = np.zeros((total, B))
    for r, idx in enumerate(combinations(range(B), k)):
        A[r, list(idx)] = 1.0
    A.setflags(write=False)
    return A


def _assignments(a_obs: np.ndarray, exact_limit: int, n_perm: int, seed: int):
    """Rows of label assignments (1 = treatment block) to evaluate.

    Exact: every way to choose k treatment blocks out of B (includes the observed one).
    Monte Carlo: n_perm uniformly random relabelings (the observed one is added
    separately in the p-value).
    """
    B = len(a_obs)
    k = int(a_obs.sum())
    total = math.comb(B, k)
    if total <= exact_limit:
        return _all_assignments(B, k), "exact", total
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((n_perm, B)), axis=1)
    return a_obs[idx].astype(float), "monte_carlo", n_perm


def _p_values(t_obs: float, T: np.ndarray, method: str) -> dict:
    tol = 1e-9 * (abs(t_obs) + float(np.std(T)) + 1e-300)
    gt = int(np.sum(T > t_obs + tol))
    lt = int(np.sum(T < t_obs - tol))
    eq = len(T) - gt - lt
    if method == "exact":
        m = len(T)
        p_greater = (gt + eq) / m
        p_less = (lt + eq) / m
        p_mid = (gt + 0.5 * eq) / m
    else:
        m = len(T) + 1  # the observed assignment counts as one permutation
        p_greater = (gt + eq + 1) / m
        p_less = (lt + eq + 1) / m
        p_mid = (gt + 0.5 * eq + 0.5) / m
    p_two = min(1.0, 2 * min(p_greater, p_less))
    p_mid = min(max(p_mid, 0.5 / m), 1 - 0.5 / m)
    return {"p_greater": p_greater, "p_less": p_less, "p_two_sided": p_two,
            # Normal-equivalent of the permutation result: positive when the
            # statistic is high. Used for combining sessions (Stouffer).
            "z": float(sps.norm.isf(p_mid))}


def select_p(pv: dict, tail: str) -> float:
    return {"greater": pv["p_greater"], "less": pv["p_less"], "two-sided": pv["p_two_sided"]}[tail]


# --------------------------------------------------------------------------
# Intention: p1(HIGH) - p1(LOW), both generators pooled
# --------------------------------------------------------------------------

def lean_estimates(bs: BlockSums, label: str = "REST") -> dict:
    m = bs.labels == label
    if bs.bits[m].sum() == 0:
        m = np.ones(len(bs.labels), bool)
    bits = bs.bits[m].sum()
    return {"p1": float(bs.O1[m].sum() / bits), "p2": float(bs.O2[m].sum() / bits),
            "pooled": float((bs.O1[m].sum() + bs.O2[m].sum()) / (2 * bits)),
            "from": label if (bs.labels == label).any() else "all"}


def intention_test(bs: BlockSums, tail: str = "greater", exact_limit: int = 200000,
                   n_perm: int = 20000, seed: int | None = None) -> dict:
    act = np.isin(bs.labels, ["HIGH", "LOW"]) & (bs.count > 0)
    a = bs.subset(act)
    a_obs = (a.labels == "HIGH").astype(float)
    if a_obs.sum() < 1 or (1 - a_obs).sum() < 1:
        raise ValueError("Need at least one HIGH and one LOW block with data")
    O = a.O1 + a.O2
    N = 2 * a.bits

    def stat(A):
        return (A @ O) / (A @ N) - ((1 - A) @ O) / ((1 - A) @ N)

    t_obs = float(stat(a_obs[None, :])[0])
    seed = secrets.randbits(63) if seed is None else seed
    A, method, m = _assignments(a_obs, exact_limit, n_perm, seed)
    pv = _p_values(t_obs, stat(A), method)

    lean = lean_estimates(bs)
    pq = lean["pooled"] * (1 - lean["pooled"])
    n_h, n_l = float(N @ a_obs), float(N @ (1 - a_obs))
    z_par = t_obs / math.sqrt(pq * (1 / n_h + 1 / n_l))
    return {
        "statistic": t_obs,
        "p": select_p(pv, tail), "tail": tail, **pv,
        "z_parametric": z_par,
        "method": method, "n_assignments": m, "seed": seed if method == "monte_carlo" else None,
        "p_high": float(a_obs @ O / n_h), "p_low": float((1 - a_obs) @ O / n_l),
        "lean": lean,
        "blocks": {"HIGH": int(a_obs.sum()), "LOW": int((1 - a_obs).sum()),
                   "REST": int(((bs.labels == "REST") & (bs.count > 0)).sum())},
        "bits_analyzed": float(N.sum()),
    }


# --------------------------------------------------------------------------
# Connection: mean z1*z2 in CONNECT - REST, z standardized from REST blocks
# --------------------------------------------------------------------------

def _connection_stats(A: np.ndarray, a: BlockSums):
    """Return (product statistic, NetVar statistic) for assignment rows A (1 = CONNECT)."""
    R = 1 - A
    cR = R @ a.count
    cC = A @ a.count
    m1 = (R @ a.S1) / cR
    m2 = (R @ a.S2) / cR
    v1 = ((R @ a.Q1) - cR * m1 ** 2) / (cR - 1)
    v2 = ((R @ a.Q2) - cR * m2 ** 2) / (cR - 1)
    s1 = np.sqrt(v1)
    s2 = np.sqrt(v2)

    def sums(W, cnt):
        S1, S2 = W @ a.S1, W @ a.S2
        prod = ((W @ a.C12) - m2 * S1 - m1 * S2 + cnt * m1 * m2) / (s1 * s2)
        zz1 = ((W @ a.Q1) - 2 * m1 * S1 + cnt * m1 ** 2) / v1
        zz2 = ((W @ a.Q2) - 2 * m2 * S2 + cnt * m2 ** 2) / v2
        netvar = (zz1 + zz2 + 2 * prod) / 2  # sum of ((z1 + z2) / sqrt 2)^2
        return prod / cnt, netvar / cnt

    pc, nc = sums(A, cC)
    pr, nr = sums(R, cR)
    return pc - pr, nc - nr, (pc, pr, nc, nr)


def connection_test(bs: BlockSums, tail: str = "two-sided", exact_limit: int = 200000,
                    n_perm: int = 20000, seed: int | None = None) -> dict:
    act = np.isin(bs.labels, ["CONNECT", "REST"]) & (bs.count > 0)
    a = bs.subset(act)
    a_obs = (a.labels == "CONNECT").astype(float)
    if a_obs.sum() < 1 or (1 - a_obs).sum() < 2:
        raise ValueError("Need at least one CONNECT and two REST blocks with data")
    t_obs, nv_obs, parts = _connection_stats(a_obs[None, :], a)
    t_obs, nv_obs = float(t_obs[0]), float(nv_obs[0])
    seed = secrets.randbits(63) if seed is None else seed
    A, method, m = _assignments(a_obs, exact_limit, n_perm, seed)
    T, NV, _ = _connection_stats(A, a)
    pv = _p_values(t_obs, T, method)
    pv_nv = _p_values(nv_obs, NV, method)

    cC, cR = float(a_obs @ a.count), float((1 - a_obs) @ a.count)
    return {
        "statistic": t_obs,
        "p": select_p(pv, tail), "tail": tail, **pv,
        "z_parametric": t_obs / math.sqrt(1 / cC + 1 / cR),
        "method": method, "n_assignments": m, "seed": seed if method == "monte_carlo" else None,
        "mean_product_connect": float(parts[0][0]), "mean_product_rest": float(parts[1][0]),
        "secondary_netvar": {
            "statistic": nv_obs,
            # NetVar is pre-specified as one-sided: elevated in CONNECT.
            "p": pv_nv["p_greater"], "tail": "greater", **pv_nv,
            "mean_netvar_connect": float(parts[2][0]), "mean_netvar_rest": float(parts[3][0]),
        },
        "lean": lean_estimates(bs),
        "blocks": {"CONNECT": int(a_obs.sum()), "REST": int((1 - a_obs).sum())},
        "bits_analyzed": float(2 * a.bits.sum()),
    }


def run_primary(session_type: str, bs: BlockSums, tail: str, exact_limit: int, n_perm: int,
                seed: int | None = None) -> dict:
    if session_type == "intention":
        return intention_test(bs, tail, exact_limit, n_perm, seed)
    if session_type == "connection":
        return connection_test(bs, tail, exact_limit, n_perm, seed)
    raise ValueError(session_type)


def stouffer(z: list) -> tuple[float, int]:
    z = [v for v in z if v is not None and np.isfinite(v)]
    if not z:
        return float("nan"), 0
    return float(np.sum(z) / math.sqrt(len(z))), len(z)
