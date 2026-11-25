import csv

import numpy as np
import pytest

from dualrng.ledger import FIELDS, append_row, combine, format_report, read_ledger


def row(i, stype="intention", cond="real", z=1.0, status="complete"):
    return {"session_id": f"s{i}", "session_type": stype, "condition": cond, "status": status,
            "z": z, "p": 0.5, "bits_analyzed": 1000, "secondary_z": ""}


def test_append_is_idempotent(tmp_path):
    path = str(tmp_path / "ledger.csv")
    assert append_row(path, row(1))
    assert not append_row(path, row(1))
    assert append_row(path, row(2))
    rows = read_ledger(path)
    assert [r["session_id"] for r in rows] == ["s1", "s2"]
    with open(path) as f:
        assert next(csv.reader(f)) == FIELDS


def test_stouffer_separates_real_and_sham(tmp_path):
    path = str(tmp_path / "ledger.csv")
    zs_real = [1.0, 2.0, 0.5, -0.2]
    zs_sham = [0.1, -0.3]
    for i, z in enumerate(zs_real):
        append_row(path, row(i, z=z))
    for i, z in enumerate(zs_sham):
        append_row(path, row(100 + i, cond="sham", z=z))
    append_row(path, row(200, z=3.0, status="stopped"))
    groups = {(g["session_type"], g["condition"]): g for g in combine(read_ledger(path))}
    real = groups[("intention", "real")]
    assert real["with_result"] == 5 and real["incomplete"] == 1
    assert real["stouffer_z"] == pytest.approx(sum(zs_real + [3.0]) / np.sqrt(5))
    only = {(g["session_type"], g["condition"]): g for g in combine(read_ledger(path), complete_only=True)}
    assert only[("intention", "real")]["stouffer_z"] == pytest.approx(sum(zs_real) / 2)
    assert groups[("intention", "sham")]["stouffer_z"] == pytest.approx(sum(zs_sham) / np.sqrt(2))
    text = format_report(path)
    assert "Real vs sham" in text
