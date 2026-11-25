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


def test_old_ledger_is_migrated_to_new_columns(tmp_path):
    path = str(tmp_path / "ledger.csv")
    old_fields = [f for f in FIELDS if f != "feedback"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=old_fields)
        w.writeheader()
        w.writerow({k: "" for k in old_fields} | {"session_id": "old1", "session_type": "intention",
                                                    "condition": "real", "status": "complete", "z": "1.5"})
    new = row(2, z=0.5)
    new["feedback"] = "trace"
    assert append_row(path, new)
    rows = read_ledger(path)
    with open(path) as f:
        assert next(csv.reader(f)) == FIELDS
    assert [r["session_id"] for r in rows] == ["old1", "s2"]
    assert rows[0]["z"] == "1.5" and rows[0]["feedback"] == "" and rows[1]["feedback"] == "trace"


def test_report_splits_by_display_when_mixed(tmp_path):
    from dualrng.ledger import by_feedback
    path = str(tmp_path / "ledger.csv")
    for i, (fb, z) in enumerate([("trace", 1.0), ("trace", 2.0), ("wave", -1.0)]):
        r = row(i, z=z)
        r["feedback"] = fb
        append_row(path, r)
    parts = {g["feedback"]: g for g in by_feedback(read_ledger(path))}
    assert parts["trace"]["k"] == 2 and parts["trace"]["stouffer_z"] == pytest.approx(3 / np.sqrt(2))
    assert parts["wave"]["k"] == 1
    assert "By display" in format_report(path)
