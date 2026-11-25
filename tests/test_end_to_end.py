"""Whole-program tests on the simulated source: GUI, headless, and crash recovery.

These cover the original bug where GUI sessions were not saved: every way of
ending a session must leave a finalized session and a ledger row.
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = ["--source", "sim", "--block-seconds", "1", "--gap-seconds", "0.3",
        "--blocks-per-label", "2", "--sim-rate", "50"]


def ledger_rows(base):
    import csv
    path = os.path.join(base, "ledger.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def only_session(base):
    dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    assert len(dirs) == 1, dirs
    return os.path.join(base, dirs[0])


def test_headless_complete_session(tmp_path):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "app.py"), "--base-dir", str(tmp_path),
                          "run", "--type", "connection", *FAST],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    d = only_session(tmp_path)
    for f in ("meta.json", "schedule.csv", "reads.bin", "results.json", "exploratory.json", "session.png"):
        assert os.path.exists(os.path.join(d, f)), f
    rows = ledger_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["status"] == "complete" and rows[0]["condition"] == "real"
    assert rows[0]["secondary_name"] == "netvar_connect_minus_rest"
    # Exploratory output must never reach the ledger
    assert "p_scan" not in open(os.path.join(tmp_path, "ledger.csv")).read()


def test_injected_effect_lands_in_the_right_blocks(tmp_path):
    """Through the real recorder process: a strong HIGH/LOW effect must show up
    with the right sign. This only works if reads are assigned to the blocks
    they were recorded in."""
    out = subprocess.run([sys.executable, os.path.join(ROOT, "app.py"), "--base-dir", str(tmp_path),
                          "run", "--type", "intention", "--source", "sim", "--block-seconds", "1.5",
                          "--gap-seconds", "0.3", "--blocks-per-label", "3", "--sim-rate", "118",
                          "--sim-delta", "0.01"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    res = json.load(open(os.path.join(only_session(tmp_path), "results.json")))
    p = res["primary"]
    assert p["p_high"] - p["p_low"] > 0.015          # injected difference is 0.02
    assert p["p"] == pytest.approx(1 / 20)          # the most extreme of the 20 assignments
    for b in res["blocks"]:
        if b["label"] in ("HIGH", "LOW") and b["used"]:
            dev = (b["p1"] + b["p2"]) / 2 - p["lean"]["pooled"]
            assert dev > 0 if b["label"] == "HIGH" else dev < 0, b


def test_killed_session_is_recovered_on_next_start(tmp_path):
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "app.py"), "--base-dir", str(tmp_path),
                             "run", "--type", "intention", "--block-seconds", "2", "--gap-seconds", "0.2",
                             "--blocks-per-label", "3", "--source", "sim"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(12)  # several blocks in
    os.kill(proc.pid, signal.SIGKILL)  # like a power cut: no cleanup at all
    proc.wait()
    d = only_session(tmp_path)
    # The orphaned recorder must notice and release the device by itself.
    log = os.path.join(d, "acq_log.csv")
    assert _wait(lambda: "parent_gone" in open(log).read(), timeout=10)
    assert not os.path.exists(os.path.join(d, "results.json"))
    assert ledger_rows(tmp_path) == []
    out = subprocess.run([sys.executable, os.path.join(ROOT, "app.py"), "--base-dir", str(tmp_path),
                          "analyze", "--pending"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    rows = ledger_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["status"] in ("interrupted", "interrupted-no-result")
    res = json.load(open(os.path.join(d, "results.json")))
    assert any(b["used"] for b in res["blocks"])


@pytest.fixture
def gui_env(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    sys.path.insert(0, ROOT)
    yield


def _run_gui(tmp_path, script):
    """Run the real GUI loop in this thread while `script(gui, click)` drives it from another."""
    import importlib

    import pygame
    app = importlib.import_module("app")
    from sine_gui import SineGUI

    parser_args = ["--base-dir", str(tmp_path), "gui", "--windowed", *FAST]
    args = _parse(app, parser_args)
    gui = SineGUI(fullscreen=False)
    gui.controller = app.Controller(gui, args)
    errors = []

    def click(x, y):
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(x), int(y)), button=1))

    def driver():
        try:
            script(gui, click)
        except Exception as e:  # surface driver failures in the test
            errors.append(e)
            gui.running = False

    t = threading.Thread(target=driver, daemon=True)
    t.start()
    gui.run()
    t.join(5)
    assert not errors, errors
    assert gui._closed
    return gui


def _parse(app, argv):
    captured = {}

    def fake_gui(args):
        captured["args"] = args
    orig = app.cmd_gui
    app.cmd_gui = fake_gui
    try:
        app.main(argv)
    finally:
        app.cmd_gui = orig
    return captured["args"]


def _wait(cond, timeout=30):
    t = time.time()
    while time.time() - t < timeout:
        if cond():
            return True
        time.sleep(0.1)
    return False


def test_gui_stop_button_saves_session(tmp_path, gui_env):
    def script(gui, click):
        time.sleep(0.5)
        cx, cy = gui.width // 2, gui.height // 2
        click(cx, cy - 100)                       # Connection
        time.sleep(0.2)
        click(cx, cy + 20)                        # Start Session
        assert _wait(lambda: gui.mode == "session")
        time.sleep(3.0)
        click(30, gui.height - 20)                # Stop (arms)
        time.sleep(0.3)
        click(30, gui.height - 20)                # Stop (confirms)
        assert _wait(lambda: gui.mode == "idle")
        assert "Connection" in gui.summary
        gui.running = False

    _run_gui(tmp_path, script)
    d = only_session(tmp_path)
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta["status"] == "stopped" and meta["config"]["session_type"] == "connection"
    rows = ledger_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["status"].startswith("stopped")


def test_gui_quit_mid_session_still_saves(tmp_path, gui_env):
    def script(gui, click):
        time.sleep(0.5)
        cx, cy = gui.width // 2, gui.height // 2
        click(cx + 230, cy - 100)                 # Sham on
        time.sleep(0.2)
        click(cx, cy + 20)                        # Start Session
        assert _wait(lambda: gui.mode == "session")
        time.sleep(2.5)
        import pygame
        pygame.event.post(pygame.event.Event(pygame.QUIT))  # window closed mid-session

    _run_gui(tmp_path, script)
    d = only_session(tmp_path)
    assert os.path.exists(os.path.join(d, "results.json"))
    rows = ledger_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["condition"] == "sham"
    assert rows[0]["status"].startswith("stopped")


def test_gui_close_display_is_idempotent(gui_env):
    from sine_gui import SineGUI
    g = SineGUI(fullscreen=False)
    g._close_display()
    g._close_display()  # the original bug: AttributeError on _closed
