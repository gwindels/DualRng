#!/usr/bin/env python3
"""DualRng command line.

    python app.py --gui                      fullscreen GUI (pick session type on screen)
    python app.py run --type intention       headless session (cues printed in the terminal)
    python app.py run --type connection --sham
    python app.py analyze --pending          finalize sessions left without results
    python app.py report                     combined confirmatory report from the ledger
    python app.py scan sessions/<id>         exploratory scan of one session
    python app.py scan --legacy data         exploratory scan of pre-2.0 session CSVs
    python app.py probe                      inspect what the device sends

Add --source sim to any session command to run without hardware.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dualrng.analysis import finalize_pending, finalize_session, read_json, summarize  # noqa: E402
from dualrng import clock  # noqa: E402
from dualrng.live import LiveFeedback  # noqa: E402
from dualrng.protocol import ProtocolConfig  # noqa: E402
from dualrng.session import SessionRunner  # noqa: E402

DEFAULT_BASE = os.path.join(ROOT, "sessions")


def ledger_path(args) -> str:
    return os.path.join(args.base_dir, "ledger.csv")


def build_config(args, session_type: str, sham: bool) -> ProtocolConfig:
    return ProtocolConfig(
        session_type=session_type, sham=sham,
        block_seconds=args.block_seconds, gap_seconds=args.gap_seconds,
        blocks_per_label=args.blocks_per_label or 0, bits_per_read=args.bits_per_read,
    )


def source_spec(args) -> dict:
    if args.source == "sim":
        return {"kind": "sim", "lean": list(args.sim_lean), "intention_delta": args.sim_delta,
                "shared_rho": args.sim_rho, "reads_per_second": args.sim_rate}
    spec = {"kind": "hardware"}
    if args.port:
        spec["port"] = args.port
    return spec


def tally_color(theme, z: float):
    # Same palette as the old NetVar categories, indexed by how unusual the block was.
    defs = theme.get_nvar_def()
    if z > 3.09:
        return defs[4][3]
    if z > 2.33:
        return defs[3][3]
    if z > 1.645:
        return defs[2][3]
    return defs[1][3]


def short_summary(result: dict, session_dir: str) -> str:
    p = result.get("primary")
    kind = f"{result.get('session_type', '').capitalize()} ({result.get('condition')}, {result.get('status')})"
    if not p:
        return f"{kind}: no result\n{result.get('error', '')}"
    return (f"{kind}: p = {p['p']:.3f}, z = {p['z']:+.2f}\n"
            f"Saved to sessions/{os.path.basename(session_dir)}")


# ----------------------------------------------------------------------
# GUI controller
# ----------------------------------------------------------------------

class Controller:
    """Owns the session; the GUI calls tick() every frame and start/stop on taps."""

    def __init__(self, gui, args):
        self.gui = gui
        self.args = args
        self.runner = None
        self.live = None
        self.worker = None
        self._last_poll = 0.0
        self._last_index = None
        self.selection_changed(gui.session_type, gui.sham)

    def selection_changed(self, session_type, sham):
        cfg = build_config(self.args, session_type, sham)
        self.gui.plan_text = (f"{cfg.n_blocks} blocks x {cfg.block_seconds:g} s  "
                              f"({' / '.join(cfg.labels)})  ~{cfg.duration_seconds / 60:.0f} min"
                              + ("  no cues" if sham else ""))

    def start(self, session_type, sham):
        if self.worker and self.worker.is_alive():
            return
        self.gui.mode = "starting"
        cfg = build_config(self.args, session_type, sham)

        def job():
            runner = SessionRunner(cfg, self.args.base_dir, source_spec(self.args), ledger_path(self.args))
            try:
                runner.start()
            except Exception as e:
                if runner.dir:
                    runner.stop(early=True)
                    finalize_session(runner.dir, ledger_path(self.args), make_plots=False)
                self.gui.summary = f"Could not start: {e}"
                self.gui.mode = "idle"
                return
            self.live = LiveFeedback(runner.dir, cfg.session_type, runner.plan)
            self._last_index = None
            self.runner = runner
            self.gui.mode = "session"

        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()

    def _block_value(self, index):
        z = self.live.block_z(index)
        if z is None:
            return None
        return z if self.runner.config.session_type == "intention" else abs(z)

    def tick(self):
        r = self.runner
        if self.gui.mode != "session" or r is None:
            return
        now = clock.now()
        if now - self._last_poll > 0.25:
            self.live.poll()
            self._last_poll = now
        st = r.state(now)
        sham = r.config.sham
        g = self.gui
        n = st.get("n_blocks", len(r.plan))
        if st["phase"] == "done":
            self._close_block()
            self.finish(early=False)
            return
        idx = st["index"]
        if st["phase"] == "block" and self._last_index is not None and idx != self._last_index:
            self._close_block()
        if st["phase"] == "gap" and self._last_index is not None:
            self._close_block()
        g.progress = f"Block {idx + 1}/{n}"
        if sham:
            g.cue, g.cue_dim, g.feedback, g.coherence = "CONTROL", True, "", 0.0
        elif st["phase"] == "gap":
            g.cue, g.cue_dim, g.feedback, g.coherence = f"Next: {st['cue']}", True, "", 0.0
        else:
            self._last_index = idx
            g.cue, g.cue_dim = st["cue"], False
            v = self._block_value(idx)
            raw = self.live.block_z(idx)
            g.feedback = "" if raw is None else f"z {raw:+.2f}"
            g.coherence = 0.0 if v is None else max(0.0, min(1.0, v / 2.5))
        if not r.recorder_ok():
            g.status = "Recorder stopped - see acq_log.csv"
        elif self.live.last_read_mono is not None and now - self.live.last_read_mono > 3:
            g.status = f"No data for {now - self.live.last_read_mono:.0f} s"
        else:
            g.status = ""

    def _close_block(self):
        """Called once when a real block ends: add a tally circle if it went the intended way."""
        if self._last_index is None or self.runner is None:
            return
        idx, self._last_index = self._last_index, None
        if self.runner.config.sham:
            return
        self.live.poll()
        v = self._block_value(idx)
        if v is not None and v > 1.0:
            self.gui.add_tally(tally_color(self.gui.theme, v))

    def stop(self):
        if self.gui.mode == "session":
            self.finish(early=True)

    def finish(self, early):
        runner, self.runner = self.runner, None
        if runner is None:
            return
        self.gui.mode = "finishing"

        def job():
            runner.stop(early=early)
            try:
                result = runner.finalize()
                self.gui.summary = short_summary(result, runner.dir)
            except Exception as e:
                self.gui.summary = f"Analysis failed ({e}); data saved in {runner.dir}"
            self.gui.mode = "idle"

        self.worker = threading.Thread(target=job, daemon=False)
        self.worker.start()

    def shutdown(self):
        """Called when the GUI exits for any reason: never leave a session unsaved."""
        if self.runner is not None:
            try:
                self.gui.draw_message("Saving session...")
                import pygame
                pygame.display.flip()
            except Exception:
                pass
            self.finish(early=True)
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=120)


def cmd_gui(args):
    from sine_gui import SineGUI

    gui = SineGUI(fullscreen=not args.windowed)
    controller = Controller(gui, args)
    gui.controller = controller

    def on_signal(signum, frame):
        gui.running = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def pending():
        for d, res in finalize_pending(args.base_dir, ledger_path(args)):
            print(f"Finalized earlier session {os.path.basename(d)}: {summarize(res)}")
    threading.Thread(target=pending, daemon=True).start()
    gui.run()


# ----------------------------------------------------------------------
# Headless session
# ----------------------------------------------------------------------

def cmd_run(args):
    for d, res in finalize_pending(args.base_dir, ledger_path(args)):
        print(f"Finalized earlier session {os.path.basename(d)}: {summarize(res)}")
    cfg = build_config(args, args.type, args.sham)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda s, f: stop.set())
    signal.signal(signal.SIGTERM, lambda s, f: stop.set())

    runner = SessionRunner(cfg, args.base_dir, source_spec(args), ledger_path(args))
    runner.start()
    live = LiveFeedback(runner.dir, cfg.session_type, runner.plan)
    print(f"Session {runner.session_id}: {cfg.n_blocks} blocks x {cfg.block_seconds:g} s, "
          f"~{cfg.duration_seconds / 60:.1f} min. Ctrl-C stops early (the session is still saved).")
    print(f"Pre-specified statistic: {cfg.primary_statistic} ({cfg.tail})")
    last = None
    early = False
    try:
        while not runner.done():
            if stop.wait(0.2):
                early = True
                break
            live.poll()
            st = runner.state()
            key = (st["phase"], st.get("index"))
            if key != last and st["phase"] in ("gap", "block"):
                i, n = st["index"] + 1, st["n_blocks"]
                if st["phase"] == "block":
                    print(f"[{i}/{n}] " + ("block" if cfg.sham else st["cue"]), flush=True)
                elif last is not None and last[0] == "block" and not cfg.sham:
                    z = live.block_z(last[1])
                    if z is not None:
                        print(f"       block z {z:+.2f}", flush=True)
                last = key
            if not runner.recorder_ok():
                print("Recorder stopped; ending session. See acq_log.csv.")
                break
    finally:
        runner.stop(early=early)
        result = runner.finalize()
        print("\n" + summarize(result))
        ex = os.path.join(runner.dir, "exploratory.json")
        if os.path.exists(ex):
            e = read_json(ex)
            if "p_scan" in e:
                print(f"Exploratory scan (not confirmatory): M {e['M']:.2f}, p_scan {e['p_scan']:.3f}")
        print(f"Saved in {runner.dir}")


# ----------------------------------------------------------------------
# Analysis commands
# ----------------------------------------------------------------------

def cmd_analyze(args):
    if args.pending:
        done = finalize_pending(args.base_dir, ledger_path(args))
        if not done:
            print("No sessions waiting for analysis.")
        for d, res in done:
            print(f"{os.path.basename(d)}: {summarize(res)}")
    for d in args.sessions:
        res = finalize_session(d, ledger_path(args), force=args.force)
        print(f"{os.path.basename(os.path.normpath(d))}: {summarize(res)}")


def cmd_report(args):
    from dualrng.ledger import format_report
    print(format_report(ledger_path(args), complete_only=args.complete_only))


def cmd_scan(args):
    from dualrng.scan import format_legacy, legacy_scan
    if args.legacy:
        res = legacy_scan(args.legacy, n_shifts=args.shifts)
        text = format_legacy(res)
        print(text)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text + "\n")
        return
    from dualrng.analysis import exploratory, load_session_arrays
    for d in args.sessions:
        meta = read_json(os.path.join(d, "meta.json"))
        cfg = ProtocolConfig.from_dict(meta["config"])
        cfg.scan_shifts = args.shifts
        reads, plan = load_session_arrays(d)
        import numpy as np
        rs = float(np.median(reads["t_end"] - reads["t_start"]))
        print(os.path.basename(os.path.normpath(d)))
        print(json.dumps(exploratory(cfg, reads, plan, rs), indent=2))


def cmd_probe(args):
    from dualrng.probe import run_probe
    print(run_probe(os.path.join(ROOT, "probe")))


# ----------------------------------------------------------------------

def add_session_args(p):
    p.add_argument("--block-seconds", type=float, default=30.0)
    p.add_argument("--gap-seconds", type=float, default=3.0)
    p.add_argument("--blocks-per-label", type=int, default=0,
                   help="blocks of each label (default 8 for intention, 12 for connection)")
    p.add_argument("--bits-per-read", type=int, default=1024)
    p.add_argument("--source", choices=("hardware", "sim"), default="hardware")
    p.add_argument("--port", help="serial port (default: first TrueRNG found)")
    p.add_argument("--sim-lean", type=float, nargs=2, default=(0.4992, 0.4992), metavar=("P1", "P2"))
    p.add_argument("--sim-delta", type=float, default=0.0,
                   help="simulated intention effect: added to p(1) in HIGH, subtracted in LOW")
    p.add_argument("--sim-rho", type=float, default=0.0,
                   help="simulated shared signal: correlation of per-read z in CONNECT blocks")
    p.add_argument("--sim-rate", type=float, default=50.0, help="simulated reads per second")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--gui" in argv:  # backwards compatible: python app.py --gui
        argv.remove("--gui")
        argv.insert(0, "gui")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", default=DEFAULT_BASE, help="where sessions and the ledger live")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("gui", help="fullscreen GUI")
    add_session_args(p)
    p.add_argument("--windowed", action="store_true")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("run", help="headless session")
    add_session_args(p)
    p.add_argument("--type", choices=("intention", "connection"), required=True)
    p.add_argument("--sham", action="store_true", help="no cues; control session")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze", help="finalize sessions")
    p.add_argument("sessions", nargs="*")
    p.add_argument("--pending", action="store_true", help="finalize every session without results")
    p.add_argument("--force", action="store_true", help="recompute results.json (the ledger row is not changed)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("report", help="combined confirmatory report")
    p.add_argument("--complete-only", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("scan", help="EXPLORATORY scan")
    p.add_argument("sessions", nargs="*")
    p.add_argument("--legacy", metavar="DIR", help="scan pre-2.0 session_*.csv files in DIR")
    p.add_argument("--shifts", type=int, default=1000)
    p.add_argument("--out", help="also write the legacy table to this file")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("probe", help="inspect the device's raw output")
    p.set_defaults(func=cmd_probe)

    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return 1
    args.base_dir = os.path.abspath(args.base_dir)
    os.makedirs(args.base_dir, exist_ok=True)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
