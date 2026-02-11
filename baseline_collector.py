import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pytz import timezone
from rng_connector import Connector
from data_computer import DataComputer
from netvar_calculator import NetVarCalculator
from multiscale_analyzer import MultiScaleAnalyzer


class BaselineCollector:
    """Collects baseline data from the RNG hardware to build empirical null distributions.

    Runs the same data collection pipeline as a normal trial but stores results
    in a separate baseline/ directory. Builds empirical CDFs for correlation,
    Stouffer's Z, NetVar, and event rates at each window scale.
    """

    def __init__(self, baseline_dir="baseline"):
        self.baseline_dir = baseline_dir
        self.cst = timezone('US/Central')
        self.data_computer = DataComputer()
        self.netvar_calculator = NetVarCalculator()
        self.chunk_size = 1024
        os.makedirs(self.baseline_dir, exist_ok=True)

    def collect(self, duration_minutes=5, num_sessions=5):
        """Run N baseline sessions, each collecting data for the given duration.

        Args:
            duration_minutes: Length of each baseline session in minutes.
            num_sessions: Number of sessions to collect.
        """
        rng = Connector("MODE_UNWHITENED")
        try:
            rng.open()
            for session in range(num_sessions):
                print(f"\n=== Baseline session {session + 1}/{num_sessions} ===")
                session_data = self._run_session(rng, duration_minutes)
                if session_data is not None:
                    self._save_session(session_data, session + 1)
                print(f"Session {session + 1} complete.")
        finally:
            rng.close()

        print(f"\nAll {num_sessions} baseline sessions complete.")
        print("Building null distributions...")
        self.build_null_distributions()

    def _run_session(self, rng, duration_minutes):
        """Run a single baseline session (same pipeline as CoherenceAnalyzer.run_analysis)."""
        duration_seconds = duration_minutes * 60
        z_scores_1, z_scores_2, stouf, nvar, timestamps = [], [], [], [], []
        bytes_per_ascii_sample = 5
        ascii_chunk_size = self.chunk_size * bytes_per_ascii_sample * 8

        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            try:
                data1_raw = rng.read(ascii_chunk_size, "rng1")
                data1 = rng.extract_lsb_stream_from_ascii(data1_raw)
                data2_raw = rng.read(ascii_chunk_size, "rng2")
                data2 = rng.extract_lsb_stream_from_ascii(data2_raw)

                if not data1 or not data2:
                    time.sleep(0.05)
                    continue

                z1 = rng.lsb_z_score(data1, chunk_size=self.chunk_size * 8)
                z2 = rng.lsb_z_score(data2, chunk_size=self.chunk_size * 8)
                z_scores_1.append(z1)
                z_scores_2.append(z2)

                stouffers_z = self.data_computer.compute_stouffers_z(z_scores_1, z_scores_2)
                netvar_val = self.netvar_calculator.compute_netvar(stouffers_z[-1])
                stouf.append(stouffers_z[-1])
                nvar.append(netvar_val)
                timestamps.append(datetime.now())
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("Session interrupted.")
                break
            except Exception as e:
                print(f"Read error: {e}")
                time.sleep(0.05)

        if not timestamps:
            return None

        # Bin the data at 5s intervals
        timestamps_binned, smoothed_values = self.data_computer.aggregate_data_by_time(
            timestamps, z_scores_1, z_scores_2, stouf, nvar, time_window="5s"
        )
        if not timestamps_binned:
            return None

        smoothed_z1, smoothed_z2, smoothed_stouf, smoothed_nv = smoothed_values
        min_len = min(len(timestamps_binned), len(smoothed_z1))

        # Run multi-scale analysis
        analyzer = MultiScaleAnalyzer()
        multiscale_results = analyzer.analyze(
            timestamps_binned[:min_len],
            smoothed_z1[:min_len],
            smoothed_z2[:min_len]
        )

        # Collect per-scale correlation distributions
        per_scale_corr = {}
        for scale_name, data in multiscale_results.items():
            per_scale_corr[scale_name] = data["correlations"].values

        return {
            "timestamps": timestamps_binned[:min_len],
            "z1": smoothed_z1[:min_len],
            "z2": smoothed_z2[:min_len],
            "stouffers_z": smoothed_stouf[:min_len],
            "netvar": smoothed_nv[:min_len],
            "per_scale_corr": per_scale_corr,
            "significant_events": analyzer.get_significant_events(),
            "duration_minutes": (time.time() - (time.time() - len(timestamps) * 0.1)) / 60,
        }

    def _save_session(self, session_data, session_num):
        """Save a single baseline session to CSV."""
        ts_str = datetime.now(self.cst).strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.baseline_dir, f"baseline_session_{session_num}_{ts_str}.csv")
        df = pd.DataFrame({
            "timestamp": session_data["timestamps"],
            "z1": session_data["z1"],
            "z2": session_data["z2"],
            "stouffers_z": session_data["stouffers_z"],
            "netvar": session_data["netvar"],
        })
        df.to_csv(filename, index=False)
        print(f"  Saved: {filename}")

    def build_null_distributions(self):
        """Build empirical null distributions from all saved baseline sessions.

        Reads all baseline session CSVs and computes empirical CDFs for:
        - Per-scale rolling correlation distributions
        - Stouffer's Z distributions
        - NetVar distributions
        - Significant event rate (events per minute)

        Saves to baseline/null_distributions.npz
        """
        session_files = sorted([
            f for f in os.listdir(self.baseline_dir)
            if f.startswith("baseline_session_") and f.endswith(".csv")
        ])
        if not session_files:
            print("No baseline sessions found.")
            return

        all_stouffers = []
        all_netvar = []
        all_per_scale_corr = {}
        event_rates = []

        for fname in session_files:
            filepath = os.path.join(self.baseline_dir, fname)
            df = pd.read_csv(filepath, parse_dates=["timestamp"])
            all_stouffers.extend(df["stouffers_z"].dropna().values)
            all_netvar.extend(df["netvar"].dropna().values)

            # Re-run multi-scale analysis to get per-scale correlations
            if len(df) >= 3:
                analyzer = MultiScaleAnalyzer()
                results = analyzer.analyze(
                    df["timestamp"].tolist(),
                    df["z1"].values,
                    df["z2"].values
                )
                for scale_name, data in results.items():
                    if scale_name not in all_per_scale_corr:
                        all_per_scale_corr[scale_name] = []
                    all_per_scale_corr[scale_name].extend(data["correlations"].values)

                # Event rate: significant events per minute
                n_events = len(analyzer.get_significant_events())
                duration_min = max(1, len(df)) * 5 / 60  # approx minutes from 5s bins
                event_rates.append(n_events / duration_min)

        # Save null distributions
        save_dict = {
            "stouffers_z": np.array(all_stouffers),
            "netvar": np.array(all_netvar),
            "event_rates": np.array(event_rates),
        }
        for scale_name, corr_vals in all_per_scale_corr.items():
            save_dict[f"corr_{scale_name}"] = np.array(corr_vals)

        outpath = os.path.join(self.baseline_dir, "null_distributions.npz")
        np.savez(outpath, **save_dict)
        print(f"Null distributions saved to {outpath}")
        print(f"  Stouffer's Z: {len(all_stouffers)} values")
        print(f"  NetVar: {len(all_netvar)} values")
        print(f"  Event rates: {len(event_rates)} sessions")
        for scale_name, vals in all_per_scale_corr.items():
            print(f"  {scale_name} correlations: {len(vals)} values")

    @staticmethod
    def load_baseline(baseline_dir="baseline"):
        """Load saved null distributions from disk.

        Returns:
            dict with keys: stouffers_z, netvar, event_rates, corr_<scale_name>
            Returns None if no baseline file exists.
        """
        filepath = os.path.join(baseline_dir, "null_distributions.npz")
        if not os.path.exists(filepath):
            return None
        data = np.load(filepath, allow_pickle=True)
        return dict(data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect baseline RNG data for null distributions")
    parser.add_argument("--duration", type=int, default=5,
                        help="Duration per session in minutes (default: 5)")
    parser.add_argument("--sessions", type=int, default=5,
                        help="Number of baseline sessions to collect (default: 5)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild null distributions from existing session files without collecting new data")
    args = parser.parse_args()

    collector = BaselineCollector()

    if args.rebuild:
        collector.build_null_distributions()
    else:
        collector.collect(duration_minutes=args.duration, num_sessions=args.sessions)
