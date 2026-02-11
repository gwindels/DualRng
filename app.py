import serial
import math
import numpy as np
import signal
import time
import gc  # Garbage collection
import matplotlib
import pandas as pd
from pytz import timezone
import os
import csv
matplotlib.use('Agg')  # Disable UI display (headless mode)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
from datetime import datetime
import argparse
from netvar_calculator import NetVarCalculator
from data_computer import DataComputer
from plotter import Plotter
from data_saver import DataSaver
from sine_gui import SineGUI
from color_theme import ColorTheme
from rng_connector import Connector
from multiscale_analyzer import MultiScaleAnalyzer
from baseline_collector import BaselineCollector

class CoherenceAnalyzer:
    def __init__(self, gui_mode=False, refresh_mm=0.1):
        self.chunk_size = 1024  # Bytes per read
        self.plot_interval = 60 * refresh_mm  # Refresh plot every X minutes
        self.cst = timezone('US/Central')
        self.start_time = datetime.now(self.cst).replace(tzinfo=None)
        self.last_event_time = None
        self.refresh_mm = refresh_mm
        self.total_mm = 15 # Default to 15 minutes
        self.data_dir = "data"
        self.chart_dir = "charts"
        self.plotter = Plotter(self.start_time, self.chart_dir)
        self.data_saver = DataSaver(self.start_time, self.data_dir)
        self.netvar_calculator = NetVarCalculator()
        self.data_computer = DataComputer()
        self.rng = Connector("MODE_UNWHITENED")
        self.multiscale = MultiScaleAnalyzer()
        self.baseline = BaselineCollector.load_baseline()
        self.gui_mode = gui_mode
        if self.gui_mode:
            self.gui = SineGUI()
        # Define NetVar gauge categories and colors
        self.theme = ColorTheme()
        self.color_ranges = self.theme.get_nvar_def()
        self.show_summary = False
        self.global_summary = None
        self.session_data = []
        self._stop_flag = False
        os.chdir(os.path.dirname(os.path.abspath(__file__))) # Set the working directory to the script's location
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.chart_dir, exist_ok=True)

    def run_analysis(self, duration_minutes=None):

        if duration_minutes is not None:
            self.total_mm = duration_minutes
        duration_seconds = self.total_mm * 60

        z_scores_1, z_scores_2, stouffers_z, nvar, svar, timestamps = [], [], [], [], [], []

        try:
            self.rng.open()
            start_time = time.time()
            last_plot_time = start_time
            self._stop_flag = False

            while time.time() - start_time < duration_seconds:
                if self._stop_flag:
                    print("Trial stopped early.")
                    break
                try:
                    bytes_per_ascii_sample = 5 # esimated 
                    ascii_chunk_size = self.chunk_size * bytes_per_ascii_sample * 8  # x8 to convert to bits

                    # Read from RNG1 (unwhitened)
                    data1_raw = self.rng.read(ascii_chunk_size, "rng1")
                    data1 = self.rng.extract_lsb_stream_from_ascii(data1_raw)

                    # Read from RNG2 (unwhitened)
                    data2_raw = self.rng.read(ascii_chunk_size, "rng2")
                    data2 = self.rng.extract_lsb_stream_from_ascii(data2_raw)

                    if not data1 or not data2:
                        print("Warning: Missing data. Retrying...")
                        time.sleep(0.05)
                        continue

                    z1 = self.rng.lsb_z_score(data1, chunk_size=self.chunk_size * 8) # x8 to convert bytes to bits
                    z2 = self.rng.lsb_z_score(data2, chunk_size=self.chunk_size * 8)
                    z_scores_1.append(z1)
                    z_scores_2.append(z2)
                    stouffers_z.append((z1 + z2) / math.sqrt(2))
                    netvar = self.netvar_calculator.compute_netvar(stouffers_z[-1])
                    scaled_netvar = self.netvar_calculator.scale_to_netvar(netvar)
                    nvar.append(netvar)
                    svar.append(scaled_netvar)
                    timestamps.append(datetime.now())

                    # Refresh data based on self.plot_interval
                    if time.time() - last_plot_time > self.plot_interval:
                        category = self.netvar_calculator.categorize_netvar(scaled_netvar)
                        # self.save_plot(z_scores_1, z_scores_2, stouffers_z, timestamps, svar, category)
                        self.update_data(z_scores_1, z_scores_2, stouffers_z, timestamps, svar, nvar, category)
                        # self.plotter.plot_correlation(z_scores_1, z_scores_2, stouffers_z, timestamps, svar, category)
                        # self.plotter.plot_histogram(z_scores_1, z_scores_2, stouffers_z, self.start_time)
                        last_plot_time = time.time()

                    time.sleep(0.05)

                except KeyboardInterrupt:
                    print("\nProcess interrupted. Closing connection.")
                    break  # Exit loop on Ctrl+C
            
            # Collect, combine, and save all current session data
            self.session_data = [{
                'timestamps': timestamps[i], 
                'z_scores_1': z_scores_1[i], 
                'z_scores_2': z_scores_2[i], 
                'stouffers_z': stouffers_z[i], 
                'nvar': nvar[i],
                'svar': svar[i]
            } for i in range(len(z_scores_1))] # merge arrays into session_data
            self.data_saver.save_session_data(self.session_data) # save to csv

        except Exception as e:
            print(f"Analysis Error: {e}")
        finally:
            self.rng.close()

        return np.array(z_scores_1), np.array(z_scores_2), timestamps, np.array(nvar)

    def stop(self):
        self._stop_flag = True

    def label_coherence_strength(self, r):
        abs_r = abs(r)
        if abs_r < 0.1:
            return "Negligible"
        elif abs_r < 0.3:
            return "Mild"
        elif abs_r < 0.5:
            return "Moderate"
        elif abs_r < 0.7:
            return "Strong"
        elif abs_r < 0.9:
            return "Very Strong"
        else:
            return "Near Perfect"

    def update_data(self, z1, z2, stouffers_z, timestamps, netvar, nvar_raw, category):
        if len(z1) == 0 or len(z2) == 0: print("No data available, skipping update_data"); return

        # Group data into bins for plotting and analysis
        # nvar_raw (per-sample Z²) is binned alongside so mean(Z²) preserves E=1 under H0
        timestamps_binned, smoothed_values = self.data_computer.aggregate_data_by_time(
            timestamps, z1, z2, stouffers_z, netvar, nvar_raw, time_window="2s"
        )
        if not timestamps_binned: print("Warning: No valid timestamps after binning."); return
        # Store smoothed values individually
        smoothed_z1, smoothed_z2, smoothed_stouf, smoothed_nv, smoothed_nv_raw = smoothed_values
        # Ensure timestamps_binned and smoothed_nv have the same length
        min_length = min(len(timestamps_binned), len(smoothed_nv))
        timestamps_binned = timestamps_binned[:min_length]
        smoothed_nv = smoothed_nv[:min_length]
        smoothed_nv_raw = smoothed_nv_raw[:min_length]
        smoothed_z1 = smoothed_z1[:min_length]
        smoothed_z2 = smoothed_z2[:min_length]
        smoothed_stouf = smoothed_stouf[:min_length]
        # Calculate rolling values
        stouf_rolling = pd.Series(smoothed_stouf, index=timestamps_binned).rolling(window=5, min_periods=1).mean()

        # -- Run multi-scale analysis --
        multiscale_results = self.multiscale.analyze(timestamps_binned, smoothed_z1, smoothed_z2)
        netvar_multiscale_results = self.multiscale.analyze_netvar(timestamps_binned, smoothed_nv_raw)

        # -- Plot netvar and histogram charts --
        self.plotter.plot_netvar(smoothed_z1, smoothed_z2, smoothed_stouf, stouf_rolling, timestamps_binned, smoothed_nv, category)
        self.plotter.plot_multiscale_netvar_scaled(timestamps_binned, smoothed_nv, self.multiscale.window_scales)
        # self.plotter.plot_histogram(z1, z2, stouffers_z, self.start_time)

        # -- Compute rolling correlation (primary 10-sample window) --
        df = pd.DataFrame({"timestamp": timestamps_binned, "z1": smoothed_z1, "z2": smoothed_z2})
        df.set_index("timestamp", inplace=True)
        df["rolling_corr"] = df["z1"].rolling(10, min_periods=1).corr(df["z2"])  # Rolling correlation
        df["rolling_corr_ma"] = df["rolling_corr"].rolling(20).mean() # Moving Average
        valid_corr = df["rolling_corr"].dropna()
        valid_corr = valid_corr[(valid_corr > -1) & (valid_corr < 1)]  # Ensure valid range
        
        # Only continue with correlation if there are enough values
        if valid_corr.empty: 
            # -- Only update nvar on screen
            if self.gui_mode:
                self.gui.set_coherence(0, smoothed_nv[-1])
            # Don't go further
            print("No valid correlation values after filtering.")
            return

        valid_corr = valid_corr.clip(-0.9999, 0.9999)  # Prevent division by zero
        z_values = 0.5 * np.log((1 + valid_corr) / (1 - valid_corr))
        p_values = self.data_computer.compute_p_value(z_values, window_size=10)
        effect_sizes = np.abs(z_values)
        # Store latest correlation value for display
        last_value = valid_corr.iloc[-1]

        # -- Identify significant events --
        # Ensure `significance_threshold` is a Pandas Series aligned with `df.index`
        significance_threshold = pd.Series((p_values < 0.05) & (effect_sizes > 0.5), index=valid_corr.index)
        # Reindex `significance_threshold` to match `df.index`, filling missing values with False
        significance_threshold = significance_threshold.reindex(df.index, fill_value=False)
        # Correctly filter `df` using the aligned boolean mask
        significant_points = df[significance_threshold]
        # Convert P values and Effect Sizes to Panda series to support loc
        p_values = pd.Series(p_values, index=valid_corr.index)
        effect_sizes = pd.Series(effect_sizes, index=valid_corr.index)
        # Define minimum logging time threshold
        min_log_time = pd.Timestamp(self.start_time).tz_localize(None) + pd.Timedelta(seconds=10)
        # Filter points down to 10 second or later to avoid early noise
        filtered_points = significant_points[significant_points.index > min_log_time]
        should_announce = False
        if self.last_event_time is not None:
            last_ts = pd.Timestamp(self.last_event_time).tz_localize(None)
            new_events = filtered_points[filtered_points.index > last_ts]
            if not new_events.empty:
                should_announce = True
                most_recent = new_events.index[-1]
        else:
            # First run — consider the latest filtered event new
            if not filtered_points.empty:
                should_announce = True
                most_recent = filtered_points.index[-1]
        # Store events in a list
        significant_events = [
            {
                "timestamp": index,
                "correlation": row["rolling_corr"],
                "p_value": p_values.loc[index],
                "effect_size": effect_sizes.loc[index]
            }
            for index, row in filtered_points.iterrows()
        ]
        if significant_events:
            # Save to CSV
            self.data_saver.save_significant_events(significant_events)
            if should_announce:
                print(f"New significant event at {most_recent}")
                # if self.gui_mode:
                    # self.gui.trigger_flash()
                self.last_event_time = pd.Timestamp(most_recent).tz_localize(None)

        # -- Track whether the trial as a whole is significant
        p_empirical = None
        Z_global, p_global = self.data_computer.compute_global_trial_significance(valid_corr)
        if Z_global is not None:
            # Empirical p-value if baseline is available
            p_empirical = None
            if self.baseline is not None and "corr_20s" in self.baseline:
                p_empirical = self.data_computer.bootstrap_trial_p_value(
                    Z_global, self.baseline["corr_20s"], len(valid_corr)
                )

            # Bayes Factor
            bf10 = self.data_computer.bayes_factor_z(Z_global, len(valid_corr))

            # Correlation confidence interval
            mean_corr = valid_corr.mean()
            ci_lower, ci_upper = self.data_computer.correlation_confidence_interval(
                mean_corr, len(valid_corr)
            )

            # Multi-scale significance summary
            scale_summary = self.multiscale.get_scale_summary()
            sig_scales = [name for name, s in scale_summary.items() if s["n_significant"] > 0]
            strongest = self.multiscale.get_strongest_event()

            # FDR-corrected event count
            all_ms_events = self.multiscale.get_significant_events()
            n_fdr_events = len(all_ms_events)

            # Enhanced console summary
            elapsed = (timestamps_binned[-1] - timestamps_binned[0]).total_seconds() / 60 if len(timestamps_binned) > 1 else 0
            print(f"\nTrial Summary ({elapsed:.0f} min)")
            print(f"   Global Z: {Z_global:.3f}, p: {p_global:.4f} (theoretical)", end="")
            if p_empirical is not None:
                print(f", p: {p_empirical:.4f} (bootstrap)", end="")
            print()
            print(f"   Bayes Factor (BF10): {bf10:.2f}")
            print(f"   Mean correlation: {mean_corr:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
            if sig_scales:
                scale_details = []
                for name in sig_scales:
                    s = scale_summary[name]
                    scale_details.append(f"{name} ({s['n_significant']} events, min p={s['min_p_value']:.4f})")
                print(f"   Significant scales: {', '.join(scale_details)}")
            else:
                print(f"   Significant scales: none")
            if strongest:
                print(f"   Strongest event: {strongest['timestamp']} at {strongest['scale']} scale "
                      f"(r={strongest['correlation']:.4f}, p={strongest['p_value']:.6f})")
            print(f"   Events: {n_fdr_events} survived FDR correction")

            if p_global < 0.05:
                self.global_summary = f"Statistically significant {self.total_mm} minute trial"
            elif p_global < 0.1:
                self.global_summary = f"{self.total_mm} minute trial approached significance"
            else:
                self.global_summary = f"{self.total_mm} minute trial complete"
            print(f"   {self.global_summary}.")

        # Check whether rolling average is available
        rolling_ma = df["rolling_corr_ma"].dropna()
        if rolling_ma.empty: 
            if self.gui_mode:
                # -- Only update nvar and raw_coherence
                self.gui.set_coherence(0, nvar=smoothed_nv[-1], raw_coherence=last_value)
            # Don't go further
            print("Not enough data for rolling avg.")
            return
        
        r_ma = rolling_ma.iloc[-1]
        z_ma = self.data_computer.fisher_z_transform(r_ma)
        p_ma = self.data_computer.compute_p_value(z_ma)
        coherence_strength = round(min(1.0, abs(r_ma)), 3)
        coherence_label = self.label_coherence_strength(r_ma)

        print(f"Rolling correlation MA (r): {round(r_ma, 4)}")
        print(f"   z: {round(z_ma, 4)}")
        print(f"   p-value: {round(p_ma, 4)}")
        print(f"   Significance: {coherence_label}")
        print(f"   Coherence strength: {coherence_strength}")

        # -- Update the GUI
        if self.gui_mode:
            self.gui.set_coherence((coherence_strength * 2), nvar=smoothed_nv[-1], raw_coherence=last_value)
            self.gui.set_z_global(self.global_summary)

        # -- Plot rolling correlation and save raw data to CSV --
        trial_stats = {
            "scale_summary": self.multiscale.get_scale_summary(),
            "bf10": self.data_computer.bayes_factor_z(
                Z_global, len(valid_corr)
            ) if Z_global is not None else None,
            "p_empirical": p_empirical if Z_global is not None else None,
        }
        self.plotter.plot_rolling_correlation(
            df, significant_points, self.global_summary,
            baseline=self.baseline, trial_stats=trial_stats
        )
        self.data_saver.save_correlation_values(df)

        # -- Plot multi-scale correlation and save events --
        if multiscale_results:
            self.plotter.plot_multiscale_correlation(multiscale_results)
            ms_events = self.multiscale.get_significant_events()
            if ms_events:
                self.data_saver.save_multiscale_events(ms_events)

        # -- Plot multi-scale NetVar --
        if netvar_multiscale_results:
            self.plotter.plot_multiscale_netvar(netvar_multiscale_results)
            nv_scale_summary = self.multiscale.get_netvar_scale_summary()
            nv_sig_scales = [name for name, s in nv_scale_summary.items() if s["n_significant"] > 0]
            if nv_sig_scales:
                nv_details = []
                for name in nv_sig_scales:
                    s = nv_scale_summary[name]
                    nv_details.append(f"{name} ({s['n_significant']} events, min p={s['min_p_value']:.4f})")
                print(f"   NetVar significant scales: {', '.join(nv_details)}")
            else:
                print(f"   NetVar significant scales: none")

    def reset_trial(self):
        print("Resetting analyzer state for new trial...")
        self.start_time = datetime.now(self.cst).replace(tzinfo=None)
        self.last_event_time = None
        self.session_data = []
        self.global_summary = None
        self.plotter.reset(self.start_time)
        self.data_saver.reset(self.start_time)
        self.netvar_calculator.reset()

        if self.gui_mode:
            self.gui.reset()


# --- Helpers to cleanly manage lifecycle of application ---

def graceful_shutdown(analyzer, analysis_thread=None):
    """Stop analysis, join the worker, close GUI, close plots."""
    try:
        analyzer.stop()  # sets _stop_flag so long loops exit
        if analysis_thread and analysis_thread.is_alive():
            analysis_thread.join(timeout=5.0)
    finally:
        # If GUI is being used, stop it (this calls SineGUI._close_display())
        if getattr(analyzer, "gui_mode", False) and hasattr(analyzer, "gui"):
            analyzer.gui.stop()
        # Close any open figures
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass
        # Optional: if your Connector has a close(), do it
        if hasattr(analyzer, "rng") and hasattr(analyzer.rng, "close"):
            try:
                analyzer.rng.close()
            except Exception:
                pass

def _signal_handler(signum, frame, analyzer_ref=None, analysis_thread_ref=None):
    if analyzer_ref:
        self.graceful_shutdown(analyzer_ref, analysis_thread_ref)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run script in CLI or GUI mode")
    parser.add_argument("--gui", action="store_true", help="Launch GUI mode")
    parser.add_argument("--duration", type=int, help="Duration to run the analysis in minutes (default: 15)")
    parser.add_argument("--baseline", action="store_true", help="Run in baseline collection mode (no GUI)")
    parser.add_argument("--baseline-sessions", type=int, default=5, help="Number of baseline sessions to collect (default: 5)")
    args = parser.parse_args()

    if args.baseline:
        collector = BaselineCollector()
        duration = args.duration if args.duration else 5
        collector.collect(duration_minutes=duration, num_sessions=args.baseline_sessions)
        exit(0)

    print(f"args.gui = {args.gui}")
    print(f"args.duration = {args.duration}")

    if args.gui:
        import threading

        analyzer = CoherenceAnalyzer(gui_mode=True)
        trial_active = threading.Event()
        analysis_thread = None  # track the current analysis run thread

        def start_new_trial():
            global analysis_thread
            if trial_active.is_set():
                print("Trial already running.")
                return
            trial_active.set()

            def analysis_job():
                try:
                    analyzer.run_analysis(duration_minutes=analyzer.gui.duration_minutes)
                finally:
                    trial_active.clear()
                    analyzer.gui.is_idle = True  # back to idle when done

            analyzer.reset_trial()
            analysis_thread = threading.Thread(target=analysis_job, daemon=False)  # not daemon → let us join
            analysis_thread.start()

        analyzer.gui.on_start_trial = start_new_trial
        analyzer.gui.is_idle = True  # Start with the idle screen

        if args.duration:
            analyzer.gui.duration_minutes = args.duration
            analyzer.gui.duration_text = str(args.duration)

        # Hook signals now that we have objects
        signal.signal(signal.SIGINT,  lambda s, f: _signal_handler(s, f, analyzer, analysis_thread))
        signal.signal(signal.SIGTERM, lambda s, f: _signal_handler(s, f, analyzer, analysis_thread))

        try:
            # Run the GUI loop in the main thread so OS signals are delivered there
            analyzer.gui.run()
        except KeyboardInterrupt:
            print("Interrupted. Exiting cleanly.")
        finally:
            graceful_shutdown(analyzer, analysis_thread)

    else:
        # Headless mode — no GUI
        analyzer = CoherenceAnalyzer(gui_mode=False)

        # Hook signals to stop long loops
        signal.signal(signal.SIGINT,  lambda s, f: _signal_handler(s, f, analyzer, None))
        signal.signal(signal.SIGTERM, lambda s, f: _signal_handler(s, f, analyzer, None))

        try:
            analyzer.run_analysis(duration_minutes=args.duration)
        except KeyboardInterrupt:
            print("Interrupted. Exiting cleanly.")
        finally:
            graceful_shutdown(analyzer, None)
