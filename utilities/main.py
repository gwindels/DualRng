import serial
import numpy as np
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

class CoherenceAnalyzer:
    def __init__(self, gui_mode=False, refresh_mm=0.1):
        self.chunk_size = 32  # Bytes per read
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

    def identify_truerng_port(self):
        """Find the TrueRNG Pro 2 device."""
        import serial.tools.list_ports
        for port in serial.tools.list_ports.comports():
            if "TrueRNG" in port.description:
                return port.device
        raise Exception("TrueRNG Pro 2 not found")

    def bits_to_trials(self, data):
        """Convert raw byte data to 200-bit trial results."""
        if not data:
            return np.array([])
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        if len(bits) < 200: # Previously 500
            return np.array([])
        return np.array([np.sum(chunk) for chunk in np.array_split(bits, len(bits)//200)])
    
    def run_analysis(self, duration_minutes=None):

        if duration_minutes is not None:
            self.total_mm = duration_minutes
        duration_seconds = self.total_mm * 60

        """Reads **unwhitened** data from both RNG cores, calculates coherence, and saves a plot periodically."""
        port = self.identify_truerng_port()
        z_scores_1, z_scores_2, stouf, nvar, svar, timestamps = [], [], [], [], [], []

        try:
            with serial.Serial(port, 115200, timeout=1) as rng:
                rng.write(b'\xC4')  # Enable RNG1 unwhitened mode
                time.sleep(0.1)

                start_time = time.time()
                last_plot_time = start_time
                self._stop_flag = False

                while time.time() - start_time < duration_seconds:
                    if self._stop_flag:
                        print("Trial stopped early.")
                        break
                    try:
                        # Read from RNG1 (unwhitened)
                        rng.write(b'\xC4')
                        data1 = rng.read(self.chunk_size)

                        # Read from RNG2 (unwhitened)
                        rng.write(b'\xC5')
                        data2 = rng.read(self.chunk_size)

                        if not data1 or not data2:
                            print("Warning: Missing data. Retrying...")
                            time.sleep(0.05)
                            continue

                        trials1 = self.bits_to_trials(data1)
                        trials2 = self.bits_to_trials(data2)

                        if trials1.size == 0 or trials2.size == 0:
                            print("Warning: Insufficient data. Skipping iteration.")
                            time.sleep(0.05)
                            continue

                        z1 = self.data_computer.calculate_zscore(trials1, self.chunk_size)
                        z2 = self.data_computer.calculate_zscore(trials2, self.chunk_size)
                        z_scores_1.append(z1)
                        z_scores_2.append(z2)
                        stouffers_z = self.data_computer.compute_stouffers_z(z_scores_1, z_scores_2)
                        netvar = self.netvar_calculator.compute_netvar(stouffers_z[-1])  # Use latest value
                        scaled_netvar = self.netvar_calculator.scale_to_netvar(netvar)
                        nvar.append(netvar)
                        svar.append(scaled_netvar)
                        timestamps.append(datetime.now())

                        # Refresh data based on self.plot_interval
                        if time.time() - last_plot_time > self.plot_interval:
                            category = self.netvar_calculator.categorize_netvar(scaled_netvar)
                            # self.save_plot(z_scores_1, z_scores_2, stouffers_z, timestamps, svar, category)
                            self.update_data(z_scores_1, z_scores_2, stouffers_z, timestamps, svar, category)
                            # self.plotter.plot_correlation(z_scores_1, z_scores_2, stouffers_z, timestamps, svar, category)
                            # self.plotter.plot_histogram(z_scores_1, z_scores_2, stouffers_z, self.start_time)
                            last_plot_time = time.time()

                        time.sleep(0.1)  # Maintain ~1Hz sampling

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

    def update_data(self, z1, z2, stouffers_z, timestamps, netvar, category):
        if len(z1) == 0 or len(z2) == 0: print("No data available, skipping update_data"); return
        
        # Group data into bins for plotting and analysis
        timestamps_binned, smoothed_values = self.data_computer.aggregate_data_by_time(
            timestamps, z1, z2, stouffers_z, netvar, time_window="5s"
        )
        if not timestamps_binned: print("Warning: No valid timestamps after binning."); return
        # Store smoothed values individually
        smoothed_z1, smoothed_z2, smoothed_stouf, smoothed_nv = smoothed_values
        # Ensure timestamps_binned and smoothed_nv have the same length
        min_length = min(len(timestamps_binned), len(smoothed_nv))
        timestamps_binned = timestamps_binned[:min_length]
        smoothed_nv = smoothed_nv[:min_length]
        smoothed_z1 = smoothed_z1[:min_length]
        smoothed_z2 = smoothed_z2[:min_length]
        smoothed_stouf = smoothed_stouf[:min_length]

        # -- Plot netvar and histogram charts --
        self.plotter.plot_netvar(smoothed_z1, smoothed_z2, smoothed_stouf, timestamps_binned, smoothed_nv, category)
        self.plotter.plot_histogram(z1, z2, stouffers_z, self.start_time)
        
        # -- Compute rolling correlation --
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
        p_values = self.data_computer.compute_p_value(z_values)
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

        # -- Track whether the trial as a whole is signfiicant
        Z_global, p_global = self.data_computer.compute_global_trial_significance(valid_corr)
        if Z_global is not None:
            print(f"Global Trial Significance")
            print(f"   Combined Z: {Z_global:.3f}")
            print(f"   p-value: {p_global:.4f}")
            if p_global < 0.05:
                self.global_summary = "Statistically anomalous trial"
            elif p_global < 0.1:
                self.global_summary = "Approaching significance"
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
        self.plotter.plot_rolling_correlation(df, significant_points)
        self.data_saver.save_correlation_values(df)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run script in CLI or GUI mode")
    parser.add_argument("--gui", action="store_true", help="Launch GUI mode")
    parser.add_argument("--duration", type=int, help="Duration to run the analysis in minutes (default: 15)")
    args = parser.parse_args()

    print(f"args.gui = {args.gui}")
    print(f"args.duration = {args.duration}")

    if args.gui:
        import threading

        analyzer = CoherenceAnalyzer(gui_mode=True)
        trial_active = threading.Event()

        def start_new_trial():
            if trial_active.is_set():
                print("Trial already running.")
                return
            trial_active.set()

            def analysis_job():
                analyzer.run_analysis(duration_minutes=args.duration)
                trial_active.clear()
                analyzer.gui.is_idle = True  # Show restart button again
            
            analyzer.reset_trial()
            threading.Thread(target=analysis_job, daemon=True).start()

        analyzer.gui.on_start_trial = start_new_trial
        analyzer.gui.is_idle = True  # Start with the idle screen
        try:
            analyzer.gui.run()
        except KeyboardInterrupt:
            print("Interrupted. Exiting cleanly.")
            analyzer.stop()
        finally:
            analyzer.stop()
    else:
        # Headless mode — no GUI
        analyzer = CoherenceAnalyzer(gui_mode=False)
        analyzer.run_analysis(duration_minutes=args.duration)