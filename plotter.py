import gc
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
from data_computer import DataComputer
from color_theme import ColorTheme

class Plotter:
    def __init__(self, start_time, chart_dir, update_score_gui=False):
        self.start_time = start_time
        self.chart_dir = chart_dir
        self.update_score_gui = update_score_gui
        self.data_computer = DataComputer()
        self.theme = ColorTheme()
        self.color_ranges = self.theme.get_nvar_def()
        self.archive_existing_charts()
        # Locator/formatter created fresh per-axis via helper to avoid
        # sharing state and to include second-level ticks for short ranges.
        self._fmt_str = "%I:%M:%S %p"

    def _apply_date_axis(self, ax):
        """Configure a date x-axis with a fresh locator that handles short ranges."""
        locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
        locator.intervald[mdates.SECONDLY] = [1, 5, 10, 15, 30]
        locator.intervald[mdates.MINUTELY] = [1, 2, 5, 10, 15, 30]
        locator.intervald[mdates.HOURLY] = [1, 2, 3, 6]
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(self._fmt_str))

    def archive_existing_charts(self):
        charts_dir = "charts"
        saved_dir = os.path.join(charts_dir, "saved")

        os.makedirs(saved_dir, exist_ok=True)

        for filename in os.listdir(charts_dir):
            file_path = os.path.join(charts_dir, filename)

            if os.path.isdir(file_path) or filename == "saved":
                continue

            dest_path = os.path.join(saved_dir, filename)
            shutil.move(file_path, dest_path)
            print(f"Moved {file_path} -> {dest_path}")


    def plot_netvar(self, z1, z2, stouffers_z, stouffers_z_rolling, timestamps, netvar, category):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # ==== Chart 1 ====
        # Plot Z scores
        if len(timestamps) == len(z1):
            ax1.plot(timestamps, z1, label='RNG1 (Unwhitened)', alpha=0.7)
            ax1.plot(timestamps, z2, label='RNG2 (Unwhitened)', alpha=0.7)
            ax1.plot(timestamps, stouffers_z, label='Stouffer\'s Z', alpha=0.7)
            ax1.plot(timestamps, stouffers_z_rolling, 'k--', linewidth=2, label="Stouffer\'s Rolling Avg (10s)")

            # Add anomalous plots
            threshold = 2.5 # Assumed significance
            for i, value in enumerate(stouffers_z):
                if abs(value) > threshold:  # Only highlight extreme anomalies
                    ax1.scatter(timestamps[i], value, color="#00945b", s=100, label="_nolegend_")
                    ax1.annotate(
                        f"{value:.1f}",
                        (timestamps[i], value),
                        textcoords="offset points",
                        xytext=(5,5),
                        ha='center', fontsize=10, color="#006b42"
                    )
        # Configure chart
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_title('Z-scores over time')
        ax1.set_ylim(-3.25, 3.25)
        self._apply_date_axis(ax1)
        ax1.legend()

        # ==== Chart 2 ====
        # Plot smoothed NetVar
        if len(timestamps) == len(netvar) and len(netvar) > 0:
            ax2.plot(timestamps, netvar, label=f'NetVar: {netvar[-1]:.2f} ({category})', color="#8a8a8a")
        else:
            print(f"Warning: Timestamp length mismatch (timestamps={len(timestamps)}, NetVar={len(netvar)})")

        # Plot extreme anomalies for raw netvar
        for i, value in enumerate(netvar):
            for lower, upper, color, rgb, label in self.color_ranges:
                if lower <= value < upper:
                    if label in ["Very High", "Extreme"]:  # Only highlight extreme anomalies
                        ax2.scatter(timestamps[i], value, color=color, s=100, label="_nolegend_")

        # Add coherence thresholds
        for lower, upper, color, rgb, label in self.color_ranges:
            ax2.axhline(y=lower, color=color, linestyle="--", linewidth=1, label=label)
        # Configure chart
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_title('NetVar')
        self._apply_date_axis(ax2)
        if len(ax2.lines) > 0:  # Ensure at least one plotted line
            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend(handles[::-1], labels[::-1], loc='upper left',
                    fontsize='small', frameon=True, facecolor='white', framealpha=0.8)
        else:
            print("Warning: No plotted data for NetVar legend.")
        plt.tight_layout()

        start_time_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        file_name = os.path.join(self.chart_dir, f"plot_netvar_{start_time_str}.png")
        plt.savefig(file_name)
        plt.close(fig)  # Close the figure to free memory
        gc.collect()  # Force garbage collection

    def plot_rolling_correlation(self, df, significant_points, global_summary, baseline=None, trial_stats=None):
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot rolling correlation, rolling average, and significant events
        ax.plot(df.index, df["rolling_corr"], color='b', alpha=0.5, label='Rolling Correlation (20s)')
        ax.plot(df.index, df["rolling_corr_ma"], color='r', linestyle="--", label='Moving Average Trendline (40s)')
        if not significant_points.empty:
            ax.scatter(significant_points.index, significant_points["rolling_corr"],
                color='red', label='Significant (p < 0.05, Z > 0.5)', zorder=3)

        # Baseline confidence band (95th percentile envelope)
        if baseline is not None and "corr_20s" in baseline:
            null_corr = baseline["corr_20s"]
            p95 = np.percentile(null_corr, 97.5)
            p05 = np.percentile(null_corr, 2.5)
            ax.axhspan(p05, p95, alpha=0.1, color='blue', label='Baseline 95% CI')

        ax.axhline(y=0.6, color='green', linestyle="dashed", linewidth=1.5, label="High Positive Corr (0.6)")
        ax.axhline(y=-0.6, color='#684736', linestyle="dashed", linewidth=1.5, label="High Negative Corr (-0.6)")
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Rolling Correlation Between RNG1 & RNG2 (Significance & Effect Size)')
        self._apply_date_axis(ax)
        ax.legend(fontsize='small')

        # Build summary text with trial stats if available
        summary_lines = [global_summary] if global_summary else []
        if trial_stats:
            if "scale_summary" in trial_stats:
                sig_scales = [name for name, s in trial_stats["scale_summary"].items()
                              if s["n_significant"] > 0]
                if sig_scales:
                    summary_lines.append(f"Sig. scales: {', '.join(sig_scales)}")
            if "bf10" in trial_stats:
                summary_lines.append(f"BF10: {trial_stats['bf10']:.2f}")
            if "p_empirical" in trial_stats and trial_stats["p_empirical"] is not None:
                summary_lines.append(f"p(boot): {trial_stats['p_empirical']:.4f}")
        summary_text = "  |  ".join(summary_lines) if summary_lines else ""
        fig.text(0.5, 0.02, summary_text, ha='center', va='bottom', fontsize=9)

        start_time_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.chart_dir, f"plot_corr_{start_time_str}.png")
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.savefig(filename)
        plt.close(fig)  # Close the figure to free memory
        gc.collect()  # Force garbage collection
        
    def plot_multiscale_correlation(self, multiscale_results):
        """Plot rolling correlations at multiple window scales with significance markers."""
        n_scales = len(multiscale_results)
        if n_scales == 0:
            return

        fig, axes = plt.subplots(n_scales, 1, figsize=(12, 3 * n_scales), sharex=True)
        if n_scales == 1:
            axes = [axes]

        scale_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, (scale_name, data) in enumerate(multiscale_results.items()):
            ax = axes[i]
            corr = data["correlations"]
            sig = data["significant"]

            ax.plot(corr.index, corr.values, color=scale_colors[i % len(scale_colors)],
                    alpha=0.7, label=f"{scale_name} correlation")
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0.6, color='green', linestyle=':', alpha=0.4)
            ax.axhline(y=-0.6, color='green', linestyle=':', alpha=0.4)

            # Mark FDR-significant points
            sig_times = sig[sig].index
            if len(sig_times) > 0:
                sig_corr = corr.loc[sig_times]
                ax.scatter(sig_times, sig_corr, color='red', s=40, zorder=3,
                          label=f"Significant (FDR < 0.05)")

            ax.set_ylabel(scale_name)
            ax.set_ylim(-1.05, 1.05)
            ax.legend(loc='upper right', fontsize='small')
            self._apply_date_axis(ax)

        axes[-1].set_xlabel("Time")
        fig.suptitle("Multi-Scale Rolling Correlation (FDR-Corrected)", fontsize=14)
        plt.tight_layout()

        start_time_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.chart_dir, f"plot_multiscale_{start_time_str}.png")
        plt.savefig(filename)
        plt.close(fig)
        gc.collect()

    def plot_multiscale_netvar(self, netvar_results):
        """Plot rolling mean of raw NetVar at multiple window scales with FDR significance markers."""
        n_scales = len(netvar_results)
        if n_scales == 0:
            return

        fig, axes = plt.subplots(n_scales, 1, figsize=(12, 3 * n_scales), sharex=True)
        if n_scales == 1:
            axes = [axes]

        scale_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for i, (scale_name, data) in enumerate(netvar_results.items()):
            ax = axes[i]
            rolling_mean = data["rolling_mean"]
            sig = data["significant"]

            ax.plot(rolling_mean.index, rolling_mean.values,
                    color=scale_colors[i % len(scale_colors)],
                    alpha=0.7, label=f"{scale_name} mean NetVar")
            ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label="H0 expectation")

            # Mark FDR-significant points
            sig_times = sig[sig].index
            if len(sig_times) > 0:
                sig_vals = rolling_mean.loc[sig_times]
                ax.scatter(sig_times, sig_vals, color='red', s=40, zorder=3,
                          label="Significant (FDR < 0.05)")

            ax.set_ylabel(scale_name)
            ax.legend(loc='upper right', fontsize='small')
            self._apply_date_axis(ax)

        axes[-1].set_xlabel("Time")
        fig.suptitle("Multi-Scale NetVar (FDR-Corrected)", fontsize=14)
        plt.tight_layout()

        start_time_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.chart_dir, f"plot_multiscale_netvar_{start_time_str}.png")
        plt.savefig(filename)
        plt.close(fig)
        gc.collect()

    def plot_multiscale_netvar_scaled(self, timestamps, scaled_netvar, window_scales):
        """Plot rolling mean of scaled NetVar (1-324) at multiple window scales."""
        series = pd.Series(scaled_netvar, index=timestamps)
        n_points = len(scaled_netvar)

        # Build rolling means per scale (skip scales larger than data)
        scale_data = {}
        for scale_name, window_size in window_scales.items():
            if n_points < window_size:
                continue
            min_periods = min(window_size, max(2, window_size // 2))
            rolling_mean = series.rolling(window_size, min_periods=min_periods).mean().dropna()
            if not rolling_mean.empty:
                scale_data[scale_name] = rolling_mean

        n_scales = len(scale_data)
        if n_scales == 0:
            return

        fig, axes = plt.subplots(n_scales, 1, figsize=(12, 3 * n_scales), sharex=True)
        if n_scales == 1:
            axes = [axes]

        scale_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for i, (scale_name, rolling_mean) in enumerate(scale_data.items()):
            ax = axes[i]

            ax.plot(rolling_mean.index, rolling_mean.values,
                    color=scale_colors[i % len(scale_colors)],
                    alpha=0.7, linewidth=1.5, label=f"{scale_name} mean")

            # Add color range threshold lines matching the main NetVar chart
            for lower, upper, color, rgb, label in self.color_ranges:
                ax.axhline(y=lower, color=color, linestyle="--", linewidth=1, label=label)

            ax.set_ylabel(scale_name)
            ax.legend(loc='upper right', fontsize='small')
            self._apply_date_axis(ax)

        axes[-1].set_xlabel("Time")
        fig.suptitle("Multi-Scale NetVar (Scaled)", fontsize=14)
        plt.tight_layout()

        start_time_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.chart_dir, f"plot_multiscale_netvar_scaled_{start_time_str}.png")
        plt.savefig(filename)
        plt.close(fig)
        gc.collect()

    def plot_histogram(self, z1, z2, stouf, start_time):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(z1, bins=30, alpha=0.6, color='blue', edgecolor='black', label="RNG1 Z-score")
        ax.hist(z2, bins=30, alpha=0.6, color='orange', edgecolor='black', label="RNG2 Z-score")
        # ax.hist(stouf, bins=30, alpha=0.6, color='green', edgecolor='black', label="Stouffer's Z")

        ax.axvline(0, color='black', linestyle="dashed", linewidth=1.5)
        ax.axvline(3, color='red', linestyle="dashed", linewidth=2, label="Z = 3 (Unusual)")
        ax.axvline(-3, color='red', linestyle="dashed", linewidth=2)

        ax.set_title("Histogram of Z-Scores (RNG1, RNG2)")
        ax.set_xlabel("Z-Score")
        ax.set_ylabel("Frequency")
        ax.legend()

        start_time_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        file_name = os.path.join(self.chart_dir, f"histogram_{start_time_str}.png")
        plt.savefig(file_name)
        plt.close(fig)  # Close the figure to free memory
        gc.collect()  # Force garbage collection

    def reset(self, new_start_time):
        self.start_time = new_start_time