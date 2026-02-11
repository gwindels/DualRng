import os
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
from data_computer import DataComputer
from multiscale_analyzer import MultiScaleAnalyzer
from baseline_collector import BaselineCollector


class PostHocAnalyzer:
    """Reanalyzes historical DualRead session data with improved multi-scale
    statistics, FDR correction, and baseline comparison."""

    def __init__(self, data_dir="data", baseline_dir="baseline"):
        self.data_dir = data_dir
        self.baseline_dir = baseline_dir
        self.data_computer = DataComputer()
        self.sessions = []
        self.baseline = BaselineCollector.load_baseline(baseline_dir)

    def load_sessions(self, data_dir=None):
        """Load all session CSVs from the data directory.

        Returns:
            list of DataFrames, one per session.
        """
        data_dir = data_dir or self.data_dir
        pattern = os.path.join(data_dir, "session_*.csv")
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"No session files found in {data_dir}")
            return []

        self.sessions = []
        for f in files:
            try:
                df = pd.read_csv(f, parse_dates=["timestamps"])
                df["source_file"] = os.path.basename(f)
                self.sessions.append(df)
            except Exception as e:
                print(f"  Skipped {f}: {e}")

        print(f"Loaded {len(self.sessions)} sessions from {data_dir}")
        return self.sessions

    def reanalyze_session(self, session_df):
        """Apply multi-scale analysis and FDR correction to a historical session.

        Args:
            session_df: DataFrame with columns timestamps, z_scores_1, z_scores_2

        Returns:
            dict with multi-scale results and summary statistics.
        """
        if session_df.empty:
            return None

        # Bin the data at 5s intervals
        timestamps = session_df["timestamps"].tolist()
        z1 = session_df["z_scores_1"].values
        z2 = session_df["z_scores_2"].values

        timestamps_binned, smoothed_values = self.data_computer.aggregate_data_by_time(
            timestamps, z1, z2, time_window="5s"
        )
        if not timestamps_binned:
            return None

        smoothed_z1, smoothed_z2 = smoothed_values

        # Multi-scale analysis
        analyzer = MultiScaleAnalyzer()
        results = analyzer.analyze(timestamps_binned, smoothed_z1, smoothed_z2)

        # Global significance
        df_corr = pd.DataFrame({"z1": smoothed_z1, "z2": smoothed_z2}, index=timestamps_binned)
        rolling_corr = df_corr["z1"].rolling(10, min_periods=1).corr(df_corr["z2"]).dropna()
        valid_corr = rolling_corr[(rolling_corr > -1) & (rolling_corr < 1)].clip(-0.9999, 0.9999)

        Z_global, p_global = None, None
        if not valid_corr.empty:
            Z_global, p_global = self.data_computer.compute_global_trial_significance(valid_corr)

        # Bootstrap p-value
        p_empirical = None
        if Z_global is not None and self.baseline is not None and "corr_20s" in self.baseline:
            p_empirical = self.data_computer.bootstrap_trial_p_value(
                Z_global, self.baseline["corr_20s"], len(valid_corr)
            )

        return {
            "multiscale_results": results,
            "scale_summary": analyzer.get_scale_summary(),
            "significant_events": analyzer.get_significant_events(),
            "strongest_event": analyzer.get_strongest_event(),
            "Z_global": Z_global,
            "p_global": p_global,
            "p_empirical": p_empirical,
            "n_bins": len(timestamps_binned),
            "source_file": session_df.get("source_file", pd.Series(["unknown"])).iloc[0],
        }

    def compare_sessions(self, baseline_sessions, active_sessions):
        """Statistical comparison between baseline and active session groups.

        Args:
            baseline_sessions: list of reanalysis result dicts
            active_sessions: list of reanalysis result dicts

        Returns:
            dict with comparison statistics.
        """
        comparison = {}

        # Collect per-scale correlation distributions
        for scale_name in ["20s", "30s", "1min", "3min", "5min"]:
            base_corr = []
            active_corr = []

            for s in baseline_sessions:
                if s and scale_name in s.get("multiscale_results", {}):
                    base_corr.extend(s["multiscale_results"][scale_name]["correlations"].values)
            for s in active_sessions:
                if s and scale_name in s.get("multiscale_results", {}):
                    active_corr.extend(s["multiscale_results"][scale_name]["correlations"].values)

            if base_corr and active_corr:
                base_arr = np.array(base_corr)
                active_arr = np.array(active_corr)

                # Mann-Whitney U test
                u_stat, u_p = stats.mannwhitneyu(base_arr, active_arr, alternative='two-sided')
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.ks_2samp(base_arr, active_arr)
                # Cohen's d effect size
                pooled_std = np.sqrt((np.var(base_arr) + np.var(active_arr)) / 2)
                cohens_d = (np.mean(active_arr) - np.mean(base_arr)) / pooled_std if pooled_std > 0 else 0

                comparison[scale_name] = {
                    "mann_whitney_U": u_stat,
                    "mann_whitney_p": u_p,
                    "ks_stat": ks_stat,
                    "ks_p": ks_p,
                    "cohens_d": cohens_d,
                    "baseline_mean": np.mean(base_arr),
                    "active_mean": np.mean(active_arr),
                    "baseline_n": len(base_arr),
                    "active_n": len(active_arr),
                }

        # Compare Z-score distributions
        base_z = []
        active_z = []
        for s in baseline_sessions:
            if s and s["Z_global"] is not None:
                base_z.append(s["Z_global"])
        for s in active_sessions:
            if s and s["Z_global"] is not None:
                active_z.append(s["Z_global"])

        if base_z and active_z:
            ks_stat, ks_p = stats.ks_2samp(base_z, active_z)
            comparison["global_z"] = {
                "ks_stat": ks_stat,
                "ks_p": ks_p,
                "baseline_mean": np.mean(base_z),
                "active_mean": np.mean(active_z),
            }

        # Compare event rates
        base_rates = []
        active_rates = []
        for s in baseline_sessions:
            if s:
                n_events = len(s.get("significant_events", []))
                n_bins = s.get("n_bins", 1)
                base_rates.append(n_events / max(1, n_bins * 5 / 60))  # events per minute
        for s in active_sessions:
            if s:
                n_events = len(s.get("significant_events", []))
                n_bins = s.get("n_bins", 1)
                active_rates.append(n_events / max(1, n_bins * 5 / 60))

        if base_rates and active_rates:
            u_stat, u_p = stats.mannwhitneyu(base_rates, active_rates, alternative='two-sided')
            comparison["event_rate"] = {
                "mann_whitney_U": u_stat,
                "mann_whitney_p": u_p,
                "baseline_mean": np.mean(base_rates),
                "active_mean": np.mean(active_rates),
            }

        return comparison

    def generate_summary_report(self):
        """Generate aggregate statistics across all loaded sessions.

        Returns:
            string report.
        """
        if not self.sessions:
            return "No sessions loaded."

        results = []
        for session_df in self.sessions:
            r = self.reanalyze_session(session_df)
            if r:
                results.append(r)

        if not results:
            return "No sessions could be reanalyzed."

        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"POST-HOC ANALYSIS REPORT")
        lines.append(f"{'='*60}")
        lines.append(f"Sessions analyzed: {len(results)}")
        lines.append("")

        # Global significance summary
        p_values = [r["p_global"] for r in results if r["p_global"] is not None]
        significant = [p for p in p_values if p < 0.05]
        lines.append(f"Globally significant sessions (p < 0.05): {len(significant)}/{len(p_values)}")
        if p_values:
            lines.append(f"  Mean global p-value: {np.mean(p_values):.4f}")
            lines.append(f"  Median global p-value: {np.median(p_values):.4f}")
        lines.append("")

        # Per-scale summary
        lines.append("Per-Scale Summary:")
        lines.append(f"  {'Scale':<8} {'Sig Events':>12} {'Min p':>10} {'Mean |r|':>10}")
        lines.append(f"  {'-'*42}")
        for scale_name in ["20s", "30s", "1min", "3min", "5min"]:
            total_sig = 0
            min_p = 1.0
            corr_vals = []
            for r in results:
                summary = r.get("scale_summary", {}).get(scale_name, {})
                total_sig += summary.get("n_significant", 0)
                min_p = min(min_p, summary.get("min_p_value", 1.0))
                corr_vals.append(abs(summary.get("mean_correlation", 0)))
            mean_corr = np.mean(corr_vals) if corr_vals else 0
            lines.append(f"  {scale_name:<8} {total_sig:>12} {min_p:>10.4f} {mean_corr:>10.4f}")
        lines.append("")

        # Strongest events across all sessions
        strongest_events = [r["strongest_event"] for r in results if r["strongest_event"]]
        if strongest_events:
            best = min(strongest_events, key=lambda e: e["p_value"])
            lines.append("Strongest event across all sessions:")
            lines.append(f"  Time: {best['timestamp']}")
            lines.append(f"  Scale: {best['scale']}")
            lines.append(f"  Correlation: {best['correlation']:.4f}")
            lines.append(f"  p-value: {best['p_value']:.6f}")
        lines.append("")

        # Empirical p-values if baseline available
        emp_p = [r["p_empirical"] for r in results if r["p_empirical"] is not None]
        if emp_p:
            lines.append("Empirical p-values (vs. baseline):")
            lines.append(f"  Mean: {np.mean(emp_p):.4f}")
            lines.append(f"  Median: {np.median(emp_p):.4f}")
            sig_emp = [p for p in emp_p if p < 0.05]
            lines.append(f"  Significant (p < 0.05): {len(sig_emp)}/{len(emp_p)}")

        lines.append(f"\n{'='*60}")
        report = "\n".join(lines)
        print(report)
        return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-hoc analysis of DualRead session data")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing session CSV files")
    parser.add_argument("--baseline-dir", type=str, default="baseline",
                        help="Directory containing baseline null distributions")
    args = parser.parse_args()

    analyzer = PostHocAnalyzer(data_dir=args.data_dir, baseline_dir=args.baseline_dir)
    analyzer.load_sessions()
    analyzer.generate_summary_report()
