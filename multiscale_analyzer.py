import numpy as np
import pandas as pd
import scipy.stats as stats
from data_computer import DataComputer


# Window scales expressed as sample counts based on 2s bin size
# Minimum useful window for correlation is 4 samples (for Fisher Z df > 0)
WINDOW_SCALES = {
    "8s": 4,
    "20s": 10,
    "30s": 15,
    "1min": 30,
    "3min": 90,
    "5min": 150,
}


class MultiScaleAnalyzer:
    def __init__(self, window_scales=None):
        self.window_scales = window_scales or WINDOW_SCALES
        self.data_computer = DataComputer()
        self.results = {}  # scale_name -> dict of correlation data
        self.netvar_results = {}  # scale_name -> dict of netvar data
        self.significant_events = []

    def analyze(self, timestamps, z1, z2):
        """Run multi-scale rolling correlation analysis on binned Z-score series.

        Args:
            timestamps: list of datetime timestamps (already binned at 5s)
            z1: array-like of RNG1 Z-scores (binned)
            z2: array-like of RNG2 Z-scores (binned)

        Returns:
            dict mapping scale_name -> {correlations, fisher_z, p_values, significant}
        """
        z1 = np.asarray(z1, dtype=float)
        z2 = np.asarray(z2, dtype=float)
        n_points = len(z1)

        if n_points < 3:
            return {}

        df = pd.DataFrame({"z1": z1, "z2": z2}, index=timestamps)
        all_p_values = []
        all_p_indices = []  # (scale_name, position_in_that_scale's array)

        self.results = {}

        for scale_name, window_size in self.window_scales.items():
            if n_points < window_size:
                continue

            # Rolling Pearson correlation
            min_periods = min(window_size, max(2, window_size // 2))
            rolling_corr = df["z1"].rolling(window_size, min_periods=min_periods).corr(df["z2"])

            # Drop NaN and clip for Fisher transform
            valid_mask = rolling_corr.notna() & (rolling_corr.abs() < 1.0)
            valid_corr = rolling_corr[valid_mask].clip(-0.9999, 0.9999)

            if valid_corr.empty:
                continue

            # Fisher Z-transform
            fisher_z = self.data_computer.fisher_z_transform(valid_corr)

            # P-values with correct window size
            p_values = pd.Series(
                self.data_computer.compute_p_value(fisher_z.values, window_size=window_size),
                index=valid_corr.index
            )

            self.results[scale_name] = {
                "window_size": window_size,
                "correlations": valid_corr,
                "fisher_z": fisher_z,
                "p_values": p_values,
                "significant": pd.Series(False, index=valid_corr.index),  # filled after FDR
            }

            # Collect p-values for FDR correction
            for idx, p in p_values.items():
                all_p_values.append(p)
                all_p_indices.append((scale_name, idx))

        # Apply Benjamini-Hochberg FDR correction across ALL scales and timepoints
        if all_p_values:
            adjusted_p = self.data_computer.benjamini_hochberg(np.array(all_p_values))

            for i, (scale_name, idx) in enumerate(all_p_indices):
                if adjusted_p[i] < 0.05:
                    self.results[scale_name]["significant"].loc[idx] = True

            # Store adjusted p-values back into results
            for scale_name in self.results:
                scale_adjusted = []
                scale_indices = []
                for j, (sn, idx) in enumerate(all_p_indices):
                    if sn == scale_name:
                        scale_adjusted.append(adjusted_p[j])
                        scale_indices.append(idx)
                if scale_indices:
                    self.results[scale_name]["p_values_adjusted"] = pd.Series(
                        scale_adjusted, index=scale_indices
                    )

        # Build significant events list
        self._build_significant_events()

        return self.results

    def _build_significant_events(self):
        """Collect all FDR-significant events with scale metadata."""
        self.significant_events = []
        for scale_name, data in self.results.items():
            sig_mask = data["significant"]
            sig_times = sig_mask[sig_mask].index
            for t in sig_times:
                self.significant_events.append({
                    "timestamp": t,
                    "scale": scale_name,
                    "window_size": data["window_size"],
                    "correlation": data["correlations"].loc[t],
                    "fisher_z": data["fisher_z"].loc[t],
                    "p_value": data["p_values"].loc[t],
                    "p_value_adjusted": data.get("p_values_adjusted", pd.Series(dtype=float)).get(t, np.nan),
                })

    def get_significant_events(self):
        """Return list of significant events with scale metadata."""
        return self.significant_events

    def get_scale_summary(self):
        """Return per-scale summary statistics."""
        summary = {}
        for scale_name, data in self.results.items():
            n_sig = data["significant"].sum()
            n_total = len(data["significant"])
            min_p = data["p_values"].min() if not data["p_values"].empty else 1.0
            mean_corr = data["correlations"].mean() if not data["correlations"].empty else 0.0
            summary[scale_name] = {
                "window_size": data["window_size"],
                "n_significant": int(n_sig),
                "n_total": n_total,
                "min_p_value": min_p,
                "mean_correlation": mean_corr,
            }
        return summary

    def get_strongest_event(self):
        """Return the single most significant event across all scales."""
        if not self.significant_events:
            return None
        return min(self.significant_events, key=lambda e: e["p_value"])

    def analyze_netvar(self, timestamps, raw_netvar):
        """Run multi-scale rolling mean analysis on raw NetVar (chi-square(1) values).

        Under H0 each raw NetVar value is chi-square(1) with E=1.  For a window
        of N samples, sum = N * rolling_mean ~ chi-square(N).  One-sided p-value
        tests for elevated NetVar.

        Args:
            timestamps: list of datetime timestamps (already binned)
            raw_netvar: array-like of raw NetVar values (stouffers_z ** 2)

        Returns:
            dict mapping scale_name -> {window_size, rolling_mean, p_values,
                                         p_values_adjusted, significant}
        """
        raw_netvar = np.asarray(raw_netvar, dtype=float)
        n_points = len(raw_netvar)

        if n_points < 3:
            return {}

        series = pd.Series(raw_netvar, index=timestamps)
        all_p_values = []
        all_p_indices = []  # (scale_name, timestamp_index)

        self.netvar_results = {}

        for scale_name, window_size in self.window_scales.items():
            if n_points < window_size:
                continue

            min_periods = min(window_size, max(2, window_size // 2))
            rolling_mean = series.rolling(window_size, min_periods=min_periods).mean()

            valid_mask = rolling_mean.notna()
            valid_mean = rolling_mean[valid_mask]

            if valid_mean.empty:
                continue

            # sum = N * mean ~ chi-square(N) under H0; one-sided test for elevation
            chi2_sum = window_size * valid_mean.values
            p_values = pd.Series(
                1.0 - stats.chi2.cdf(chi2_sum, df=window_size),
                index=valid_mean.index,
            )

            self.netvar_results[scale_name] = {
                "window_size": window_size,
                "rolling_mean": valid_mean,
                "p_values": p_values,
                "significant": pd.Series(False, index=valid_mean.index),
            }

            for idx, p in p_values.items():
                all_p_values.append(p)
                all_p_indices.append((scale_name, idx))

        # Global FDR correction across all scales and timepoints
        if all_p_values:
            adjusted_p = self.data_computer.benjamini_hochberg(np.array(all_p_values))

            for i, (scale_name, idx) in enumerate(all_p_indices):
                if adjusted_p[i] < 0.05:
                    self.netvar_results[scale_name]["significant"].loc[idx] = True

            for scale_name in self.netvar_results:
                scale_adjusted = []
                scale_indices = []
                for j, (sn, idx) in enumerate(all_p_indices):
                    if sn == scale_name:
                        scale_adjusted.append(adjusted_p[j])
                        scale_indices.append(idx)
                if scale_indices:
                    self.netvar_results[scale_name]["p_values_adjusted"] = pd.Series(
                        scale_adjusted, index=scale_indices
                    )

        return self.netvar_results

    def get_netvar_scale_summary(self):
        """Return per-scale summary statistics for NetVar analysis."""
        summary = {}
        for scale_name, data in self.netvar_results.items():
            n_sig = data["significant"].sum()
            n_total = len(data["significant"])
            min_p = data["p_values"].min() if not data["p_values"].empty else 1.0
            mean_nv = data["rolling_mean"].mean() if not data["rolling_mean"].empty else 0.0
            summary[scale_name] = {
                "window_size": data["window_size"],
                "n_significant": int(n_sig),
                "n_total": n_total,
                "min_p_value": min_p,
                "mean_netvar": mean_nv,
            }
        return summary
