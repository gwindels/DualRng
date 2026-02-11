import pandas as pd
import numpy as np
import scipy.stats as stats

class DataComputer:

    def calculate_zscore(self, trials, chunk_size = 32):
        """Calculate Z-score for deviation from expected mean."""
        if len(trials) == 0:
            return 0
        # print(f"Trials: {trials} and {int(trials)}")
        num_samples = chunk_size * 8
        expected_mean = num_samples / 2 # 100
        expected_std = np.sqrt(num_samples / 4)  # sqrt(npq) where n=200, p=q=0.5
        actual_mean = np.mean(int(trials))
        # return (actual_mean - expected_mean) / expected_std
        actual_z = (actual_mean - expected_mean) / expected_std
        return actual_z
    
    def compute_stouffers_z(self, *z_score_lists):
        """Computes Stouffer’s Z-score from multiple independent Z-score series."""
        z_array = np.array(z_score_lists)
        return np.sum(z_array, axis=0) / np.sqrt(z_array.shape[0])  # Normalizes correctly

    def fisher_z_transform(self, r):
        return 0.5 * np.log((1 + r) / (1 - r))

    def compute_p_value(self, r, window_size=10):
        if window_size <= 3:
            # Not enough degrees of freedom for Fisher Z test
            return np.ones_like(r) if hasattr(r, '__len__') else 1.0
        std_error = 1 / np.sqrt(window_size - 3)
        return (2 * (1 - stats.norm.cdf(abs(r) / std_error)))

    def benjamini_hochberg(self, p_values):
        """Apply Benjamini-Hochberg FDR correction to an array of p-values.
        Returns adjusted p-values (same shape as input)."""
        p = np.asarray(p_values, dtype=float)
        n = len(p)
        if n == 0:
            return p
        # Rank p-values (1-based)
        sorted_idx = np.argsort(p)
        sorted_p = p[sorted_idx]
        # BH adjustment: p_adj[i] = p[i] * n / rank[i], then enforce monotonicity
        ranks = np.arange(1, n + 1)
        adjusted = sorted_p * n / ranks
        # Enforce monotonicity from the largest rank down
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0, 1)
        # Map back to original order
        result = np.empty(n)
        result[sorted_idx] = adjusted
        return result

    def empirical_p_value(self, observed, null_distribution):
        """Compute empirical p-value as the proportion of null values
        at least as extreme as the observed value (two-tailed)."""
        null = np.asarray(null_distribution)
        if len(null) == 0:
            return 1.0
        return np.mean(np.abs(null) >= np.abs(observed))

    def bootstrap_trial_p_value(self, observed_z_global, null_correlations,
                                trial_n, n_bootstrap=10000, block_size=10):
        """Compute a bootstrap p-value for a trial-level Z_global.

        Builds a null distribution of trial-level Z_global values via block
        bootstrap from baseline correlation data, then computes a two-tailed
        p-value.

        Args:
            observed_z_global: the trial's observed Z_global statistic
            null_correlations: baseline correlation values (e.g. corr_20s)
            trial_n: number of correlation values in the trial
            n_bootstrap: number of bootstrap iterations (default 10000)
            block_size: contiguous block size for block bootstrap (default 10)

        Returns:
            Two-tailed p-value, or None if insufficient baseline data.
        """
        corr = np.asarray(null_correlations, dtype=float)
        # Filter to valid correlation range and clip
        corr = corr[np.isfinite(corr) & (corr > -1) & (corr < 1)]
        if len(corr) < block_size or trial_n <= 0:
            return None
        corr = np.clip(corr, -0.9999, 0.9999)

        # Fisher-Z transform baseline correlations
        z_baseline = self.fisher_z_transform(corr)

        # Block bootstrap
        n_blocks = int(np.ceil(trial_n / block_size))
        max_start = len(z_baseline) - block_size  # last valid block start index
        if max_start < 0:
            return None

        rng = np.random.default_rng()
        null_z_globals = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            # Sample random contiguous block start indices
            starts = rng.integers(0, max_start + 1, size=n_blocks)
            # Concatenate blocks and trim to trial length
            blocks = np.concatenate([z_baseline[s:s + block_size] for s in starts])
            samples = blocks[:trial_n]
            # Compute Z_global = sum(z) / sqrt(n)
            null_z_globals[i] = np.sum(samples) / np.sqrt(len(samples))

        # Two-tailed p-value
        return float(np.mean(np.abs(null_z_globals) >= np.abs(observed_z_global)))

    def bayes_factor_z(self, z, n, r_scale=0.707):
        """Approximate Bayes Factor (BF10) for a Z-test using the JZS prior.

        Uses the BIC approximation: BF10 ≈ exp((Z^2 - log(n)) / 2)
        This is a simple, widely-used approximation for converting a Z-score
        to evidence for H1 vs H0.

        Args:
            z: observed Z-score
            n: sample size
            r_scale: Cauchy prior scale (default 0.707, medium effect)

        Returns:
            BF10 (values > 1 favor H1, < 1 favor H0)
        """
        if n <= 0:
            return 1.0
        # BIC approximation
        bf10 = np.exp((z**2 - np.log(n)) / 2)
        return float(bf10)

    def correlation_confidence_interval(self, r, n, alpha=0.05):
        """Compute confidence interval for a Pearson correlation using Fisher Z.

        Args:
            r: observed correlation
            n: sample size
            alpha: significance level (default 0.05 for 95% CI)

        Returns:
            (lower, upper) bounds of the CI
        """
        if n < 4:
            return (-1.0, 1.0)
        z = self.fisher_z_transform(np.clip(r, -0.9999, 0.9999))
        se = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se
        # Transform back
        r_lower = np.tanh(z_lower)
        r_upper = np.tanh(z_upper)
        return (float(r_lower), float(r_upper))

    def smooth_data(self, data, window_size=10):
        """Applies a rolling average to smooth the data."""
        if len(data) < window_size:
            return data  # Not enough data to smooth
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def compute_global_trial_significance(self, valid_corr):
        if valid_corr.empty:
            print("Warning: No valid correlations to compute global significance.")
            return None, None
        # Fisher Z-transform the correlation values
        z_values = self.fisher_z_transform(valid_corr)
        # Stouffer's Z: combined significance
        Z_global = np.sum(z_values) / np.sqrt(len(z_values))
        # Two-tailed p-value
        p_global = 2 * (1 - stats.norm.cdf(abs(Z_global)))
        return Z_global, p_global
    
    def aggregate_data_by_time(self, timestamps, *data_series, time_window="3s"):
        """Aggregate multiple data series by averaging values within fixed time windows."""
        if len(timestamps) == 0:
            print("Warning: No timestamps provided. Returning empty results.")
            return [], [[] for _ in range(len(data_series))]
        df = pd.DataFrame({"timestamps": timestamps})
        for i, series in enumerate(data_series):
            df[f"data_{i}"] = series
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        df.set_index("timestamps", inplace=True)
        # Resample and handle potential empty results
        df_resampled = df.resample(time_window).mean().dropna()
        if df_resampled.empty:
            print("Warning: Resampled dataframe is empty. Returning empty lists.")
            return [], [[] for _ in range(len(data_series))]
        return df_resampled.index.to_list(), [df_resampled[f"data_{i}"].values for i in range(len(data_series))]

    def smooth_data_set(self, z1, z2, stouffers_z, timestamps, netvar, category):
        # Smooth data for plotting
        smoothed_z1 = self.smooth_data(z1)
        smoothed_z2 = self.smooth_data(z2)
        smoothed_stouf = self.smooth_data(stouffers_z)
        smoothed_nv = self.smooth_data(netvar)
        # Apply time-binned averaging in save_plot: 
        timestamps_binned, smoothed_values = self.aggregate_data_by_time(
            timestamps, z1, z2, stouffers_z, netvar, time_window="5s"
        )
        if len(timestamps_binned) == 0:
            print("Warning: No valid timestamps after binning.")
            return
        smoothed_z1, smoothed_z2, smoothed_stouf, smoothed_nv = smoothed_values
        # Ensure timestamps_binned and smoothed_nv have the same length
        min_length = min(len(timestamps_binned), len(smoothed_nv))
        timestamps_binned = timestamps_binned[:min_length]
        smoothed_nv = smoothed_nv[:min_length]
        smoothed_z1 = smoothed_z1[:min_length]
        smoothed_z2 = smoothed_z2[:min_length]
        smoothed_stouf = smoothed_stouf[:min_length]
        return timestamps_binned, smoothed_z1, smoothed_z2, smoothed_stouf, smoothed_nv, category

    def compute_correlation(self, timestamps_binned, smoothed_z1, smoothed_z2):
        # Compute Rolling Correlation
        df = pd.DataFrame({"timestamp": timestamps_binned, "z1": smoothed_z1, "z2": smoothed_z2})
        df.set_index("timestamp", inplace=True)
        df["rolling_corr"] = df["z1"].rolling(10, min_periods=1).corr(df["z2"])  # Rolling correlation
        df["rolling_corr_ma"] = df["rolling_corr"].rolling(20).mean() # Moving Average Trendline
        return df
    
    def compute_significant_points(self, df, window_size=10):
        # Ensure there are correlation values
        valid_corr = df["rolling_corr"].dropna()
        if valid_corr.empty:
            print("Warning: No valid correlation values after rolling window computation.")
            return
        # Ensure valid range
        valid_corr = valid_corr[(valid_corr > -1) & (valid_corr < 1)]
        if valid_corr.empty:
            print("Warning: No valid correlation values after filtering.")
            return
        # Prevent division by zero
        valid_corr = valid_corr.clip(-0.9999, 0.9999)
        # ==== Determine Significant Events ====
        z_values = 0.5 * np.log((1 + valid_corr) / (1 - valid_corr))
        std_error = 1 / np.sqrt(window_size - 3)
        p_values = 2 * (1 - stats.norm.cdf(abs(z_values) / std_error))
        effect_sizes = np.abs(z_values)
        # Ensure `significance_threshold` is a Pandas Series aligned with `df.index`
        significance_threshold = pd.Series((p_values < 0.05) & (effect_sizes > 0.5), index=valid_corr.index)
        # Reindex `significance_threshold` to match `df.index`, filling missing values with False
        significance_threshold = significance_threshold.reindex(df.index, fill_value=False)
        # Correctly filter `df` using the aligned boolean mask
        significant_points = df[significance_threshold]
        return significant_points, valid_corr, p_values, effect_sizes
    
    def compute_significant_events(self, significant_points, valid_corr, p_values, effect_sizes):
        # Convert to Pandas Series to allow .loc[] indexing
        p_values = pd.Series(p_values, index=valid_corr.index)
        effect_sizes = pd.Series(effect_sizes, index=valid_corr.index)
        # Store events in a list
        significant_events = [] 
        for index, row in significant_points.iterrows():
            significant_events.append({
                "timestamp": index,
                "correlation": row["rolling_corr"],
                "p_value": p_values.loc[index],
                "effect_size": effect_sizes.loc[index]
            })
        return significant_events
    
