# DualRead — RNG Coherence Analyzer

Analyzes correlation between two independent unwhitened TrueRNG streams using Z-scores, rolling correlation, multi-scale analysis, and Bayesian inference.

## Setup

```bash
pip install -r requirements.txt
```

## Running a Trial

### GUI mode

```bash
python app.py --gui
python app.py --gui --duration 20
```

The idle screen shows a duration input field (default 15 minutes), a **New Trial** button, and a **Quit** button. Click the duration field, type a number, then click **New Trial**. The `--duration` flag sets the initial value but can be changed in the GUI before each trial.

During a trial the screen shows:
- An animated sine wave whose complexity and color reflect coherence strength
- **GCP** (NetVar) value and category label (top corners)
- **Cor** (rolling correlation) value (bottom-right)
- Colored tally circles for each significant NetVar spike
- Full-screen flash when an extreme NetVar event occurs

Double-tap to exit fullscreen. Press Escape to toggle windowed mode or quit.

### Headless mode

```bash
python app.py
python app.py --duration 30
```

Runs without a GUI. Stats print to the console every ~6 seconds. Ctrl-C stops the trial early.

### Baseline collection

```bash
python app.py --baseline --duration 5 --baseline-sessions 10
```

Records null-hypothesis data (no intention) across multiple short sessions. Produces `baseline/null_distributions.npz`, which later trials use to compute bootstrap p-values. Collect baseline before running real trials for the most rigorous analysis.

Rebuild distributions from existing session files:

```bash
python baseline_collector.py --rebuild
```

## Choosing a Duration

| Duration | Pros | Cons |
|----------|------|------|
| **5 min** | Quick iteration; exercises all 5 multi-scale windows (20s through 5min) | Few data points; rolling averages may not stabilize; wide confidence intervals |
| **15 min** (default) | Good balance of statistical power and practicality; rolling correlation MA converges; enough samples for meaningful FDR correction | May miss very slow coherence patterns |
| **30 min** | Strong statistical power; Bayes factors become more decisive; multi-scale analysis has deep history at every window size | Requires sustained focus if doing intention-based work |
| **60+ min** | Maximum sensitivity; tightest confidence intervals; robust bootstrap p-values against baseline | Diminishing returns per additional minute; large CSV files; fatigue effects |

**Minimum viable trial**: 5 minutes (all multi-scale windows need at least 60 five-second bins).
**Recommended starting point**: 15 minutes.

### How duration affects the statistics

- **Rolling correlation** uses a 10-bin window (~50 seconds). Short trials have few independent windows, so individual spikes carry disproportionate weight.
- **Stouffer's global Z** scales as the square root of the number of samples. Doubling duration increases sensitivity by ~1.4x.
- **Multi-scale FDR correction** tests every timepoint at every scale. Longer trials produce more tests, raising the bar each individual event must clear—but also giving real effects more chances to appear.
- **Bayes Factor (BF10)** includes a log(n) penalty, so weak effects that look promising in short trials may become inconclusive in longer ones. Only genuine effects survive.

## Reading the NetVar (NPS) Output

NetVar (network variance) measures how far the combined RNG output deviates from expected randomness at each moment, following the approach used by the Global Consciousness Project [1]. It is derived from Stouffer's Z [2] (the combined Z-score of both RNG streams): `NetVar = Stouffer_Z²`. This raw chi-square value is then scaled to a 1–324 display range.

### NetVar chart (`plot_netvar_*.png`)

The top panel shows Z1, Z2, and Stouffer's Z over time. The bottom panel shows scaled NetVar with colored horizontal bands marking category thresholds.

| NetVar Range | Label | Meaning |
|-------------|-------|---------|
| 1–139 | Normal | Baseline randomness |
| 140–165 | Elevated | Slightly unusual |
| 166–218 | High | Notable deviation |
| 219–278 | Very High | Strong effect |
| 279–324 | Extreme | Rare deviation |

### Multi-scale NetVar charts

- **`plot_multiscale_nv_*.png`** — Rolling mean of raw NetVar (Z²) at multiple window sizes (8s, 20s, 30s, 1min, 3min, 5min). Under the null hypothesis the mean should hover around 1.0. Points that deviate significantly after FDR correction are highlighted.
- **`plot_multiscale_nv_scaled_*.png`** — The same multi-scale view but using the scaled 1–324 range with category threshold lines overlaid.

### What to look for in NetVar

- **Normal trial**: values mostly in the 1–139 range, occasional spikes into Elevated.
- **Unusual trial**: sustained periods in the High range or repeated spikes into Very High/Extreme. The multi-scale chart will show the rolling mean lifting above 1.0 at one or more timescales.

## Reading the Correlation Output

Correlation measures whether the two RNG streams are moving together (positive) or apart (negative) over time, computed as a 10-bin rolling Pearson correlation [3] between binned Z1 and Z2 (~50-second window, updated every ~5 seconds).

### Correlation chart (`plot_corr_*.png`)

Shows the rolling correlation (blue) with a 20-bin moving average trendline (red dashed). Significant points (p < 0.05, effect size > 0.5) are marked in red. If baseline data is available, a blue shaded band shows the 95% confidence interval from null sessions. Dashed green/brown lines at +/-0.6 mark the high-correlation thresholds. A summary bar at the bottom shows trial outcome, BF10, significant scales, and bootstrap p-value.

| |r| | Label |
|-----|-------|
| < 0.1 | Negligible |
| 0.1–0.3 | Mild |
| 0.3–0.5 | Moderate |
| 0.5–0.7 | Strong |
| 0.7–0.9 | Very Strong |
| > 0.9 | Near Perfect |

### Multi-scale correlation chart (`plot_multiscale_*.png`)

One subplot per timescale (8s, 20s, 30s, 1min, 3min, 5min). Each shows rolling correlation at that window size. Points that survive Benjamini-Hochberg [4] FDR correction are marked in red. Longer timescales smooth out noise and reveal sustained coherence patterns; shorter timescales catch brief spikes.

### What to look for in correlation

- **Normal trial**: rolling correlation fluctuates around 0, moving average stays flat, few or no red markers.
- **Unusual trial**: sustained runs above 0.3 (or below -0.3), moving average trends away from zero, FDR-corrected events cluster at one or more timescales.

## Terminal Output Reference

At each refresh (~6 seconds) the console prints rolling correlation stats. A trial summary block prints each cycle with cumulative statistics:

```
Trial Summary (15 min)
   Global Z: 0.333, p: 0.7392 (theoretical), p: 0.4098 (bootstrap)
   Bayes Factor (BF10): 0.07
   Mean correlation: 0.0395 [-0.0924, 0.1700]
   Significant scales: 8s (1 events, min p=0.0000)
   Strongest event: 2026-02-11 00:07:46 at 8s scale (r=-0.9999, p=0.000001)
   Events: 1 survived FDR correction
   15 minute trial complete.
```

### Line-by-line breakdown

**`Global Z: 0.333`** — The trial-level test statistic. All rolling correlation values are Fisher-Z transformed [5] and combined via Stouffer's method [2]: `Z_global = sum(z_i) / sqrt(n)`. A single number summarizing the entire trial. Values near 0 mean no overall effect; values beyond +/-2 are unusual.

**`p: 0.7392 (theoretical)`** — The two-tailed p-value derived from the standard normal distribution. It answers: "If there were truly no coherence, how often would we see a Global Z this large or larger by chance?" Here, 0.74 means ~74% of the time — completely unremarkable.

**`p: 0.4098 (bootstrap)`** — A second, more conservative p-value computed by block bootstrap [6] from your baseline data. The procedure: sample 10,000 synthetic trials (contiguous blocks of baseline correlations, preserving autocorrelation), compute a Z_global for each, and count how many are as extreme as the observed value. This accounts for the actual statistical properties of your specific hardware rather than assuming textbook normality. Only appears if baseline data has been collected.

**`Bayes Factor (BF10): 0.07`** — The evidence ratio: how much more likely is the data under "coherence exists" (H1) versus "no coherence" (H0). Computed via BIC approximation [7]: `BF10 = exp((Z² - log(n)) / 2)`. Interpretation thresholds follow Jeffreys' scale as updated by Lee & Wagenmakers [8].

| BF10 | Interpretation |
|------|---------------|
| < 0.33 | Moderate evidence for H0 (no coherence) |
| 0.33–3 | Inconclusive |
| 3–10 | Moderate evidence for H1 (coherence) |
| 10–100 | Strong evidence for H1 |
| > 100 | Very strong evidence for H1 |

A BF10 of 0.07 means the data is ~14x more likely under no-coherence than under coherence.

**`Mean correlation: 0.0395 [-0.0924, 0.1700]`** — The average rolling correlation across all time bins, with a 95% confidence interval computed via Fisher-Z transform [5]. If the interval contains 0, the mean correlation is not distinguishable from chance.

**`Significant scales: 8s (1 events, min p=0.0000)`** — Which timescales produced events that survived Benjamini-Hochberg FDR correction [4] (see below), how many events at each scale, and the smallest adjusted p-value. "none" means no timescale had any surviving events.

**`Strongest event: 2026-02-11 00:07:46 at 8s scale (r=-0.9999, p=0.000001)`** — The single most statistically significant correlation event across all timescales. Shows the timestamp, which scale detected it, the correlation value, and the FDR-adjusted p-value.

**`Events: 1 survived FDR correction`** — Total count of events across all timescales that passed the Benjamini-Hochberg threshold. This is the bottom line for multi-scale analysis: how many moments of apparent coherence held up after correcting for multiple comparisons.

**`15 minute trial complete.`** — Trial outcome. Three possible messages:
- *"Statistically significant N minute trial"* — p_global < 0.05
- *"N minute trial approached significance"* — p_global between 0.05 and 0.10
- *"N minute trial complete"* — p_global >= 0.10 (no significant effect detected)

### Benjamini-Hochberg FDR correction

When testing hundreds of timepoints across multiple scales, some will appear significant by chance alone. If you test 300 points at p < 0.05, you expect ~15 false positives even with perfectly random data. The Benjamini-Hochberg (BH) procedure [4] controls the **false discovery rate** — the expected proportion of reported discoveries that are actually false. It works by ranking all p-values from smallest to largest and comparing each to a threshold that depends on its rank: `p_k <= (k/m) * alpha`, where `k` is the rank, `m` is the total number of tests, and `alpha` is the target FDR (0.05). Events that survive this correction are unlikely to be noise. Unlike the more conservative Bonferroni correction (which controls the chance of *any* false positive), BH allows more true discoveries through while still keeping the false discovery proportion under control.

## Output Files

All files are timestamped with the trial start time.

**`data/` directory:**
- `session_*.csv` — Raw per-sample data (timestamps, z1, z2, stouffers_z, nvar, svar)
- `correlation_*.csv` — Binned rolling correlations and moving averages
- `significant_events_*.csv` — Events where p < 0.05 and effect size > 0.5
- `multiscale_events_*.csv` — FDR-corrected events with scale, correlation, and adjusted p-value

**`charts/` directory:**
- `plot_netvar_*.png` — Z-scores over time (top) and NetVar with color thresholds (bottom)
- `plot_multiscale_nv_*.png` — Multi-scale NetVar with FDR-significant deviations marked
- `plot_multiscale_nv_scaled_*.png` — Multi-scale NetVar in scaled 1–324 range
- `plot_corr_*.png` — Rolling correlation with significant events, baseline CI, and summary stats
- `plot_multiscale_*.png` — Correlation at each timescale with FDR-significant points highlighted

## Examples

### What a normal trial looks like

- Global Z near 0, p around 0.5
- BF10 below 0.33 or near 1
- Bootstrap p-value well above 0.05
- Rolling correlation fluctuates around 0
- Few or no events survive FDR correction
- Summary: "[N] minute trial complete"

### What a significant trial looks like

- Global Z above 2 (or below -2), p below 0.05
- BF10 above 3
- Bootstrap p-value below 0.05, roughly agreeing with theoretical p
- Sustained periods of |r| > 0.3
- Multiple FDR-corrected events, often clustered at the 1–3 minute scales
- Summary: "Statistically significant [N] minute trial"

## References

[1] Nelson, R. D. et al. (2002). "Correlations of continuous random data with major world events." *Foundations of Physics Letters*, 15(6), 537–550. See also the [Global Consciousness Project](https://noosphere.princeton.edu/).

[2] Stouffer, S. A. et al. (1949). *The American Soldier: Adjustment During Army Life*. Princeton University Press. (Stouffer's Z method for combining independent significance tests.)

[3] Pearson, K. (1895). "Note on regression and inheritance in the case of two parents." *Proceedings of the Royal Society of London*, 58, 240–242.

[4] Benjamini, Y. & Hochberg, Y. (1995). "Controlling the false discovery rate: a practical and powerful approach to multiple testing." *Journal of the Royal Statistical Society, Series B*, 57(1), 289–300.

[5] Fisher, R. A. (1921). "On the 'probable error' of a coefficient of correlation deduced from a small sample." *Metron*, 1, 3–32. (Fisher Z transform for correlation inference.)

[6] Künsch, H. R. (1989). "The jackknife and the bootstrap for general stationary observations." *Annals of Statistics*, 17(3), 1217–1241. (Block bootstrap for dependent data.)

[7] Wagenmakers, E.-J. (2007). "A practical solution to the pervasive problems of p values." *Psychonomic Bulletin & Review*, 14(5), 779–804. (BIC approximation to Bayes factors.)

[8] Lee, M. D. & Wagenmakers, E.-J. (2013). *Bayesian Cognitive Modeling: A Practical Course*. Cambridge University Press. (Bayes factor interpretation thresholds.)
