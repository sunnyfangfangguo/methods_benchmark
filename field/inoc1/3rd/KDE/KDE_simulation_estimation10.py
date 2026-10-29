# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 09:00:45 2026

@author: sunny
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 16:27:31 2026

@author: sunny
"""

import os
import math

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns



# ==========================================
# 1. KDE METHODS (Heuristic)
# ==========================================

def plot_kde2focus_comparison(df, sy, dens, y_peak_0, y_peak_1, sim):
    sns.set_style(style='white')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Primary Axis: KDE Density ---
    line1, = ax1.plot(sy, dens, color='blue', label='KDE Density', linewidth=2)
    ax1.set_xlabel('Distance (Y)')
    ax1.set_ylabel('KDE Density')
    ax1.tick_params(axis='y')
    
    # Mark the peak found by the function
    vline1 = ax1.axvline(x=y_peak_0, color='black', linestyle='--', alpha=0.6, label=f'KDE Peak ({y_peak_0:.2f})')
    vline2 = ax1.axvline(x=y_peak_1, color='black', linestyle='--', alpha=0.6, label=f'KDE Peak ({y_peak_1:.2f})')

    # --- Secondary Axis: Real Severity ---
    ax2 = ax1.twinx()
    # Plotting actual severity as a stem plot or bar to show individual plant data
    markerline, stemlines, _ = ax2.stem(df['Y'], df['Severity'], 
                                        linefmt='orange', markerfmt='o', 
                                        label='Observed Severity', basefmt=" ")
    markerline.set_label('Observed Severity')  # stem() sets the label on the
                                            # returned container, not on
                                            # markerline itself
    plt.setp(markerline, 'color', 'orange', 'markersize', 6)
    
    ax2.set_ylabel('Observed Severity')
    ax2.tick_params(axis='y')

    # --- Formatting ---
    row = df['Row'].iloc[0]
    plt.title(f"Comparison: KDE Density vs. Real Severity (Row X={row})")
    
    # Combined legend for both axes
    lines = [line1, vline1, vline2, markerline]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig(f"./output/compare_KDE_density_real_severity_sim{sim}_row{row}.png")
    


def reflection_kde_1d(coords, weights, y_min=0.0, y_max=24.99, bw_method='silverman', n_points=100):
    """
    1D Reflection KDE bounded on [y_min, y_max].
    Reflects points across lower and upper row boundaries to eliminate edge bias.
    """
    coords = np.asarray(coords)
    weights = np.asarray(weights)
    
    # 1. Compute physical bandwidth h on original UNREFLECTED data
    kde_orig = gaussian_kde(coords, weights=weights, bw_method=bw_method)
    h_orig = np.sqrt(kde_orig.covariance[0, 0])
    
    # 2. Construct 3-fold reflected coordinates (Left: Y_min, Original, Right: Y_max)
    c_left = 2 * y_min - coords
    c_right = 2 * y_max - coords
    
    c_aug = np.concatenate([c_left, coords, c_right])
    w_aug = np.tile(weights, 3)
    
    # 3. Adjust bandwidth factor to preserve h_orig across augmented domain
    avg = np.average(c_aug, weights=w_aug)
    std_aug = np.sqrt(np.average((c_aug - avg)**2, weights=w_aug))
    bw_factor = h_orig / std_aug if std_aug > 0 else 1.0
    
    # 4. Fit KDE on augmented dataset
    kde_ref = gaussian_kde(c_aug, weights=w_aug, bw_method=bw_factor)
    
    # 5. Evaluate density over bounded domain [y_min, y_max]
    sy = np.linspace(y_min, y_max, n_points)
    # Scale by 3.0 to restore total probability mass integral over [y_min, y_max]
    dens = kde_ref(sy) * 3.0
    
    return sy, dens, kde_ref




def twofoci_kde(df, sim, buffer=0, n_points=100, enable_plot=True):
    """
    Finds the TWO highest distinct local maxima in the KDE.
    """
    # 1. Fit KDE using 'Severity' as weights
    x_coords = df['X'].values
    y_coords = df['Y'].values
    severities = df['Severity'].values.astype(float)
    
    # Fallback to uniform weights if all severities are 0 or constant
    if np.sum(severities) == 0 or np.all(severities == severities[0]):
        severities = np.ones_like(severities)
    
    # row-specific dynamic domain bounds
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Run 1D Reflection KDE
    sy, dens, kde_1d = reflection_kde_1d(
    coords=y_coords, weights=severities,
    y_min=y_min, y_max=y_max, bw_method='silverman',
    n_points=n_points)
    
    # 2. Convert a physical minimum distance threshold (e.g., 5.0 Y-units) to grid steps
    min_y_distance = 5.0  
    step_size = (y_max - y_min) / (n_points - 1)
    min_dist_steps = max(1, int(min_y_distance / step_size))
    
    # 2. Find 1D peaks using scipy
    # Pad dens with -inf so find_peaks can detect maxima directly on boundary indices (0 and -1)
    dens_padded = np.pad(dens, (1, 1), mode='constant', constant_values=-np.inf)
    peaks_idx_padded, _ = find_peaks(dens_padded, distance=min_dist_steps)
    peaks_idx = peaks_idx_padded - 1  # Shift indices back to original array scale
    
    if len(peaks_idx) >= 2:
        sorted_peaks = peaks_idx[np.argsort(dens[peaks_idx])[::-1]]
        idx0, idx1 = sorted_peaks[0], sorted_peaks[1]
        peak0_val, peak1_val = dens[idx0], dens[idx1]
        
        # Valley validation between the top 2 peaks
        lo, hi = sorted([idx0, idx1])
        valley_min = dens[lo:hi + 1].min()
        lower_peak = min(peak0_val, peak1_val)
        background = np.median(dens)
        
        # Require dip depth to be at least 10% of lower peak height above background
        if (lower_peak - valley_min) >= 0.02 * (lower_peak - background):
            y_peak_0 = min(sy[idx0], sy[idx1])
            y_peak_1 = max(sy[idx0], sy[idx1])
        else:
            y_peak_0 = sy[idx0]
            y_peak_1 = sy[idx1]
    elif len(peaks_idx) == 1:
        y_peak_0 = y_peak_1 = sy[peaks_idx[0]]
    else:
        y_peak_0 = y_peak_1 = sy[np.argmax(dens)]
    
    kde_peak = ((x_coords[0], y_peak_0), (x_coords[0], y_peak_1))

    if enable_plot:
        plot_kde2focus_comparison(df, sy, dens, y_peak_0, y_peak_1, sim)
        

    return kde_peak


def bootstrap_peak_ci_kde(
    df, sim, orig_peaks, n_boot=300, ci_level=0.95,
    buffer=0, n_points=100, random_state=None):
  """Case-resampling bootstrap CI for the two KDE peak locations.

  Resamples rows with replacement, refits gaussian_kde + relocates peaks
  via twofoci_kde each time, and looks at where the peak actually lands
  across replicates. KDE has no parametric analog to a GLM's cov_params()
  -- no coefficient vector, no asymptotic covariance for the mode -- so
  there's no faster parametric-simulation shortcut available here; this is
  the natural (and only readily available) approach for this method.

  Same conventions as the GLM's bootstrap_peak_ci: which candidate comes
  back first from twofoci_kde is arbitrary per resample, so each
  replicate's two candidates are re-matched to the ORIGINAL fit's
  peak0/peak1 (never to ground truth) before pooling. Resamples where the
  two candidates collapse to the same point are skipped, not pooled in --
  a collapsed replicate doesn't cleanly support either true focus, and
  including it would silently narrow the interval instead of honestly
  reflecting that the resampled fit couldn't resolve two peaks that time.

  Cost: n_boot refits per row instead of a closed-form calculation --
  gaussian_kde itself is cheap to fit, but evaluating it over an
  n_points x n_points grid n_boot times per row adds up. Test on a handful
  of rows first to gauge actual runtime before committing to a full batch.

  Returns (ci_x0, ci_y0, ci_x1, ci_y1, n_valid), where n_valid is how many
  of the n_boot resamples produced two distinct located peaks.

  If orig_peaks itself is collapsed (peak0_ref == peak1_ref -- the original,
  non-bootstrapped fit couldn't tell the two foci apart), this returns NaN
  immediately without attempting any resampling. The re-matching step below
  compares d_same = dist(cand0,peak0_ref)+dist(cand1,peak1_ref) against
  d_swap = dist(cand0,peak1_ref)+dist(cand1,peak0_ref); when peak0_ref
  equals peak1_ref these are identical by construction for every replicate,
  so the comparison never actually discriminates which candidate is closer
  to which reference -- on the rare resamples that do find two distinct
  (often noise-driven) peaks, whichever comes back first from that specific
  replicate gets pooled into boot_p0 with no real connection to
  peak0_ref/peak1_ref. This produced a concretely wrong CI on the GLM
  version of this same logic (a collapsed point estimate of -0.4636 got a
  7-replicate CI of [0.39, 1.96], which doesn't contain the point estimate
  at all), so it's fixed here too before it shows up the same way in KDE's
  output.
  """
  peak0_ref, peak1_ref = orig_peaks
  if peak0_ref == peak1_ref:
    nan_ci = (np.nan, np.nan)
    return nan_ci, nan_ci, nan_ci, nan_ci, 0

  rng = np.random.default_rng(random_state)
  n = len(df)
  boot_p0, boot_p1 = [], []

  for _ in range(n_boot):
    boot_idx = rng.integers(0, n, size=n)
    boot_df = df.iloc[boot_idx].reset_index(drop=True)

    try:
      cand0, cand1 = twofoci_kde(boot_df, sim, buffer=buffer, n_points=n_points, enable_plot=False)
    except Exception:
      continue
    if cand0 == cand1:
      continue

    d_same = math.dist(cand0, peak0_ref) + math.dist(cand1, peak1_ref)
    d_swap = math.dist(cand0, peak1_ref) + math.dist(cand1, peak0_ref)
    if d_same <= d_swap:
      boot_p0.append(cand0)
      boot_p1.append(cand1)
    else:
      boot_p0.append(cand1)
      boot_p1.append(cand0)

  n_valid = len(boot_p0)
  if n_valid < 2:
    nan_ci = (np.nan, np.nan)
    return nan_ci, nan_ci, nan_ci, nan_ci, n_valid

  boot_p0, boot_p1 = np.asarray(boot_p0), np.asarray(boot_p1)
  alpha = 1 - ci_level
  lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)

  def _pct_ci(values):
    return tuple(np.percentile(values, [lo_pct, hi_pct]))

  return (
      _pct_ci(boot_p0[:, 0]), _pct_ci(boot_p0[:, 1]),
      _pct_ci(boot_p1[:, 0]), _pct_ci(boot_p1[:, 1]),
      n_valid,
  )


# ==========================================
# 2. STATISTICAL SUMMARY GENERATOR FUNCTIONS
# ==========================================

def create_summary_table_focus(
    focal_param, all_metrics_df, model_type="Weighted KDE", est_prefix="KDE_peak"
):
  """Same metric set and table shape as the Spatial GLM pipeline's version:
  Model, Focal Parameter, True Value, N Observations, Mean Estimate, Mean
  Bias, RMSE, Coverage Rate %, Mean 95% CI, Mean CI Width -- so results
  from the two methods drop into the same comparison table directly.
  """
  est_col = f"{est_prefix}{focal_param}"

  # Flexible candidate mapping for true_x0 vs trueX0 naming formats
  param_map = {
      "X0": ["true_x0", "trueX0"],
      "Y0": ["true_y0", "trueY0"],
      "X1": ["true_x1", "trueX1"],
      "Y1": ["true_y1", "trueY1"],
  }

  candidates = param_map.get(
      focal_param, [f"true_{focal_param.lower()}", f"true{focal_param}"]
  )
  true_col = next((c for c in candidates if c in all_metrics_df.columns), None)

  if true_col is None:
    raise KeyError(
        f"Could not find ground truth column for {focal_param} in DataFrame."
    )

  ci_low_col = f"CI{focal_param.lower()}_Lower"
  ci_high_col = f"CI{focal_param.lower()}_Upper"

  n_total = len(all_metrics_df)
  # NaN CIs come from bootstrap_peak_ci_kde when fewer than 2 resamples
  # produced two resolvable, distinct peaks (n_valid < 2). KDE_peak*/true_*
  # are never NaN themselves, only the CI bounds can be. Comparing a real
  # number to NaN with >=/<= evaluates to False, not NaN, so without this
  # filter those rows would silently count as "true value not covered" and
  # deflate the coverage rate without ever showing a NaN anywhere.
  # Filtering here (not just for the CI columns) also keeps every number in
  # this table referring to the same subset of rows, rather than
  # N Observations reporting one count while Mean CI Width was silently
  # computed over a smaller one.
  valid_df = all_metrics_df.dropna(subset=[ci_low_col, ci_high_col])
  n_dropped = n_total - len(valid_df)

  true_series = valid_df[true_col]
  est_series = valid_df[est_col]

  bias_series = est_series - true_series
  rmse_val = np.sqrt(np.mean(bias_series**2))

  ci_low_mean = valid_df[ci_low_col].mean()
  ci_high_mean = valid_df[ci_high_col].mean()
  ci_widths = valid_df[ci_high_col] - valid_df[ci_low_col]

  coverage_mask = (true_series >= valid_df[ci_low_col]) & (
      true_series <= valid_df[ci_high_col]
  )
  coverage_rate = coverage_mask.mean() * 100 if len(valid_df) else float("nan")
  n_obs = len(valid_df)

  true_val_disp = (
      f"{true_series.iloc[0]:.2f}"
      if true_series.nunique() == 1
      else f"{true_series.mean():.2f}"
  )

  return pd.DataFrame({
      "Metric": [
          "Model",
          "Focal Parameter",
          "True Value",
          "N Observations",
          "Mean Estimate",
          "Mean Bias",
          "RMSE",
          "Coverage Rate %",
          "Mean 95% CI",
          "Mean CI Width",
      ],
      "Value": [
          model_type,
          focal_param,
          true_val_disp,
          f"{n_obs} (of {n_total} total; {n_dropped} dropped for NaN CI)",
          f"{est_series.mean():.2f} ± {est_series.std():.2f}",
          f"{bias_series.mean():.2f} ± {bias_series.std():.2f}",
          f"{rmse_val:.2f}",
          f"{coverage_rate:.2f}%",
          f"[{ci_low_mean:.2f}, {ci_high_mean:.2f}]",
          f"{ci_widths.mean():.2f} ± {ci_widths.std():.2f}",
      ],
  })


def generate_all_foci_summary(
    kde_df, params=("X0", "Y0", "X1", "Y1"), model_type="Weighted KDE",
    est_prefix="KDE_peak",
):
  """Combines summary tables for all foci coordinates into a single comparative DataFrame."""
  tables = [
      create_summary_table_focus(
          p, kde_df, model_type=model_type, est_prefix=est_prefix
      ).set_index("Metric")
      for p in params
  ]
  combined_df = pd.concat(tables, axis=1)
  combined_df.columns = [f"Param_{p}" for p in params]
  return combined_df.reset_index()


def severity2proportion(x):
    """Convert severity to proportion [0,1]"""
    if pd.isna(x):
        return np.nan
    return np.clip(x / 100, 0.0, 1.0)


if __name__ == "__main__":

    os.makedirs("./output", exist_ok=True)

    n_sims = 1
    nrows = 4
    
    sims = []
    rows = []
    trueX0s = []
    trueY0s = []
    trueX1s = []
    trueY1s = []
    
    kde_peakX0s = []
    kde_peakY0s = []
    kde_peakX1s = []
    kde_peakY1s = []
    
    ci_x0_lows, ci_x0_highs = [], []
    ci_x1_lows, ci_x1_highs = [], []
    ci_y0_lows, ci_y0_highs = [], []
    ci_y1_lows, ci_y1_highs = [], []
    ci_n_valids = []
    
    for sim in range(n_sims):
        df = pd.read_csv('disease_severity_of_year_2023_3rd_survey_twofoci.csv')
        df['Severity'] = df['Severity_Count'].apply(severity2proportion) 
        
        select_rows = [7, 9, 14, 17]
        df = df[df['Row'].isin(select_rows)]
        
        df['X'] = df['InterrowDistance']
        df['Y'] = df['Distance']
        for row in select_rows:
            row_df = df[df['Row']==row]
            foci_df = row_df[row_df['Plant']=='focus']

            true_x0 = foci_df['X'].values[0]
            true_y0 = foci_df['Y'].values[0]
    
            true_x1 = foci_df['X'].values[1]
            true_y1 = foci_df['Y'].values[1]
                
            buffer = 0
            n_points = 100
    
            try:
                kde_peak = twofoci_kde(row_df, sim, buffer, n_points, enable_plot=True)
            except Exception as e:
                print(f"KDE failed for simulation {sim} row {row}: {e}")
                continue
    
            "check and avoid switch foci labels"
            candidate_peakX0, candidate_peakY0 = kde_peak[0]
            candidate_peakX1, candidate_peakY1 = kde_peak[1]
    
            ci_x0, ci_y0, ci_x1, ci_y1, n_valid_ci = bootstrap_peak_ci_kde(
                row_df, sim, kde_peak, n_boot=300, buffer=buffer,
                n_points=n_points, random_state=sim * nrows + row,
            )
            
            # NEW ASSIGNMENT LOGIC: Evaluate total distance of both configurations
            dist_config1 = (math.dist((candidate_peakX0, candidate_peakY0), (true_x0, true_y0)) + 
                            math.dist((candidate_peakX1, candidate_peakY1), (true_x1, true_y1)))
                            
            dist_config2 = (math.dist((candidate_peakX0, candidate_peakY0), (true_x1, true_y1)) + 
                            math.dist((candidate_peakX1, candidate_peakY1), (true_x0, true_y0)))
            
            # Choose the configuration that minimizes total error
            if dist_config1 <= dist_config2:
                peakX0, peakY0, ci_0 = candidate_peakX0, candidate_peakY0, (ci_x0, ci_y0)
                peakX1, peakY1, ci_1 = candidate_peakX1, candidate_peakY1, (ci_x1, ci_y1)
            else:
                peakX0, peakY0, ci_0 = candidate_peakX1, candidate_peakY1, (ci_x1, ci_y1)
                peakX1, peakY1, ci_1 = candidate_peakX0, candidate_peakY0, (ci_x0, ci_y0)
            
                    
            sims.append(sim)
            rows.append(row)
            trueX0s.append(true_x0)
            trueY0s.append(true_y0)
            trueX1s.append(true_x1)
            trueY1s.append(true_y1)
            kde_peakX0s.append(peakX0)
            kde_peakY0s.append(peakY0)
            kde_peakX1s.append(peakX1)
            kde_peakY1s.append(peakY1)
    
            ci_x0_lows.append(ci_0[0][0])
            ci_x0_highs.append(ci_0[0][1])
            ci_y0_lows.append(ci_0[1][0])
            ci_y0_highs.append(ci_0[1][1])
            ci_x1_lows.append(ci_1[0][0])
            ci_x1_highs.append(ci_1[0][1])
            ci_y1_lows.append(ci_1[1][0])
            ci_y1_highs.append(ci_1[1][1])
            ci_n_valids.append(n_valid_ci)
            
    kde_focus_est = pd.DataFrame(data={'simulation':sims, 'row':rows, 'trueX0':trueX0s, 'trueY0':trueY0s,
                                       'trueX1':trueX1s, 'trueY1':trueY1s, 'KDE_peakX0':kde_peakX0s,
                                       'KDE_peakY0':kde_peakY0s, 'KDE_peakX1':kde_peakX1s, 'KDE_peakY1':kde_peakY1s,
                                       'CIx0_Lower':ci_x0_lows, 'CIx0_Upper':ci_x0_highs,
                                       'CIy0_Lower':ci_y0_lows, 'CIy0_Upper':ci_y0_highs,
                                       'CIx1_Lower':ci_x1_lows, 'CIx1_Upper':ci_x1_highs,
                                       'CIy1_Lower':ci_y1_lows, 'CIy1_Upper':ci_y1_highs,
                                       'CI_n_valid_boot':ci_n_valids})        
            
    # The permutation-matching step above (dist_config1 vs dist_config2) already
    # resolves label-switching for both the 1D and 2D cases uniformly -- it picks
    # whichever global assignment of the two candidates to the two true foci
    # minimizes total distance, which is the exact (and here, exactly solvable,
    # since there are only 2 candidates) bipartite matching this note was asking
    # for. By the time KDE_peakX0/Y0/X1/Y1 are set, they're already matched to
    # the nearer true focus -- the bias calculation below is safe to use directly
    # for the 2D case too, not just 1D.
    kde_focus_est['coordX0_bias'] = kde_focus_est['KDE_peakX0'] - kde_focus_est['trueX0']
    kde_focus_est['coordY0_bias'] = kde_focus_est['KDE_peakY0'] - kde_focus_est['trueY0']
    kde_focus_est['coordX1_bias'] = kde_focus_est['KDE_peakX1'] - kde_focus_est['trueX1']
    kde_focus_est['coordY1_bias'] = kde_focus_est['KDE_peakY1'] - kde_focus_est['trueY1'] 
    
    
    # Calculate Euclidean Distance between True Focus and GLM Predicted Focus
    kde_focus_est['KDE0_Distance_Error'] = np.sqrt(
        (kde_focus_est['trueX0'] - kde_focus_est['KDE_peakX0'])**2 + 
        (kde_focus_est['trueY0'] - kde_focus_est['KDE_peakY0'])**2
    )
    
    kde_focus_est['KDE1_Distance_Error'] = np.sqrt(
        (kde_focus_est['trueX1'] - kde_focus_est['KDE_peakX1'])**2 + 
        (kde_focus_est['trueY1'] - kde_focus_est['KDE_peakY1'])**2
    )
    
    # Export raw estimates
    kde_focus_est.to_csv('1dKDE_estimated_simulated_twofoci.csv', index=False, encoding='utf-8-sig')
    print("Raw estimates saved to '1dKDE_estimated_simulated_twofoci.csv'.")
    
    # Generate & Print Statistical Summary Tables
    print("\n" + "=" * 55)
    print(" STATISTICAL SUMMARY: FOCUS 0 (Y-Coordinate) ")
    print("=" * 55)
    summary_y0 = create_summary_table_focus("Y0", kde_focus_est, model_type="Weighted KDE")
    print(summary_y0.to_string(index=False))
    
    print("\n" + "=" * 55)
    print(" COMBINED COMPARATIVE SUMMARY FOR ALL FOCI PARAMS ")
    print("=" * 55)
    all_foci_summary = generate_all_foci_summary(
        kde_focus_est, params=("X0", "Y0", "X1", "Y1")
    )
    print(all_foci_summary.to_string(index=False))
    
    # Export metrics summary table
    all_foci_summary.to_csv("KDE_Summary_Metrics.csv", index=False, encoding="utf-8-sig")
    print(
        "\nComparative statistical metrics exported to"
        " 'KDE_Summary_Metrics.csv'."
    )
