# -*- coding: utf-8 -*-
"""Spatial mgcv Two-Foci Simulation Pipeline with Statistical Summary Metrics.

This script calls R's mgcv library via rpy2 to fit Thin Plate Regression 
Splines (bs="tp") with REML penalty optimization, extracts peak coordinates 
and confidence intervals, and exports summary metrics.
"""
import os
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from skimage.feature import peak_local_max
import seaborn as sns
from scipy.interpolate import RegularGridInterpolator

# Interface with R's mgcv package via rpy2 using localconverter
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


mgcv = importr('mgcv')
stats = importr('stats')
# New dependencies for the ZOIB fix: run in R first if not already installed:
#   install.packages(c('gamlss', 'gamlss.dist', 'gamlss.add'))
gamlss = importr('gamlss')
gamlss_dist = importr('gamlss.dist')
gamlss_add = importr('gamlss.add')  # provides ga(), the mgcv-smooth bridge


# ==============================================================================
# 1. SPATIAL GAM & CONFIDENCE INTERVAL FUNCTIONS
# ==============================================================================
def plot_gam_twofoci(df, grid_coords, Z, peak0, peak1, sim, axis_name="Y"):
  """Plots predicted spatial GAM risk curves against true disease severity stem plot."""
  sns.set_style(style="white")
  fig, ax1 = plt.subplots(figsize=(10, 6))

  (line1,) = ax1.plot(
      grid_coords, Z, color="green", label="mgcv Predicted Risk", linewidth=2
  )
  ax1.set_xlabel(f"Distance ({axis_name})")
  ax1.set_ylabel("Predicted Probability")
  ax1.set_ylim(-0.05, 1.05)

  vline1 = ax1.axvline(
      x=peak0,
      color="red",
      linestyle="--",
      alpha=0.8,
      label=f"GAM Focus 1 ({peak0:.2f})",
  )
  vline2 = ax1.axvline(
      x=peak1,
      color="purple",
      linestyle="--",
      alpha=0.8,
      label=f"GAM Focus 2 ({peak1:.2f})",
  )

  ax2 = ax1.twinx()
  markerline, stemlines, _ = ax2.stem(
      df[axis_name],
      df["Severity"],
      linefmt="orange",
      markerfmt="o",
      label="Observed Severity",
      basefmt=" ",
  )
  markerline.set_label("Observed Severity")
  plt.setp(markerline, "color", "orange", "markersize", 6)

  ax2.set_ylabel("Observed Severity")
  ax2.set_ylim(-0.05, 1.05)

  row = df["Row"].iloc[0] if "Row" in df.columns else 0
  plt.title(f"Comparison: mgcv Prediction vs. Observed Severity (Row {row})")

  lines = [line1, vline1, vline2, markerline]
  labels = [l.get_label() for l in lines]
  ax1.legend(lines, labels, loc="upper left")

  plt.tight_layout()
  plt.savefig(f"./output/compare_GAM_pred_real_severity_sim{sim}_row{row}.png")
  plt.close()


def plot_gam_twofoci_2d(df, sx, sy, Z_2d, peak0, peak1, sim):
  """Plots 2D predicted spatial GAM risk heatmap overlaid with actual plant severity and predicted foci coordinates."""
  sns.set_style("white")
  fig, ax = plt.subplots(figsize=(9, 7))

  SX, SY = np.meshgrid(sx, sy)
  contour = ax.contourf(
      SX, SY, Z_2d, levels=25, cmap="YlOrRd", alpha=0.85, vmin=0, vmax=1
  )
  cbar = fig.colorbar(contour, ax=ax)
  cbar.set_label("mgcv Predicted Probability")

  scatter = ax.scatter(
      df["X"],
      df["Y"],
      c=df["Severity"],
      cmap="Blues",
      edgecolors="black",
      linewidth=0.5,
      s=50,
      label="Observed Severity",
  )

  ax.scatter(
      [peak0[0]],
      [peak0[1]],
      color="lime",
      edgecolors="black",
      marker="*",
      s=200,
      zorder=5,
      label=f"GAM Focus 1 ({peak0[0]:.1f}, {peak0[1]:.1f})",
  )
  ax.scatter(
      [peak1[0]],
      [peak1[1]],
      color="cyan",
      edgecolors="black",
      marker="*",
      s=200,
      zorder=5,
      label=f"GAM Focus 2 ({peak1[0]:.1f}, {peak1[1]:.1f})",
  )

  row = df["Row"].iloc[0] if "Row" in df.columns else 0
  ax.set_xlabel("X (Interrow Distance)")
  ax.set_ylabel("Y (Distance)")
  ax.set_title(f"2D Isotropic mgcv Predicted Risk Surface vs. Observed Severity (Row {row})")
  ax.legend(loc="upper right", framealpha=0.9)

  plt.tight_layout()
  plt.savefig(
      f"./output/compare_GAM_pred_real_severity_2D_sim{sim}_row{row}.png"
  )
  plt.close()


def _twofoci_result(peaks, model, fit_error=None):
  """Assembles twofoci_spatialGAM's return dict."""
  return {"peaks": peaks, "model": model, "fit_error": fit_error}


def twofoci_spatialGAM(
    df, sim, scale_factor=1.0, buffer=0, n_points=100, enable_plot=False,
    x_mean=None, y_mean=None, x_lim=None, y_lim=None):

  """Fits an R gamlss ZOIB (BEINF family) GAM via rpy2 and returns spatial risk peaks."""
  if x_mean is None: x_mean = df["X"].mean()
  if y_mean is None: y_mean = df["Y"].mean()
  
  # Use fixed limits if provided, otherwise compute from local df
  x_min = x_lim[0] if x_lim is not None else (df["X"].min() - buffer)
  x_max = x_lim[1] if x_lim is not None else (df["X"].max() + buffer)
  y_min = y_lim[0] if y_lim is not None else (df["Y"].min() - buffer)
  y_max = y_lim[1] if y_lim is not None else (df["Y"].max() + buffer)

  df = df.copy()
  df["Xs"] = (df["X"] - x_mean) / scale_factor
  df["Ys"] = (df["Y"] - y_mean) / scale_factor

  x_constant = np.all(df["X"].values == df["X"].values[0])
  y_constant = np.all(df["Y"].values == df["Y"].values[0])

  x_val = df["X"].iloc[0] if x_constant else df["X"].mean()
  y_val = df["Y"].iloc[0] if y_constant else df["Y"].mean()

  if x_constant and y_constant:
    p = (x_val, y_val)
    return _twofoci_result((p, p), None)

  # Modern rpy2 converter context manager
  fit_df = df[["Xs", "Ys", "Severity"]].dropna().copy()
  with localconverter(robjects.default_converter + pandas2ri.converter):
    r_df = robjects.conversion.py2rpy(fit_df)

  try:
    if x_constant:
      r_formula = robjects.Formula('Severity ~ ga(~s(Ys, bs="ps", k=4))') #8
    elif y_constant:
      r_formula = robjects.Formula('Severity ~ ga(~s(Xs, bs="ps", k=4))')
    else:
      # Same te() spatial term as the betar() version (k=c(10,10) chosen
      # there after k=c(5,5) was confirmed too restrictive for Ys to show
      # two bumps), now nested inside ga() so gamlss can use it as the
      # mu formula. sigma/nu/tau formulas are left as ~1 (intercept-only,
      # not spatially varying) -- deliberately: with ~100 observations per
      # row, giving all four BEINF parameters their own spatial smooth
      # risks a much harder fit to converge than what's already been
      # struggled with. Since nu/tau are spatially constant, mu's peak
      # locations and the overall expected severity's peak locations
      # coincide, so predicting mu alone is enough for peak-finding --
      # no need to separately combine mu/sigma/nu/tau via meanBEINF().
      r_formula = robjects.Formula(
          'Severity ~ ga(~te(Xs, Ys, bs="tp", k=c(5,5)))'
      )

    # BEINF (not BEZI): BEZI structurally can't represent Y=1 at all, but
    # real data does contain it -- row 3, sim 0 has Severity=1.0 exactly
    # (the focus plant itself). BEZI would give that observation zero
    # likelihood, same problem tau=0 would have caused.
    #
    # tau.fix=True, tau.start=0.0101: freely-estimated tau was the actual
    # convergence problem (only 3/408 exact ones in the real data, often
    # 0-1 per row -- too little to estimate a smoothing parameter from).
    # Fixing tau at a small nonzero value keeps "Y=1 is possible" in the
    # model, unlike tau=0, while stopping the optimizer from chasing an
    # unidentifiable estimate.
    #
    # tau is a ratio (tau = p1/p2), not P(Y=1) directly, where
    # p2 = P(0<Y<1). The simulation's own point-mass parameters are known
    # exactly: Liu & Kong's ZOIB parameterizes p=P(Y=0) directly and
    # q=P(Y=1|Y>0) (confirmed against the zoib package's own docs, which
    # link x0 to Pr(y=0) and x1 to Pr(y=1|y>0) specifically -- not
    # P(Y=1|Y in {0,1}), a plausible-looking alternative reading that
    # would have given a different number). With p=0.3, q=0.01:
    #   P(Y=0)   = p           = 0.3
    #   P(Y=1)   = (1-p)*q     = 0.7*0.01  = 0.007
    #   P(0<Y<1) = (1-p)*(1-q) = 0.7*0.99  = 0.693
    # matching the observed sample closely (30.1%/0.74%/69.1%), a good
    # sign this reading is right. tau = P(Y=1)/P(0<Y<1) = 0.007/0.693
    # = 0.0101. Still worth cross-checking against gamlss's own BEINF
    # parameterization directly if possible, since this combines a
    # confirmed formula (zoib's own docs) with an unverified one
    # (gamlss's tau = p1/p2, taken from memory, not confirmed the same
    # way).
    model = gamlss.gamlss(
        r_formula,
        sigma_formula=robjects.Formula('~1'),
        nu_formula=robjects.Formula('~1'),
        tau_formula=robjects.Formula('~1'),
        tau_fix=True,
        tau_start=robjects.FloatVector([1e-04]),
        family=gamlss_dist.BEINF(),
        data=r_df,
        # Hard cap, not a convergence target: without this, a single
        # bootstrap resample that oscillates instead of converging can run
        # indefinitely -- costly given this fit runs inside a loop of up
        # to 1000 per row. n.cyc=20 matches what most rows already
        # converged within in practice; rows that hit the cap without
        # converging are still usable results, just less precise ones.
        control=gamlss.gamlss_control(n_cyc=100, trace=False),
    )

  except Exception as e:
    print(f"gamlss failed to fit: {e}")
    p = (x_val, y_val)
    return _twofoci_result((p, p), None, fit_error=str(e))

  if x_constant:
    sy = np.linspace(y_min, y_max, n_points)
    sy_scaled = (sy - y_mean) / scale_factor

    grid_df = pd.DataFrame({"Ys": sy_scaled})
    with localconverter(robjects.default_converter + pandas2ri.converter):
      r_grid = robjects.conversion.py2rpy(grid_df)

    preds = stats.predict(model, newdata=r_grid, what="mu", type="response")
    Z = np.array(preds)
    
    interior_peaks, _ = find_peaks(Z, prominence=0.001, width=1)
    peaks = list(interior_peaks)
    
    # explicitly evaluate boundary 1 (Y_min at index 0)
    bg_level = np.median(Z)
    z_range = Z.max() - Z.min()
    
    min_prominence = 0.15 * z_range
    
    if Z[0] > Z[1] and (Z[0] - bg_level) >= min_prominence:
        if 0 not in peaks:
            peaks.append(0)
    
    # explicitly evaluate boundary 2 (Y_max at last index)
    last_idx = len(Z) - 1
    if Z[-1] > Z[-2] and (Z[-1] - bg_level) >= min_prominence:
        if last_idx not in peaks:
            peaks.append(last_idx)
    
    peaks = np.array(peaks)
    valley_frac = 0.1
    
    if len(peaks) >= 2:
        peak_heights = Z[peaks]
        top_2 = np.argsort(peak_heights)[::-1][:2]
        idx0, idx1 = peaks[top_2[0]], peaks[top_2[1]]
    
        val0, val1 = Z[idx0], Z[idx1]
        seg_start, seg_end = min(idx0, idx1), max(idx0, idx1)
        segment = Z[seg_start:seg_end + 1]

        # Trim 10% from endpoints to evaluate true intervening valley
        trim = max(1, int(len(segment) * 0.10))
        if len(segment) > (2 * trim):
            valley_min = segment[trim:-trim].min()
        else:
            valley_min = segment.min()

        lower_peak = min(val0, val1)
        dip_depth = lower_peak - valley_min
        required_dip = valley_frac * (lower_peak - bg_level)
        genuine_second_peak = dip_depth >= required_dip

        if not genuine_second_peak:
            idx1 = idx0  # Collapse onto single global peak
       
    elif len(peaks) == 1:
        idx0 = idx1 = peaks[0]
        genuine_second_peak = False
    
    else:
        idx0 = idx1 = np.argmax(Z)
        genuine_second_peak = False

    peak_y0, peak_y1 = sy[idx0], sy[idx1]
    orig_peaks = ((x_val, peak_y0), (x_val, peak_y1))

    if enable_plot:
        plot_gam_twofoci(df, sy, Z, peak_y0, peak_y1, sim, axis_name="Y")

    return _twofoci_result(orig_peaks, model)

  elif y_constant:
    sx = np.linspace(x_min, x_max, n_points)
    sx_scaled = (sx - x_mean) / scale_factor

    grid_df = pd.DataFrame({"Xs": sx_scaled})
    with localconverter(robjects.default_converter + pandas2ri.converter):
      r_grid = robjects.conversion.py2rpy(grid_df)

    preds = stats.predict(model, newdata=r_grid, what="mu", type="response")
    Z = np.array(preds)

    interior_peaks, _ = find_peaks(Z, prominence=0.01, width=1)
    peaks = list(interior_peaks)
    
    # explicitly evaluate boundary 1 (X_min at index 0)
    bg_level = np.median(Z)
    z_range = Z.max() - Z.min()
    
    min_prominence = 0.15 * z_range
    
    if Z[0] > Z[1] and (Z[0] - bg_level) >= min_prominence:
        if 0 not in peaks:
            peaks.append(0)
    
    # explicitly evaluate boundary 2 (X_max at last index)
    last_idx = len(Z) - 1
    if Z[-1] > Z[-2] and (Z[-1] - bg_level) >= min_prominence:
        if last_idx not in peaks:
            peaks.append(last_idx)
    
    peaks = np.array(peaks)
    valley_frac = 0.1
    
    if len(peaks) >= 2:
        peak_heights = Z[peaks]
        top_2 = np.argsort(peak_heights)[::-1][:2]
        idx0, idx1 = peaks[top_2[0]], peaks[top_2[1]]
    
        val0, val1 = Z[idx0], Z[idx1]
        seg_start, seg_end = min(idx0, idx1), max(idx0, idx1)
        segment = Z[seg_start:seg_end + 1]

        # Trim 10% from endpoints to evaluate true intervening valley
        trim = max(1, int(len(segment) * 0.10))
        if len(segment) > (2 * trim):
            valley_min = segment[trim:-trim].min()
        else:
            valley_min = segment.min()

        lower_peak = min(val0, val1)
        dip_depth = lower_peak - valley_min
        required_dip = valley_frac * (lower_peak - bg_level)
        genuine_second_peak = dip_depth >= required_dip

        if not genuine_second_peak:
            idx1 = idx0  # Collapse onto single global peak
        
    elif len(peaks) == 1:
        idx0 = idx1 = peaks[0]
        genuine_second_peak = False
                
    else:
        idx0 = idx1 = np.argmax(Z)
        genuine_second_peak = False

    peak_x0, peak_x1 = sx[idx0], sx[idx1]
    orig_peaks = ((peak_x0, y_val), (peak_x1, y_val))

    if enable_plot:
        plot_gam_twofoci(df, sx, Z, peak_x0, peak_x1, sim, axis_name="X")

    return _twofoci_result(orig_peaks, model)

  else:
    sx = np.linspace(x_min, x_max, n_points)
    sy = np.linspace(y_min, y_max, n_points)
    SX, SY = np.meshgrid(sx, sy)

    grid_xs = (SX.ravel() - x_mean) / scale_factor
    grid_ys = (SY.ravel() - y_mean) / scale_factor

    grid_df = pd.DataFrame({"Xs": grid_xs, "Ys": grid_ys})
    with localconverter(robjects.default_converter + pandas2ri.converter):
      r_grid = robjects.conversion.py2rpy(grid_df)

    preds = stats.predict(model, newdata=r_grid, what="mu", type="response")
    Z_flat = np.array(preds)
    Z_2d = Z_flat.reshape(n_points, n_points)


    # Peak finding
    background = np.median(Z_2d)
    valley_frac = 0.1

    peak_coords = peak_local_max(
        Z_2d,
        min_distance=3, # number of spatial grid cells
        threshold_abs=background + 0.01, #baseline background filter
        exclude_border=False,
    )

    if len(peak_coords) >= 2:
      # Rank detected topological local apexes globally by fitted height
      peak_heights = Z_2d[peak_coords[:, 0], peak_coords[:, 1]]
      top2_indices = np.argsort(peak_heights)[::-1][:2]

      idx_peak0 = peak_coords[top2_indices[0]]  # Global Max (#1)
      idx_peak1 = peak_coords[top2_indices[1]]  # 2nd Tallest Global Peak (#2)

      y0_i, x0_i = idx_peak0[0], idx_peak0[1]
      y1_i, x1_i = idx_peak1[0], idx_peak1[1]

      x0v, y0v = sx[x0_i], sy[y0_i]
      x1v, y1v = sx[x1_i], sy[y1_i]
      peak0_val, peak1_val = Z_2d[y0_i, x0_i], Z_2d[y1_i, x1_i]

      # 3. Smooth bilinear interpolation along trajectory for valley detection      
      interp_func = RegularGridInterpolator((sy, sx), Z_2d, bounds_error=False, fill_value=None)

      # Sampling straight-line trajectory to check for an intervening valley
      t = np.linspace(0, 1, 50)
      line_x = x0v + t * (x1v - x0v)
      line_y = y0v + t * (y1v - y0v)
      line_points = np.column_stack([line_y, line_x])
      line_vals = interp_func(line_points)

      trim = max(1, int(len(line_vals) * 0.10))
      valley_min = line_vals[trim:-trim].min() if len(line_vals) > (2 * trim) else min(peak0_val, peak1_val)
      
      lower_peak = min(peak0_val, peak1_val)
      dip_depth = lower_peak - valley_min
      required_dip = valley_frac * (lower_peak - background)
      genuine_second_peak = dip_depth >= required_dip

      print(
          f"[valley-check] sim={sim} peak0=({x0v:.2f},{y0v:.2f}) "
          f"peak1=({x1v:.2f},{y1v:.2f}) dip_depth={dip_depth:.4f} "
          f"required={required_dip:.4f} genuine={genuine_second_peak}"
      )
    elif len(peak_coords) == 1:
      y0_i = y1_i = peak_coords[0][0]
      x0_i = x1_i = peak_coords[0][1]
      x0v = x1v = sx[x0_i]
      y0v = y1v = sy[y0_i]
      genuine_second_peak = False
    else:
      flat_idx0 = np.argmax(Z_2d)
      y0_i, x0_i = np.unravel_index(flat_idx0, Z_2d.shape)
      y1_i, x1_i = y0_i, x0_i
      x0v = x1v = sx[x0_i]
      y0v = y1v = sy[y0_i]
      genuine_second_peak = False

    if genuine_second_peak:
      peak_y_idx0, peak_x_idx0 = y0_i, x0_i
      peak_y_idx1, peak_x_idx1 = y1_i, x1_i
    else:
      peak_y_idx0 = peak_y_idx1 = y0_i
      peak_x_idx0 = peak_x_idx1 = x0_i

    peak_x0, peak_y0 = sx[peak_x_idx0], sy[peak_y_idx0]
    peak_x1, peak_y1 = sx[peak_x_idx1], sy[peak_y_idx1]
    orig_peaks = ((peak_x0, peak_y0), (peak_x1, peak_y1))

    if enable_plot:
      plot_gam_twofoci_2d(df, sx, sy, Z_2d, (peak_x0, peak_y0), (peak_x1, peak_y1), sim)

    return _twofoci_result(orig_peaks, model)


def bootstrap_peak_ci(
    df, sim, orig_peaks, n_boot=100, ci_level=0.95,
    scale_factor=1.0, buffer=0, n_points=100,
    random_state=None,
):
  """Case-resampling bootstrap CI for the two peak locations via mgcv."""
  peak0_ref, peak1_ref = orig_peaks
  if peak0_ref == peak1_ref:
    nan_ci = (np.nan, np.nan)
    return nan_ci, nan_ci, nan_ci, nan_ci, 0

  rng = np.random.default_rng(random_state)
  n = len(df)
  boot_p0, boot_p1 = [], []
  orig_x_mean, orig_y_mean = df["X"].mean(), df["Y"].mean()
  orig_x_lim = (df["X"].min() - buffer, df["X"].max() + buffer)
  orig_y_lim = (df["Y"].min() - buffer, df["Y"].max() + buffer)
  
  for _ in range(n_boot):
    boot_idx = rng.integers(0, n, size=n)
    boot_df = df.iloc[boot_idx].reset_index(drop=True)

    try:      
      result = twofoci_spatialGAM(
          boot_df, sim, scale_factor=scale_factor, buffer=buffer,
          n_points=n_points, enable_plot=False,
          x_mean=orig_x_mean, y_mean=orig_y_mean,
          x_lim=orig_x_lim, y_lim=orig_y_lim
      )
    except Exception:
      continue
    if result["model"] is None:
      continue

    cand0, cand1 = result["peaks"]
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


# ==============================================================================
# 2. STATISTICAL SUMMARY GENERATOR FUNCTIONS
# ==============================================================================

def create_summary_table_focus(
    focal_param, all_metrics_df, model_type="R mgcv GAM"
):
  est_col = f"GAM_peak{focal_param}"

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
    gam_df, params=["X0", "Y0", "X1", "Y1"], model_type="R mgcv GAM"
):
  """Combines summary tables for all foci coordinates into a single comparative DataFrame."""
  tables = [
      create_summary_table_focus(p, gam_df, model_type=model_type).set_index(
          "Metric"
      )
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


# ==============================================================================
# 3. MAIN EXECUTION PIPELINE
# ==============================================================================



os.makedirs("./output", exist_ok=True)

nrows = 4

sims, rows = [], []
trueX0s, trueY0s, gam_peakX0s, gam_peakY0s = [], [], [], []
trueX1s, trueY1s, gam_peakX1s, gam_peakY1s = [], [], [], []

ci_x0_lows, ci_x0_highs = [], []
ci_x1_lows, ci_x1_highs = [], []
ci_y0_lows, ci_y0_highs = [], []
ci_y1_lows, ci_y1_highs = [], []
ci_n_valids = []
fit_errors = []

print("Starting Spatial R mgcv Fitting via rpy2 across simulation runs...")

  
sim = 0
df = pd.read_csv("disease_severity_of_year_2023_1st_survey_twofoci.csv")
df["X"] = df["InterrowDistance"]
df["Y"] = df["Distance"]
df['Severity'] = df['Severity_Count'].apply(severity2proportion)


select_rows = [3, 10, 13, 20]
df = df[df['Row'].isin(select_rows)]


for row in select_rows:
  row_df = df[df["Row"] == row]
  foci_df = row_df[row_df["Plant"] == "focus"]

  true_x0, true_y0 = foci_df["X"].iloc[0], foci_df["Y"].iloc[0]
  true_x1, true_y1 = foci_df["X"].iloc[1], foci_df["Y"].iloc[1]

  result = twofoci_spatialGAM(
      row_df, sim, scale_factor=1.0, buffer=0, n_points=100,
      enable_plot=True,
  )
  fit_error = result["fit_error"]
  print("sim", sim)
  print("Detected Foci Pairs:", result["peaks"])
  print("Is Collapsed?:", result["peaks"][0] == result["peaks"][1])

  # Diagnostic: what does the fitted surface actually predict AT the
  # true focus locations, versus the known true mu there (~0.5 given
  # weight=0.5, fz=1.0 in the simulation, for Delta large enough that
  # the other focus's contribution is negligible)? If this comes back
  # well below 0.5 (e.g. 0.1-0.2), that's direct evidence the GAM is
  # under-fitting the true peak's sharp, narrow shape (half-max width
  # ~2 units) given how few of the ~100 observations per row actually
  # fall near it -- not a peak-finding problem, a GAM-fit problem.
  if result["model"] is not None:
    x_mean_diag = row_df["X"].mean()
    y_mean_diag = row_df["Y"].mean()
    true_check_df = pd.DataFrame({
        "Xs": [true_x0 - x_mean_diag, true_x1 - x_mean_diag],
        "Ys": [true_y0 - y_mean_diag, true_y1 - y_mean_diag],
    })
    with localconverter(robjects.default_converter + pandas2ri.converter):
      r_true_check = robjects.conversion.py2rpy(true_check_df)
    true_mu_preds = np.array(
        stats.predict(result["model"], newdata=r_true_check, what="mu", type="response")
    )
    print(
        f"[focus-fit-check] sim={sim} row={row} "
        f"fitted_mu_at_true_focus0={true_mu_preds[0]:.4f} "
        f"fitted_mu_at_true_focus1={true_mu_preds[1]:.4f} "
        f"(expect ~0.5 if the fit captured the true peak)"
    )

  candidate_peakX0, candidate_peakY0 = result["peaks"][0]
  candidate_peakX1, candidate_peakY1 = result["peaks"][1]

  if result["model"] is not None:
    ci_x0, ci_y0, ci_x1, ci_y1, n_valid_ci = bootstrap_peak_ci(
        row_df, sim, result["peaks"], n_boot=100,
        scale_factor=1.0, buffer=0, n_points=100,
        random_state=sim * nrows + row,
    )
  else:
    nan_ci = (float("nan"), float("nan"))
    ci_x0 = ci_y0 = ci_x1 = ci_y1 = nan_ci
    n_valid_ci = None

  # Permutation matching to minimize total spatial allocation distance
  dist_config1 = math.dist(
      (candidate_peakX0, candidate_peakY0), (true_x0, true_y0)
  ) + math.dist((candidate_peakX1, candidate_peakY1), (true_x1, true_y1))

  dist_config2 = math.dist(
      (candidate_peakX0, candidate_peakY0), (true_x1, true_y1)
  ) + math.dist((candidate_peakX1, candidate_peakY1), (true_x0, true_y0))

  if dist_config1 <= dist_config2:
    peakX0, peakY0, ci_0 = candidate_peakX0, candidate_peakY0, (
        ci_x0,
        ci_y0,
    )
    peakX1, peakY1, ci_1 = candidate_peakX1, candidate_peakY1, (
        ci_x1,
        ci_y1,
    )
  else:
    peakX0, peakY0, ci_0 = candidate_peakX1, candidate_peakY1, (
        ci_x1,
        ci_y1,
    )
    peakX1, peakY1, ci_1 = candidate_peakX0, candidate_peakY0, (
        ci_x0,
        ci_y0,
    )

  sims.append(sim)
  rows.append(row)
  trueX0s.append(true_x0)
  trueY0s.append(true_y0)
  gam_peakX0s.append(peakX0)
  gam_peakY0s.append(peakY0)

  trueX1s.append(true_x1)
  trueY1s.append(true_y1)
  gam_peakX1s.append(peakX1)
  gam_peakY1s.append(peakY1)

  ci_x0_lows.append(ci_0[0][0])
  ci_x0_highs.append(ci_0[0][1])
  ci_x1_lows.append(ci_1[0][0])
  ci_x1_highs.append(ci_1[0][1])

  ci_y0_lows.append(ci_0[1][0])
  ci_y0_highs.append(ci_0[1][1])
  ci_y1_lows.append(ci_1[1][0])
  ci_y1_highs.append(ci_1[1][1])

  ci_n_valids.append(n_valid_ci)
  fit_errors.append(fit_error)

  # Build aggregated DataFrame
  gam_focus_est = pd.DataFrame(
      data={
          "simulation": sims,
          "row": rows,
          "trueX0": trueX0s,
          "trueY0": trueY0s,
          "GAM_peakX0": gam_peakX0s,
          "GAM_peakY0": gam_peakY0s,
          "CIx0_Lower": ci_x0_lows,
          "CIx0_Upper": ci_x0_highs,
          "CIy0_Lower": ci_y0_lows,
          "CIy0_Upper": ci_y0_highs,
          "trueX1": trueX1s,
          "trueY1": trueY1s,
          "GAM_peakX1": gam_peakX1s,
          "GAM_peakY1": gam_peakY1s,
          "CIx1_Lower": ci_x1_lows,
          "CIx1_Upper": ci_x1_highs,
          "CIy1_Lower": ci_y1_lows,
          "CIy1_Upper": ci_y1_highs,
          "CI_n_valid_boot": ci_n_valids,
          "fit_error": fit_errors,
      }
  )

  # Distance Errors
  gam_focus_est["GAM0_Distance_Error"] = np.sqrt(
      (gam_focus_est["trueX0"] - gam_focus_est["GAM_peakX0"]) ** 2
      + (gam_focus_est["trueY0"] - gam_focus_est["GAM_peakY0"]) ** 2
  )
  gam_focus_est["GAM1_Distance_Error"] = np.sqrt(
      (gam_focus_est["trueX1"] - gam_focus_est["GAM_peakX1"]) ** 2
      + (gam_focus_est["trueY1"] - gam_focus_est["GAM_peakY1"]) ** 2
  )

  # Export raw estimates
  gam_focus_est.to_csv(
      "SpatialGAM_estimated_simulated_twofoci.csv", index=False,
      encoding="utf-8-sig",
  )
  print("Raw estimates saved to 'SpatialGAM_estimated_simulated_twofoci.csv'.")

  # Generate & Print Statistical Summary Tables
  print("\n" + "=" * 55)
  print(" STATISTICAL SUMMARY: FOCUS 0 (Y-Coordinate) ")
  print("=" * 55)
  summary_y0 = create_summary_table_focus(
      "Y0", gam_focus_est, model_type="R mgcv GAM"
  )
  print(summary_y0.to_string(index=False))

  print("\n" + "=" * 55)
  print(" COMBINED COMPARATIVE SUMMARY FOR ALL FOCI PARAMS ")
  print("=" * 55)
  all_foci_summary = generate_all_foci_summary(
      gam_focus_est, params=["X0", "Y0", "X1", "Y1"], model_type="R mgcv GAM"
  )
  print(all_foci_summary.to_string(index=False))

  # Export metrics summary table
  all_foci_summary.to_csv(
      "SpatialGAM_Summary_Metrics.csv", index=False, encoding="utf-8-sig"
  )
  print(
      "\nComparative statistical metrics exported to"
      " 'SpatialGAM_Summary_Metrics.csv'."
  )
