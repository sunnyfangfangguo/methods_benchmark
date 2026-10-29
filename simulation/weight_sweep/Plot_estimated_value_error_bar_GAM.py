# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:10:21 2026

@author: sunny
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def add_panel(ax, df, distances_sorted, cat_pos, true_col, fit_col, lo_col, hi_col,
              title, ylabel):
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.32, 0.32, size=len(df))
    yerr_lo = (df[fit_col] - df[lo_col]).clip(lower=0)
    yerr_hi = (df[hi_col] - df[fit_col]).clip(lower=0)
    ax.errorbar(
        df['xcat'] + jitter, df[fit_col],
        yerr=[yerr_lo, yerr_hi],
        fmt='ko', ecolor='steelblue', alpha=0.85, markersize=6,
        elinewidth=0.9, capsize=2.5, label='Estimation with 95% CI',
    )
    # true-value reference: a horizontal line segment spanning each distance
    # category's width, at the true value's height -- reads more clearly as
    # "the target for this whole group" than a single point marker, and is
    # more robust to being visually lost under overlapping data points
    true_by_weight = df.drop_duplicates('weight').set_index('weight')[true_col]
    half_width = 0.4
    for d in distances_sorted:
        x0, x1 = cat_pos[d] - half_width, cat_pos[d] + half_width
        ax.plot([x0, x1], [true_by_weight[d]] * 2, color='orange', linewidth=2.5,
                solid_capstyle='butt', zorder=5,
                label='True value' if d == distances_sorted[0] else None)
    ax.set_xticks(range(len(distances_sorted)))
    ax.set_xticklabels([str(d) for d in distances_sorted], fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel('Weight', fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    # ax.set_title(title)
    ax.legend(fontsize=20, loc='best')



df = pd.read_excel('SpatialGAM_estimation_summary_weight_1.xlsx', sheet_name='Sheet1')
df.columns = [c.strip() for c in df.columns]

weights_sorted = sorted(df['weight'].unique())
cat_pos = {d: i for i, d in enumerate(weights_sorted)}
df['xcat'] = df['weight'].map(cat_pos)

sns.set_theme(style="ticks", palette="pastel")


fig, axes = plt.subplots(4, 1, figsize=(20, 16))

add_panel(axes[0], df, weights_sorted, cat_pos,
          'trueX0', 'GAM_peakX0', 'CIx0_Lower', 'CIx0_Upper',
          'Focus 0 -- X coordinate',
          r'Estimated $X_1$')
add_panel(axes[1], df, weights_sorted, cat_pos,
          'trueX1', 'GAM_peakX1', 'CIx1_Lower', 'CIx1_Upper',
          'Focus 1 -- X coordinate',
          r'Estimated $X_2$')
add_panel(axes[2], df, weights_sorted, cat_pos,
          'trueY0', 'GAM_peakY0', 'CIy0_Lower', 'CIy0_Upper',
          'Focus 0 -- Y coordinate',
          r'Estimated $Y_1$')
add_panel(axes[3], df, weights_sorted, cat_pos,
          'trueY1', 'GAM_peakY1', 'CIy1_Lower', 'CIy1_Upper',
          'Focus 1 -- Y coordinate',
          r'Estimated $Y_2$')

# fig.suptitle('SpatialGAM fitted vs. true focus coordinates across weight sweep', fontsize=15)
plt.tight_layout()
plt.savefig('SpatialGAM_fitted_vs_true_focus_coords_weight_1.png', dpi=600, bbox_inches='tight')


