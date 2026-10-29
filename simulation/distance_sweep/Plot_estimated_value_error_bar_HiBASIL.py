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
        elinewidth=0.9, capsize=2.5, label='Estimation with 95% HDI',
    )
    # true-value reference: a horizontal line segment spanning each distance
    # category's width, at the true value's height -- reads more clearly as
    # "the target for this whole group" than a single point marker, and is
    # more robust to being visually lost under overlapping data points
    true_by_dist = df.drop_duplicates('distance').set_index('distance')[true_col]
    half_width = 0.4
    for d in distances_sorted:
        x0, x1 = cat_pos[d] - half_width, cat_pos[d] + half_width
        ax.plot([x0, x1], [true_by_dist[d]] * 2, color='orange', linewidth=2.5,
                solid_capstyle='butt', zorder=5,
                label='True value' if d == distances_sorted[0] else None)
    ax.set_xticks(range(len(distances_sorted)))
    ax.set_xticklabels([str(d) for d in distances_sorted], fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel('Distance', fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    # ax.set_title(title)
    ax.legend(fontsize=20, loc='best')




df = pd.read_excel('HiBASIL_estimation_summary.xlsx', sheet_name='Sheet1')
df.columns = [c.strip() for c in df.columns]

distances_sorted = sorted(df['distance'].unique())
cat_pos = {d: i for i, d in enumerate(distances_sorted)}
df['xcat'] = df['distance'].map(cat_pos)

sns.set_theme(style="ticks", palette="pastel")


fig, axes = plt.subplots(4, 1, figsize=(20, 16))

add_panel(axes[0], df, distances_sorted, cat_pos,
          'trueX1', 'fx1_post_mean', 'fx1_hdi_lower', 'fx1_hdi_upper',
          'Focus 1 -- X coordinate',
          r'Estimated $X_1$')
add_panel(axes[1], df, distances_sorted, cat_pos,
          'trueX2', 'fx2_post_mean', 'fx2_hdi_lower', 'fx2_hdi_upper',
          'Focus 2 -- X coordinate',
          r'Estimated $X_2$')
add_panel(axes[2], df, distances_sorted, cat_pos,
          'trueY1', 'fy1_post_mean', 'fy1_hdi_lower', 'fy1_hdi_upper',
          'Focus 1 -- Y coordinate',
          r'Estimated $Y_1$')
add_panel(axes[3], df, distances_sorted, cat_pos,
          'trueY2', 'fy2_post_mean', 'fy2_hdi_lower', 'fy2_hdi_upper',
          'Focus 2 -- Y coordinate',
          r'Estimated $Y_2$')

# fig.suptitle('HiBASIL fitted vs. true focus coordinates across distance sweep', fontsize=15)
plt.tight_layout()
plt.savefig('HiBASIL_fitted_vs_true_focus_coords.png', dpi=600, bbox_inches='tight')


