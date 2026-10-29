# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:36:30 2026

@author: sunny
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



hibasil_df = pd.read_csv("power_law_all_metrics_df_summary.csv")

df1 = pd.DataFrame(
    {
        "method": "HiBASIL",
        "true_Y1": 0,
        "true_Y2": 50,
        "est_Y1": hibasil_df["fy1_post_mean"],
        "CI_Y1_lower": hibasil_df["fy1_hdi_lower"],
        "CI_Y1_upper": hibasil_df["fy1_hdi_upper"],
        "est_Y2": hibasil_df["fy2_post_mean"],
        "CI_Y2_lower": hibasil_df["fy2_hdi_lower"],
        "CI_Y2_upper": hibasil_df["fy2_hdi_upper"],
    }
)


gam_df = pd.read_csv('SpatialGAM_estimated_simulated_twofoci.csv')

df2 = pd.DataFrame(
    {
        "method": "Spatial GAM",
        "true_Y1": gam_df["trueY0"],
        "true_Y2": gam_df["trueY1"],
        "est_Y1": gam_df["GAM_peakY0"],
        "CI_Y1_lower": gam_df["CIy0_Lower"],
        "CI_Y1_upper": gam_df["CIy0_Upper"],
        "est_Y2": gam_df["GAM_peakY1"],
        "CI_Y2_lower": gam_df["CIy1_Lower"],
        "CI_Y2_upper": gam_df["CIy1_Upper"],
    }
)


kde_df = pd.read_csv('1dKDE_estimated_simulated_twofoci.csv')

df3 = pd.DataFrame(
    {
        "method": "Weighted KDE",
        "true_Y1": kde_df["trueY0"],
        "true_Y2": kde_df["trueY1"],
        "est_Y1": kde_df["KDE_peakY0"],
        "CI_Y1_lower": kde_df["CIy0_Lower"],
        "CI_Y1_upper": kde_df["CIy0_Upper"],
        "est_Y2": kde_df["KDE_peakY1"],
        "CI_Y2_lower": kde_df["CIy1_Lower"],
        "CI_Y2_upper": kde_df["CIy1_Upper"],
    }
)

combine_df = pd.concat([df1, df2, df3], ignore_index=True)                                   

sns.set_theme(style="ticks")


def add_panel(ax, df, methods_sorted, cat_pos, true_col,
    fit_col, lo_col, hi_col, title, ylabel):
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.32, 0.32, size=len(df))

    # Compute lower and upper error bounds
    yerr_lo = (df[fit_col] - df[lo_col]).clip(lower=0)
    yerr_hi = (df[hi_col] - df[fit_col]).clip(lower=0)

    # 1. Plot estimation scatter with 95% CI error bars
    ax.errorbar(
        df["xcat"] + jitter,
        df[fit_col],
        yerr=[yerr_lo, yerr_hi],
        fmt="ko",
        ecolor="steelblue",
        alpha=0.85,
        markersize=6,
        elinewidth=0.9,
        capsize=2.5,
        label="Estimation with 95% CI",
    )

    # 2. Draw horizontal line segment spanning each method category for the true value
    true_by_method = df.groupby("method")[true_col].first()
    half_width = 0.4

    for i, m in enumerate(methods_sorted):
        x0, x1 = cat_pos[m] - half_width, cat_pos[m] + half_width
        ax.plot(
            [x0, x1],
            [true_by_method[m]] * 2,
            color="orange",
            linewidth=2.0,
            solid_capstyle="butt",
            zorder=5,
            label="True value" if i == 0 else None,
        )

    # Axes and labeling
    ax.set_xticks(range(len(methods_sorted)))
    ax.set_xticklabels(methods_sorted, fontsize=14, fontweight="bold")
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel("Method", fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")

    # Deduplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        fontsize=14,
        loc="best",
        frameon=True,
        facecolor="white",
    )


# --- Execution Loop ---

# Map method categories to numeric positions (xcat)
methods_sorted = list(combine_df["method"].unique())
cat_pos = {m: i for i, m in enumerate(methods_sorted)}
combine_df["xcat"] = combine_df["method"].map(cat_pos)

# Create 4 panels for parameters X1, X2, Y1, Y2
fig, axes = plt.subplots(2, 1, figsize=(10,8), sharex=True)

params = [
    ("Y1", "Distance (Y1)"),
    ("Y2", "Distance (Y2)"),
]

for ax, (param, label) in zip(axes, params):
    add_panel(
        ax=ax,
        df=combine_df,
        methods_sorted=methods_sorted,
        cat_pos=cat_pos,
        true_col=f"true_{param}",
        fit_col=f"est_{param}",
        lo_col=f"CI_{param}_lower",
        hi_col=f"CI_{param}_upper",
        title=f"Estimation for {param}",
        ylabel=label,
    )
    
axes[0].text(-0.10, 1.05, 'A', transform=axes[0].transAxes, 
        fontsize=20, fontweight='bold', verticalalignment='top')

axes[1].text(-0.10, 1.05, 'B', transform=axes[1].transAxes, 
        fontsize=20, fontweight='bold', verticalalignment='top')

plt.tight_layout()
plt.savefig('Methods_performance_on_1D_data.png', dpi=600, bbox_inches='tight')

