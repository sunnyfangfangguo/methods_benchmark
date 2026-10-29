# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 18:13:12 2026

@author: sunny
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



ambiguous_area_data = pd.read_excel("area_proportions_summary.xlsx")

sns.set_theme(style="ticks")
sns.lineplot(data=ambiguous_area_data, x='Distance', y='area_ambiguous_proportion', marker='o')
plt.ylabel("Proportion")
plt.tight_layout()
plt.savefig("ambiguous_area_proportion_distance_sweep.png", dpi=600)