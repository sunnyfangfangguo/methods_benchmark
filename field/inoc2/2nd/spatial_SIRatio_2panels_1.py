# -*- coding: utf-8 -*-
"""
2D Spatial Landscape & SIR Boundary - Unified Color Scale [-15, 25]
Sets identical colorbar range (vmin=-15, vmax=25) for Panels A and B.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# =============================================================================
# SIMULATION KERNEL & SAMPLING FUNCTIONS
# =============================================================================



def two_foci_kernel(x, y, focus1, focus2, scale1, scale2, weight, model_type="power_law", exponent1=None, exponent2=None):
    """Unpacks foci, calculates distances, and returns S1, S2, and combined kernel."""
    fx1, fy1, fz1 = focus1
    fx2, fy2, fz2 = focus2
    
    distance1 = np.sqrt((x - fx1)**2 + (y - fy1)**2)
    distance2 = np.sqrt((x - fx2)**2 + (y - fy2)**2)
    
    "dispersal kernel functions with shared parameters"
    if model_type == 'exponential':
        kernel1 = fz1 * np.exp(-distance1 / scale1)
        kernel2 = fz2 * np.exp(-distance2 / scale2)
        
    elif model_type == 'gaussian':
        kernel1 = fz1 * np.exp(-(distance1**2) / (2 * scale1**2))
        kernel2 = fz2 * np.exp(-(distance2**2) / (2 * scale2**2))
    
    elif model_type == 'power_law':
        if (exponent1 is None) or (exponent2 is None):
            raise ValueError("exponent must be provided for power_law model")            
        kernel1 = fz1 / np.power(1 + distance1 / scale1, exponent1)
        kernel2 = fz2 / np.power(1 + distance2 / scale2, exponent2)
    else:
        raise NotImplementedError("Model type not implemented")
        
    S1 = weight * kernel1
    S2 = (1.0 - weight) * kernel2
    combined_kernel = S1 + S2
        
    return S1, S2, combined_kernel


def sampling_2d(sample_points, prob_zero, prob_one, prob_continuous, alpha_vals, beta_vals): 
    """Zero-Inflated Beta sampling logic across 2D coordinates."""
    y_observed = np.zeros(len(sample_points))
    for i in range(len(sample_points)):
        probs = np.array([prob_zero[i], prob_one[i], prob_continuous[i]])
        probs /= probs.sum()
        component = np.random.choice([0, 1, 2], p=probs)
        if component == 0:
            y_observed[i] = 0.0
        elif component == 1:
            y_observed[i] = 1.0
        else:
            y_observed[i] = np.random.beta(alpha_vals[i], beta_vals[i])
    
    df = pd.DataFrame({
        'X': sample_points[:, 0],
        'Y': sample_points[:, 1],
        'Severity': y_observed
    })
    return df


def get_cell_widths(coords):
    midpoints = (coords[:-1] + coords[1:]) / 2.0
    left_edge = coords[0] - (coords[1] - coords[0]) / 2.0
    right_edge = coords[-1] + (coords[-1] - coords[-2]) / 2.0
    edges = np.concatenate([[left_edge], midpoints, [right_edge]])
    return np.diff(edges)



sns.set_style("whitegrid")

"survey parameters"
survey = '2nd'
model_type = 'power_law'
pooling = 'partial'

fx1s = [0, 0, 0, 0]
fx2s = [0, 0, 0, 0]
fy1s = [0, 0, 0, 0]
fy2s = [25.6032, 23.7744, 26.2128, 26.2128]



n_rows = 4

eps = 1e-08

"get parameter values"
if pooling == 'partial':        
    est_df = pd.read_csv(f"./known_{pooling}_pooling/output/{model_type}_trace_summary.csv", index_col=0)

elif pooling == 'complete':
    est_df = pd.read_csv(f"./known/output/{model_type}_trace_summary.csv", index_col=0)

q_param = est_df.at['q_param','mean']
pi0 = est_df.at['pi0','mean']
phi = est_df.at['phi','mean']
weight = est_df.at['weight','mean']

if model_type == 'power_law':
    exponent1 = est_df.at['exponent1','mean']
    exponent2 = est_df.at['exponent2','mean']

"unknown foci estimation"
unknown_model_performance = pd.read_csv("./unknown/output/models_performances.csv", index_col=0)
unknown_model_type = unknown_model_performance.loc['rank_loo'].astype(float).idxmin()
est_unknown_df = pd.read_csv(f"./unknown/output/{unknown_model_type}_trace_summary.csv", index_col=0)

"calculate indices and plot"

"Store the areas"
row_list = []
ambiguous = []
clears = []
total_areas = []
ambiguous_portions = []
clears_portions = []
est_fx1s = []
est_fx2s = []
est_fy1s = []
est_fy2s = []

true_fx1s = []
true_fx2s = []
true_fy1s = []
true_fy2s = []

coverage_fx1s = []
coverage_fx2s = []
coverage_fy1s = []
coverage_fy2s = []

bias_fx1s = []
bias_fx2s = []
bias_fy1s = []
bias_fy2s = []

sd_fx1s = []
sd_fx2s = []
sd_fy1s = []
sd_fy2s = []

hdi_lower_fx1s = []
hdi_lower_fx2s = []
hdi_lower_fy1s = []
hdi_lower_fy2s = []

hdi_upper_fx1s = []
hdi_upper_fx2s = []
hdi_upper_fy1s = []
hdi_upper_fy2s = []

hdi_width_fx1s = []
hdi_width_fx2s = []
hdi_width_fy1s = []
hdi_width_fy2s = []


fig, axs = plt.subplots(2, n_rows, figsize=(24, 13))
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

for i in range(n_rows): 
    # Focus 1 (Bottom, Y=0)
    if pooling == 'partial':
        scale1 = est_df.at[f'scale1[{i}]','mean']
        scale2 = est_df.at[f'scale2[{i}]','mean']            
    elif pooling == 'complete':
        scale1 = est_df.at['scale1','mean']
        scale2 = est_df.at['scale2','mean']
        
    if model_type == 'power_law':
        exponent1 = est_df.at['exponent1','mean']
        exponent2 = est_df.at['exponent2','mean']
        
    elif model_type in ['exponential', 'gaussian']:
        exponent1=None
        exponent2=None
        
    fz1 = est_df.at[f'fz1[{i}]','mean']
    fz2 = est_df.at[f'fz2[{i}]','mean']
    
    est_fx1 = est_unknown_df.at['fx1','mean']
    est_fx2 = est_unknown_df.at['fx2','mean']
    est_fy1 = est_unknown_df.at['fy1','mean']
    est_fy2 = est_unknown_df.at['fy2','mean']
    
    sd_fx1 = est_unknown_df.at['fx1','sd']
    sd_fx2 = est_unknown_df.at['fx2','sd']
    sd_fy1 = est_unknown_df.at['fy1','sd']
    sd_fy2 = est_unknown_df.at['fy2','sd']
    
    hdi_lower_fx1 = est_unknown_df.at['fx1','hdi_2.5%']
    hdi_lower_fx2 = est_unknown_df.at['fx2','hdi_2.5%']
    hdi_lower_fy1 = est_unknown_df.at['fy1','hdi_2.5%']
    hdi_lower_fy2 = est_unknown_df.at['fy2','hdi_2.5%']
    
    hdi_upper_fx1 = est_unknown_df.at['fx1','hdi_97.5%']
    hdi_upper_fx2 = est_unknown_df.at['fx2','hdi_97.5%']
    hdi_upper_fy1 = est_unknown_df.at['fy1','hdi_97.5%']
    hdi_upper_fy2 = est_unknown_df.at['fy2','hdi_97.5%']
    
    hdi_width_fx1 = est_unknown_df.at['fx1','hdi_97.5%'] - est_unknown_df.at['fx1','hdi_2.5%']
    hdi_width_fx2 = est_unknown_df.at['fx2','hdi_97.5%'] - est_unknown_df.at['fx2','hdi_2.5%']
    hdi_width_fy1 = est_unknown_df.at['fy1','hdi_97.5%'] - est_unknown_df.at['fy1','hdi_2.5%']
    hdi_width_fy2 = est_unknown_df.at['fy2','hdi_97.5%'] - est_unknown_df.at['fy2','hdi_2.5%']
    

    # Focus 1
    fx1 = fx1s[i]
    fy1 = fy1s[i]
    focus1 = [fx1, fy1, fz1]  # [fx, fy, fz]

    # Focus 2 (Top, Y=50)
    fx2 = fx2s[i]
    fy2 = fy2s[i]
    focus2 = [fx2, fy2, fz2] # [fx, fy, fz]
    
    bias_fx1 = (est_fx1 - fx1) # true value is 0
    bias_fx2 = (est_fx2 - fx2) # true value is 0
    bias_fy1 = (est_fy1 - fy1) # true value is 0
    bias_fy2 = (est_fy2 - fy2)
    
    
    coverage_fx1 = fx1 in pd.Interval(left=est_unknown_df.at['fx1','hdi_2.5%'], right=est_unknown_df.at['fx1','hdi_97.5%'], closed='both')
    coverage_fx2 = fx2 in pd.Interval(left=est_unknown_df.at['fx2','hdi_2.5%'], right=est_unknown_df.at['fx2','hdi_97.5%'], closed='both')
    coverage_fy1 = fy1 in pd.Interval(left=est_unknown_df.at['fy1','hdi_2.5%'], right=est_unknown_df.at['fy1','hdi_97.5%'], closed='both')
    coverage_fy2 = fy2 in pd.Interval(left=est_unknown_df.at['fy2','hdi_2.5%'], right=est_unknown_df.at['fy2','hdi_97.5%'], closed='both')    
    
    
    
    "plot area for each row"
    x_coords = np.linspace(-5, 5, num=50)  
    y_coords = np.linspace(-10,35, num=50)          
    
    X, Y = np.meshgrid(x_coords, y_coords)
    
    S1_grid, S2_grid, mu_grid = two_foci_kernel(
        X, Y, focus1, focus2, scale1, scale2, weight, 
        model_type=model_type, exponent1=exponent1, exponent2=exponent2
    )
    
    "Relative SIR Field (Focus 2 vs Focus 1) in dB"
    SIR2_dB = 10.0 * np.log10((S2_grid + eps) / (S1_grid + eps))
    
    "Dominant SIR Field (Signal Clarity) in dB"
    SIR_dom_dB = np.abs(SIR2_dB) # simple way to calculate dominance in two-foci scenario
    
    "The definition way to calculate the dominant SIR field"
    # SIR_dom_dB = 10.0 * np.log10((np.maximum(S2_grid,S1_grid) + eps) / (np.minimum(S1_grid, S2_grid) + eps))
    
    # -------------------------------------------------------------------------
    "Figure Layout with Shared Limits [-15, 25]"
    # -------------------------------------------------------------------------
    
    
    # Define shared color scale boundaries and contour levels
    vmin_val, vmax_val = -25.0, 25.0
    shared_levels = np.linspace(vmin_val, vmax_val, 41)
    
    # -------------------------------------------------------------------------
    "PANEL A: Relative SIR Field (Unified Scale: -15 to 25 dB)"
    # -------------------------------------------------------------------------
    c1 = axs[0][i].contourf(X, Y, SIR2_dB, levels=shared_levels, cmap='coolwarm', 
                         vmin=vmin_val, vmax=vmax_val, extend='both')
    cbar1 = fig.colorbar(c1, ax=axs[0][i])
    cbar1.set_label('$\mathrm{SIR}_{2}$', fontsize=18)
    
    # 0 dB Contour Line
    b_contour = axs[0][i].contour(X, Y, SIR2_dB, levels=[0], colors='black', linewidths=2.0, linestyles='--')
    axs[0][i].clabel(b_contour, fmt='0 Boundary', fontsize=16)
    
    axs[0][i].scatter([focus1[0], focus2[0]], [focus1[1], focus2[1]], c='white', marker='o', s=120, edgecolor='black', zorder=5, label='true focus')
    # axs[0][i].scatter([est_fx1, est_fx2], [est_fy1, est_fy2], c='yellow', marker='*', s=200, edgecolor='black', zorder=5, label='HiBASIL estimated focus')
    # axs[0][i].set_title("A. Relative SIR Field ($\mathrm{SIR}_{2, \mathrm{dB}}$)\n[Focus 1 (<0dB) vs Focus 2 (>0dB)]", fontsize=11, fontweight='bold')
    axs[0][i].set_xlabel("Distance (X)", fontsize=20)
    axs[0][i].set_ylabel("Distance (Y)", fontsize=20)
    # axs[0][i].legend(loc='best')
    # # -------------------------------------------------------------------------
    # "PANEL B: Dominant SIR Field (Unified Scale: -15 to 25 dB)"
    # # -------------------------------------------------------------------------
    # c2 = axs[1].contourf(X, Y, SIR_dom_dB, levels=shared_levels, cmap='coolwarm', 
    #                      vmin=vmin_val, vmax=vmax_val, extend='both')
    # cbar2 = fig.colorbar(c2, ax=axs[1])
    # cbar2.set_label('$\mathrm{SIR}_{\mathrm{dom, dB}}$ (dB)', fontsize=10)
    
    # # 0 dB Valley Trough
    # trough_contour = axs[1].contour(X, Y, SIR_dom_dB, levels=[0.5], colors='black', linewidths=2.0, linestyles=':')
    # axs[1].clabel(trough_contour, fmt='0 dB Trench', fontsize=9)
    
    # axs[1].scatter([focus1[0], focus2[0]], [focus1[1], focus2[1]], c='yellow', marker='*', s=200, edgecolor='black', zorder=5)
    # axs[1].set_title("B. Dominant SIR Field ($\mathrm{SIR}_{\mathrm{dom, dB}}$)\n[Unified Scale: -15 to 25 dB]", fontsize=11, fontweight='bold')
    # axs[1].set_xlabel("X")
    # axs[1].set_ylabel("Y")
    
    
    # -------------------------------------------------------------------------
    "PANEL C: 2-Class Demixing Mask for N=2 Foci"
    # -------------------------------------------------------------------------
    
    # 1. Classify grid into 2 real zones for N=2:
    # Class 0: Ambiguous (0 to 3 dB)
    # Class 1: Clear (> 3 dB)
    mask_grid = np.zeros_like(SIR_dom_dB)
    mask_grid[SIR_dom_dB >= 3.0] = 1.0  # only has 0 and 1
    
    
    # 2. Discrete 2-color palette (Orange for Ambiguous, Blue for Clear)
    cmap_mask2 = ListedColormap(['#fdae61', '#2b83ba'])
    
    # 3. Plot 2-class contours
    c2 = axs[1][i].contourf(X, Y, mask_grid, levels=[-0.5, 0.5, 1.5], cmap=cmap_mask2) # mask_grid only has 0 and 1, they will fall into the 2-class contour
    
    # 4. Colorbar with exact 2-class labels
    cbar2 = fig.colorbar(c2, ax=axs[1][i], ticks=[0, 1])
    cbar2.ax.set_yticklabels(['Ambiguous\n (0-3)', 'Clear\n (>3)'])
    
    # 5. Overlay 0 dB and 3 dB decision boundary contours
    b_contour_0 = axs[1][i].contour(X, Y, SIR_dom_dB, levels=[0], colors='black', linewidths=1.8, linestyles='--')
    axs[1][i].clabel(b_contour_0, fmt='0 Midpoint', fontsize=9)
    
    # b_contour_3 = axs[1][i].contour(X, Y, SIR_dom_dB, levels=[3.0], colors='white', linewidths=1.2, linestyles=':')
    # axs[1][i].clabel(b_contour_3, fmt='3 Boundary', fontsize=8)
    
    axs[1][i].scatter([focus1[0], focus2[0]], [focus1[1], focus2[1]], c='white', marker='o', s=120, edgecolor='black', zorder=5, label='true focus')
    # axs[1][i].scatter([est_fx1, est_fx2], [est_fy1, est_fy2], c='yellow', marker='*', s=200, edgecolor='black', zorder=5, label='HiBASIL estimated focus')
    # axs[1][i].set_title("B. Demixing Difficulty Regions (N=2)\n[Ambiguous (0-3 dB) vs. Clear (>3 dB)]", fontsize=11, fontweight='bold')
    axs[1][i].set_xlabel("Distance (X)", fontsize=20)
    axs[1][i].set_ylabel("Distance (Y)", fontsize=20)
    # axs[1][i].legend(loc='best') 
    
    "calculate area within the contour lines"   
    # Calculate cell widths for X (cols) and Y (rows)
    dx = get_cell_widths(x_coords)  # size 101
    dy = get_cell_widths(y_coords)  # size 408
    
    # Outer product: dy (rows) x dx (cols) -> shape (408, 101)
    cell_area_matrix = np.outer(dy, dx)
    
    # Integrate physical spatial area
    ambiguous_mask = (SIR_dom_dB >= 0.0) & (SIR_dom_dB < 3.0)
    clear_mask = SIR_dom_dB >= 3.0
    
    area_ambiguous = np.sum(cell_area_matrix[ambiguous_mask])
    area_clear = np.sum(cell_area_matrix[clear_mask])
    total_landscape_area = np.sum(cell_area_matrix)
    print(f"Ambiguous Region Area: {area_ambiguous:.2f} spatial units² ({area_ambiguous/total_landscape_area:.1%})")
    print(f"Clear Region Area:     {area_clear:.2f} spatial units² ({area_clear/total_landscape_area:.1%})")

    row_list.append(i)
    ambiguous.append(area_ambiguous)
    clears.append(area_clear)
    total_areas.append(total_landscape_area)
    ambiguous_portions.append(area_ambiguous/total_landscape_area)
    clears_portions.append(area_clear/total_landscape_area)
    est_fx1s.append(est_fx1)
    est_fx2s.append(est_fx2)
    est_fy1s.append(est_fy1)
    est_fy2s.append(est_fy2)
    
    true_fx1s.append(fx1)
    true_fx2s.append(fx2)
    true_fy1s.append(fy1)
    true_fy2s.append(fy2)
    
    coverage_fx1s.append(coverage_fx1)
    coverage_fx2s.append(coverage_fx2)
    coverage_fy1s.append(coverage_fy1)
    coverage_fy2s.append(coverage_fy2)

    bias_fx1s.append(bias_fx1)
    bias_fx2s.append(bias_fx2)
    bias_fy1s.append(bias_fy1)
    bias_fy2s.append(bias_fy2)
    
    sd_fx1s.append(sd_fx1)
    sd_fx2s.append(sd_fx2)
    sd_fy1s.append(sd_fy1)
    sd_fy2s.append(sd_fy2)
    
    hdi_lower_fx1s.append(hdi_lower_fx1)
    hdi_lower_fx2s.append(hdi_lower_fx2)
    hdi_lower_fy1s.append(hdi_lower_fy1)
    hdi_lower_fy2s.append(hdi_lower_fy2)
    
    hdi_upper_fx1s.append(hdi_upper_fx1)
    hdi_upper_fx2s.append(hdi_upper_fx2)
    hdi_upper_fy1s.append(hdi_upper_fy1)
    hdi_upper_fy2s.append(hdi_upper_fy2)

    hdi_width_fx1s.append(hdi_width_fx1)
    hdi_width_fx2s.append(hdi_width_fx2)
    hdi_width_fy1s.append(hdi_width_fy1)
    hdi_width_fy2s.append(hdi_width_fy2)
    
    
        
plt.tight_layout()
plt.savefig(f"Inoc2_SIR_landscape_survey_{survey}_pooling_{pooling}_{model_type}_1.png", dpi=900, bbox_inches='tight')
plt.show()

area_proportion_df = pd.DataFrame(data={'Survey':survey, 'Pooling':pooling, 'Model':model_type,
                                        'Row':row_list, 'known weight':weight, 'true fx1':true_fx1s, 'true fx2':true_fx2s, 
                                        'true fy1':true_fy1s, 'true fy2':true_fy2s,
                                        'estimated fx1':est_fx1, 'estimated fx2':est_fx2, 
                                        'estimated fy1':est_fy1, 'estimated fy2':est_fy2,
                                        'bias fx1':bias_fx1s, 'bias fx2':bias_fx2s,
                                        'bias fy1':bias_fy1s, 'bias fy2':bias_fy2s,
                                        'coverage fx1':coverage_fx1s, 'coverage fx2':coverage_fx2s,
                                        'coverage fy1':coverage_fy1s, 'coverage fy2':coverage_fy2s,
                                        'sd fx1':sd_fx1s, 'sd fx2':sd_fx2s, 'sd fy1':sd_fy1s, 'sd fy2':sd_fy2s,
                                        'hdi_lower fx1':hdi_lower_fx1s, 'hdi_lower fx2':hdi_lower_fx2s, 
                                        'hdi_lower fy1':hdi_lower_fy1s, 'hdi_lower fy2':hdi_lower_fy2s,
                                        'hdi_upper fx1':hdi_upper_fx1s, 'hdi_upper fx2':hdi_upper_fx2s, 
                                        'hdi_upper fy1':hdi_upper_fy1s, 'hdi_upper fy2':hdi_upper_fy2s,                                        
                                        'hdi_width fx1':hdi_width_fx1s, 'hdi_width fx2':hdi_width_fx2s, 
                                        'hdi_width fy1':hdi_width_fy1s, 'hdi_width fy2':hdi_width_fy2s,
                                         'Ambiguous Region Area':ambiguous,
                                'Clear Region Area':clears, 'Total Landscape Area':total_areas,
                                'area_ambiguous/total_landscape_area':ambiguous_portions,
                                'area_clear/total_landscape_area':clears_portions
                                })


area_proportion_df.to_csv(f'Inoc2_area_proportions_survey_{survey}_pooling_{pooling}_{model_type}_1.csv')            
    
