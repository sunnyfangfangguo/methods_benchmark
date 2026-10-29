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
    
    if model_type == 'power_law':
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


# =============================================================================
# MAIN RUNTIME
# =============================================================================
if __name__ == "__main__":
    sns.set_style("whitegrid")
    
    "Simulation Parameters"
    eps = 1e-08
    model_type = 'power_law'
    fy1 = 10.0
    fy2 = 35.0
    
    # Focus 1 (Bottom, Y=0)
    scale1, exponent1 = 5.0, 2.0
    focus1 = [0.0, fy1, 1.0]  # [fx, fy, fz]
    weight = 0.5
    
    # Focus 2 (Top, Y=50)
    scale2, exponent2 = 5.0, 2.0
    focus2 = [0.0, fy2, 1.0] # [fx, fy, fz]
    
    "Store the areas"
    sim_list = []
    ambiguous = []
    clears = []
    total_areas = []
    ambiguous_portions = []
    clears_portions = []


    sims = 100 # all sampled locations are the same
    sim = 0

    simulated_df = pd.read_csv(f"./sim/simulated_disease_data_{sim}.csv")
    
    x_coords = np.unique(simulated_df['InterrowDistance'].values)  
    y_coords = np.unique(simulated_df['Distance'].values)          
    
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
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    
    # Define shared color scale boundaries and contour levels
    vmin_val, vmax_val = -25.0, 25.0
    shared_levels = np.linspace(vmin_val, vmax_val, 41)
    
    # -------------------------------------------------------------------------
    "PANEL A: Relative SIR Field (Unified Scale: -15 to 25 dB)"
    # -------------------------------------------------------------------------
    c1 = axs[0].contourf(X, Y, SIR2_dB, levels=shared_levels, cmap='coolwarm', 
                         vmin=vmin_val, vmax=vmax_val, extend='both')
    cbar1 = fig.colorbar(c1, ax=axs[0])
    cbar1.set_label('$\mathrm{SIR}_{2}$', fontsize=12)
    
    # 0 dB Contour Line
    b_contour = axs[0].contour(X, Y, SIR2_dB, levels=[0], colors='black', linewidths=2.0, linestyles='--')
    axs[0].clabel(b_contour, fmt='0 Boundary', fontsize=9)
    
    axs[0].scatter([focus1[0], focus2[0]], [focus1[1], focus2[1]], c='yellow', marker='*', s=200, edgecolor='black', zorder=5)
    # axs[0].set_title("A. Relative SIR Field ($\mathrm{SIR}_{2}$)\n[Focus 1 (<0) vs Focus 2 (>0)]", fontsize=11, fontweight='bold')
    axs[0].set_xlabel("Distance (X)")
    axs[0].set_ylabel("Distance (Y)")
    
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
    c2 = axs[1].contourf(X, Y, mask_grid, levels=[-0.5, 0.5, 1.5], cmap=cmap_mask2) # mask_grid only has 0 and 1, they will fall into the 2-class contour
    
    # 4. Colorbar with exact 2-class labels
    cbar2 = fig.colorbar(c2, ax=axs[1], ticks=[0, 1])
    cbar2.ax.set_yticklabels(['Ambiguous\n (0-3)', 'Clear\n (>3)'])
    
    # 5. Overlay 0 dB and 3 dB decision boundary contours
    b_contour_0 = axs[1].contour(X, Y, SIR_dom_dB, levels=[0], colors='black', linewidths=1.8, linestyles='--')
    axs[1].clabel(b_contour_0, fmt='0 Midpoint', fontsize=9)
    
    # b_contour_3 = axs[1].contour(X, Y, SIR_dom_dB, levels=[3.0], colors='white', linewidths=1.2, linestyles=':')
    # axs[1].clabel(b_contour_3, fmt='3 Boundary', fontsize=8)
    
    axs[1].scatter([focus1[0], focus2[0]], [focus1[1], focus2[1]], c='white', marker='*', s=200, edgecolor='black', zorder=5)
    # axs[1].set_title("B. Demixing Difficulty Regions (N=2)\n[Ambiguous (0-3 dB) vs. Clear (>3 dB)]", fontsize=11, fontweight='bold')
    axs[1].set_xlabel("Distance (X)")
    axs[1].set_ylabel("Distance (Y)")
    
    plt.tight_layout()
    plt.savefig(f"SIR_landscape_fy1_{fy1}_fy2_{fy2}_weight_{weight}_sim{sim}.png", dpi=600, bbox_inches='tight')
    plt.show()
    
    
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
    print(f"Simulation: {sim}")
    print(f"Ambiguous Region Area: {area_ambiguous:.2f} spatial units² ({area_ambiguous/total_landscape_area:.1%})")
    print(f"Clear Region Area:     {area_clear:.2f} spatial units² ({area_clear/total_landscape_area:.1%})")
    
    
    sim_list.append(sim)
    ambiguous.append(area_ambiguous)
    clears.append(area_clear)
    total_areas.append(total_landscape_area)
    ambiguous_portions.append(area_ambiguous/total_landscape_area)
    clears_portions.append(area_clear/total_landscape_area)
    
    area_proportion_df = pd.DataFrame(data={'Simulation':sim_list, 'fy1':fy1, 'fy2':fy2,
                                            'weight':weight, 'Ambiguous Region Area':ambiguous,
                                    'Clear Region Area':clears, 'Total Landscape Area':total_areas,
                                    'area_ambiguous/total_landscape_area':ambiguous_portions,
                                    'area_clear/total_landscape_area':clears_portions
                                    })
    
    area_proportion_df.to_csv('area_proportions.csv')            
        
