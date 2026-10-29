# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 09:45:08 2025

@author: sunny
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob





def extract_metrics(df, sim, repeat, model_type, true_values_dict):
    metrics = {'simulation':sim, 'repeat':repeat, 'model':model_type}
    
    "global parameters"
    if model_type == 'power_law':
        global_params = ['scale1', 'exponent1', 'scale2', 'exponent2', 'weight', 'q_param', 'pi0', 'phi']
        
    elif model_type in ['exponential', 'gaussian']:
        global_params = ['scale1', 'scale2', 'weight', 'q_param', 'pi0', 'phi']
        
    for param in global_params:
        row = df.loc[param]
        true_val = true_values_dict.get(param)
        
        # store posterior estimates
        metrics[f'{param}_post_mean'] =  row['mean']
        metrics[f'{param}_rhat'] = row['r_hat']
        metrics[f'{param}_ess_bulk'] = row['ess_bulk']
        
        # calculate accuracy metrics
        bias = row['mean'] - true_val
        relative_err = abs(bias) / true_val * 100 if true_val != 0 else 0
        covered = row['hdi_2.5%'] <= true_val <= row['hdi_97.5%']
        
        metrics[f'{param}_bias'] = bias
        metrics[f'{param}_abs_bias'] = abs(bias)
        metrics[f'{param}_relative_error'] = relative_err
        metrics[f'{param}_covered'] = 1 if covered else 0
        metrics[f'{param}_converged'] = 1 if row['r_hat'] <= 1.01 else 0
        
        metrics[f'{param}_hdi_lower'] = row['hdi_2.5%']
        metrics[f'{param}_hdi_upper'] = row['hdi_97.5%']
        metrics[f'{param}_hdi_width'] = row['hdi_97.5%'] - row['hdi_2.5%']
        

    for param_name in ['fx1', 'fx2', 'fy1', 'fy2', 'fz1', 'fz2']:
        param_row = df.loc[f'{param_name}[{repeat}]']
        param_true = true_values_dict.get(param_name)
        
        
        # store posterior estimates
        metrics[f'{param_name}_post_mean'] = param_row['mean']
        metrics[f'{param_name}_rhat'] = param_row['r_hat']
        metrics[f'{param_name}_ess_bulk'] = param_row['ess_bulk']
        
        # calculate accuracy metrics        
        hdi_lower = param_row['hdi_2.5%']
        hdi_upper = param_row['hdi_97.5%']
        hdi_width = hdi_upper - hdi_lower
        
        bias = param_row['mean'] - param_true
        coverage = (hdi_lower <= param_true) & (param_true <= hdi_upper)
        relative_err = abs(bias) / param_true * 100 if param_true != 0 else 0
        
        # store mu posterior estimates
        metrics[f'{param_name}_bias'] = bias
        metrics[f'{param_name}_abs_bias'] = abs(bias)
        metrics[f'{param_name}_relative_error'] = relative_err
        metrics[f'{param_name}_covered'] = 1 if coverage else 0
        metrics[f'{param_name}_converged'] = 1 if param_row['r_hat'] <= 1.01 else 0
        metrics[f'{param_name}_hdi_lower'] = hdi_lower
        metrics[f'{param_name}_hdi_upper'] = hdi_upper
        metrics[f'{param_name}_hdi_width'] = hdi_width
        
    return metrics
        



def aggregate_across_sims(n_sims, n_repeats, model_type, true_values_dict):
    
    all_metrics = []
    
    for sim in range(n_sims):
        trace_summary_file = f"{output_dir}{model_type}_trace_summary_sim{sim}.csv"
            
        df = pd.read_csv(trace_summary_file, index_col=0)
        
        for repeat in range(n_repeats):
            metrics = extract_metrics(df, sim, repeat, model_type, true_values_dict)
            
            all_metrics.append(metrics)
        
    return pd.DataFrame(all_metrics)




def create_summary_table_global(all_metrics_df, n_sims, n_repeats, model_type, true_values_dict):
    "global parameters"
    if model_type == 'power_law':
        global_params = ['scale1', 'exponent1', 'scale2', 'exponent2', 'weight', 'q_param', 'pi0', 'phi']
        
    elif model_type in ['exponential', 'gaussian']:
        global_params = ['scale1', 'scale2', 'weight', 'q_param', 'pi0', 'phi']
    
    statistic_data = []
    for param in global_params:
        # true value
        true_val = true_values_dict.get(param)
        
        # convergence statistics
        n_converged = all_metrics_df[f'{param}_converged'].sum()
        n_low_ess = (all_metrics_df[f'{param}_ess_bulk'] < 400).sum()
        
        hdi_low = all_metrics_df[f'{param}_hdi_lower'].mean()
        hdi_high = all_metrics_df[f'{param}_hdi_upper'].mean()
        width_mean = all_metrics_df[f'{param}_hdi_width'].mean()
        width_std = all_metrics_df[f'{param}_hdi_width'].std()
        
        # create statistics dict
        param_summary = {
            'Model':model_type,
            'Parameter':param,
            'True Value':f'{true_val:.2f}',
            'N Simulations':n_sims,
            'N Repeats':n_repeats,
            'Mean Estimate':f"{all_metrics_df[f'{param}_post_mean'].mean():.2f} ± {all_metrics_df[f'{param}_post_mean'].std():.2f}",
            'Mean Bias':f"{all_metrics_df[f'{param}_bias'].mean():.2f} ± {all_metrics_df[f'{param}_bias'].std():.2f}",
            'Mean Abs Bias':f"{all_metrics_df[f'{param}_abs_bias'].mean():.2f}",
            'Mean Relative Error':f"{all_metrics_df[f'{param}_relative_error'].mean():.2f}",
            'Coverage Rate':f"{all_metrics_df[f'{param}_covered'].mean()*100:.2f}%",
            'Mean 95% HDI': f"[{hdi_low:.2f}, {hdi_high:.2f}]",
            'Mean HDI Width': f"{width_mean:.2f} ± {width_std:.2f}",
            'Converged':f"{(n_converged/(n_sims*n_repeats))*100:.1f}%",
            'Low ESS':f"{(n_low_ess/(n_sims*n_repeats))*100:.1f}%",
            'Mean R-hat':f"{all_metrics_df[f'{param}_rhat'].mean():.2f}",
            'Mean ESS (bulk)':f"{all_metrics_df[f'{param}_ess_bulk'].mean():.0f}"
            }
        
        statistic_data.append(param_summary)
   
    return pd.DataFrame(statistic_data)

        

def create_summary_table_focus(all_metrics_df, n_sims, n_repeats, model_type, true_values_dict):
    focal_params = ['fx1', 'fx2', 'fy1', 'fy2', 'fz1', 'fz2']
    
    statistic_data = []
    for focal_param in focal_params:    
        true_val = true_values_dict.get(focal_param)
        
        # convergence statistics
        n_converged = all_metrics_df[f'{focal_param}_converged'].sum()
        n_low_ess = (all_metrics_df[f'{focal_param}_ess_bulk'] < 400).sum()
        
        hdi_low = all_metrics_df[f'{focal_param}_hdi_lower'].mean()
        hdi_high = all_metrics_df[f'{focal_param}_hdi_upper'].mean()
        width_mean = all_metrics_df[f'{focal_param}_hdi_width'].mean()
        width_std = all_metrics_df[f'{focal_param}_hdi_width'].std()
        
        # create statistics dict
        param_summary = {
            'Model':model_type,
            'Parameter':focal_param,
            'True Value':f'{true_val:.2f}',
            'N Simulations':n_sims,
            'N Repeats':n_repeats,
            'Mean Estimate':f"{all_metrics_df[f'{focal_param}_post_mean'].mean():.2f} ± {all_metrics_df[f'{focal_param}_post_mean'].std():.2f}",
            'Mean Bias':f"{all_metrics_df[f'{focal_param}_bias'].mean():.2f} ± {all_metrics_df[f'{focal_param}_bias'].std():.2f}",
            'Mean Abs Bias':f"{all_metrics_df[f'{focal_param}_abs_bias'].mean():.2f}",
            'Mean Relative Error':f"{all_metrics_df[f'{focal_param}_relative_error'].mean():.2f}",
            'Coverage Rate':f"{all_metrics_df[f'{focal_param}_covered'].mean()*100:.2f}%",
            'Mean 95% HDI': f"[{hdi_low:.2f}, {hdi_high:.2f}]",
            'Mean HDI Width': f"{width_mean:.2f} ± {width_std:.2f}",
            'Converged':f"{(n_converged/(n_sims*n_repeats))*100:.1f}%",
            'Low ESS':f"{(n_low_ess/(n_sims*n_repeats))*100:.1f}%",
            'Mean R-hat':f"{all_metrics_df[f'{focal_param}_rhat'].mean():.2f}",
            'Mean ESS (bulk)':f"{all_metrics_df[f'{focal_param}_ess_bulk'].mean():.0f}"
            }
        
        statistic_data.append(param_summary)

    
    return pd.DataFrame(statistic_data)
        



"read trace summary"
model_types = ['exponential', 'gaussian', 'power_law']
n_sims = 100
n_repeats = 4
output_dir = './output/'

   
for model_type in model_types:

    if model_type == "power_law":
        true_values_dict = {'fx1':0, 'fy1':0, 'fz1':1.0, 'fx2':0, 'fy2':50.0, 'fz2':1.0, 'scale1':5.0, 
                            'exponent1':2.0, 'scale2':5.0, 'exponent2':2.0, 
                            'weight':0.5, 'q_param':0.01, 'pi0':0.3, 
                        'phi':20}
    elif model_type in ['exponential', 'gaussian']:
        true_values_dict = {'fx1':0, 'fy1':0, 'fz1':1.0, 'fx2':0, 'fy2':50.0, 'fz2':1.0, 'scale1':5.0, 
                            'scale2':5.0, 'weight':0.5, 'q_param':0.01, 'pi0':0.3, 
                        'phi':20}
        
    
    all_metrics_df = aggregate_across_sims(n_sims, n_repeats, model_type, true_values_dict)    
    all_metrics_df.to_csv(f"{output_dir}{model_type}_all_metrics_df_summary.csv")        
    
    global_params_summary = create_summary_table_global(all_metrics_df, n_sims, n_repeats, model_type, true_values_dict)                
    global_params_summary.to_csv(f"{output_dir}{model_type}_global_params_summary.csv", encoding='utf-8-sig')
    
    focal_params_summary = create_summary_table_focus(all_metrics_df, n_sims, n_repeats, model_type, true_values_dict)
    focal_params_summary.to_csv(f"{output_dir}{model_type}_focal_params_summary.csv", encoding='utf-8-sig')
    




      
        
        
        
