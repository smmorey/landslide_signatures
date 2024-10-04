#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-10-04

@title: Periodic Landslide Model
@description: This script contains a function to run a Landlab landscape evolution model with periodic landslides.
@version: 1.0
@author: Susannah Morey
@contact: susannah.morey@gmail.com
@license: -----
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from landlab.utils import get_watershed_masks
import metric_library as metric
import watershed_metric_calcs as wmc
from landlab.io.netcdf import write_netcdf
import pickle


def run_model_with_periodic_landslides(scen_num, mg_name, total_time, time_step, rows, columns, node_spacing,
                          eroder_component, uplift, bedrock_coeff, sed_space_coeff,
                          plucking_coeff, sed_gbe_coeff, attrition_coeff, br_abrasion_coeff,
                          phi, cohesion, int_fric_ang, landslides, ls_return_time, ls_thresh, ls_recur_int, 
                          coarse_fractions=0.05, plotting="False", plotting_interval=1000,
                          new_grid="True", is_new_run="True", save_plots="False",
                          output_format='netcdf', output_interval=100):
    """
    Run the landscape evolution model and save outputs at specified intervals.
    
    Parameters:
    -----------
    scen_num            : int ; Scenario number.
    mg_name             : RasterModelGrid ; The model grid to run the model on.
    total_time          : float; Total time to run the model for.
    time_step           : float ; Timestep for the model.
    rows                : int ; Number of rows in the model grid.
    columns             : int ; Number of columns in the model grid.
    node_spacing        : float ; Spacing between nodes in the model grid.
    eroder_component    : str ; Erosion component to use. Either 'space' or 'abrasion'.
    uplift              : float ; Uplift rate for the model.
    bedrock_coeff       : float ; Bedrock erodibility coefficient.    
    sed_space_coeff     : float ; Sediment erodibility coefficient for the space component.
    plucking_coeff      : float ; Plucking coefficient for the abrasion component.
    sed_gbe_coeff       : float ; Sediment transport coefficient for the abrasion component.
    attrition_coeff     : float ; Attrition coefficient for the abrasion component.
    br_abrasion_coeff   : float ; Bedrock abrasion coefficient for the abrasion component.
    phi                 : float ; Friction angle for the landslides component.
    cohesion            : float ;  Cohesion for the landslides component.
    int_fric_ang        : float ; Internal friction angle for the landslides component.   
    landslides          : str ; Whether to include landslides in the model.
    ls_return_time      : float ; Return time for landslides - this is from Campforts et al. (2022).
    ls_thresh           : float  ; Threshold slope for landslides.
    ls_recur_int        : float ; Time between landslide events - new from SMM Sept 2024.
    coarse_fractions    : float ; Fraction of coarse material in the model.
    plotting            : str ; Whether to plot the model output.
    plotting_interval   : int ; Number of timesteps between each plot.
    new_grid            : str ; Whether to create a new grid for the model.
    is_new_run          : str ; Whether this is a new run of the model.
    save_plots          : str ; Whether to save the plots.
    output_format       : str ; Format to save the output. Either 'netcdf' or 'pickle'.
    output_interval     : int ; Number of timesteps between each save of the grid state.
    
    Returns:
    mg_name : RasterModelGrid ; The final state of the model grid.
    """
    
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import os
    
    # landlab things
    from landlab import RasterModelGrid, HexModelGrid
    from landlab import imshowhs_grid, imshow_grid
    from landlab.components import (PriorityFloodFlowRouter,
                                    ExponentialWeatherer,
                                    DepthDependentTaylorDiffuser, 
                                    SpaceLargeScaleEroder, 
                                    GravelBedrockEroder,
                                    BedrockLandslider,
                                    SteepnessFinder,
                                    ChiFinder,
                                    ChannelProfiler,
                                    )
    from landlab.io.netcdf import write_netcdf, read_netcdf
    import pickle

    # Setup output directory
    output_dir = f'scenario_{scen_num}_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    fr = PriorityFloodFlowRouter(
        mg_name,
        surface="topographic__elevation",
        flow_metric="D8",
        suppress_out=True,
        depression_handler="fill",
        accumulate_flow=True,
        separate_hill_flow=True,
        accumulate_flow_hill=True,
    )
    fr.run_one_step()
    
    ew = ExponentialWeatherer(
        mg_name,
        soil_production_maximum_rate=3e-4,
        soil_production_decay_depth=0.44,
    )
    
    ddTd = DepthDependentTaylorDiffuser(
        mg_name,
        soil_transport_decay_depth=0.1,
        slope_crit=int_fric_ang,
        nterms=2,
        soil_transport_velocity=0.01,
        dynamic_dt=True,
        if_unstable="raise",
        courant_factor=0.9
    )
    
    if eroder_component == 'space':
        eroder = SpaceLargeScaleEroder(mg_name, K_sed=sed_space_coeff, K_br=bedrock_coeff)
    elif eroder_component == 'abrasion':
        eroder = GravelBedrockEroder(
            mg_name, 
            intermittency_factor=0.01, 
            sediment_porosity=phi,
            number_of_sediment_classes=1,
            plucking_coefficient=plucking_coeff, 
            transport_coefficient=sed_gbe_coeff,
            abrasion_coefficients=attrition_coeff,
            bedrock_abrasion_coefficient=br_abrasion_coeff,
            coarse_fractions_from_plucking=coarse_fractions,
        )

    hy = BedrockLandslider(
        mg_name, 
        angle_int_frict=int_fric_ang, 
        threshold_slope=ls_thresh, 
        cohesion_eff=cohesion,
        landslides_return_time=ls_return_time,
        landslides_on_boundary_nodes=False,
        phi=0.3, 
        fraction_fines_LS=0.5,
    )

    # Time loop setup
    total_years = 0
    grid_states = []
    start_time = time.time()
   
    # Setup output directory
    output_dir = f'scenario_{scen_num}_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    fr = PriorityFloodFlowRouter(
        mg_name,
        surface="topographic__elevation",
        flow_metric="D8",
        suppress_out=True,
        depression_handler="fill",
        accumulate_flow=True,
        separate_hill_flow=True,
        accumulate_flow_hill=True,
    )
    
    ew = ExponentialWeatherer(
        mg_name,
        soil_production_maximum_rate=3e-4,
        soil_production_decay_depth=0.44,
    )
    
    ddTd = DepthDependentTaylorDiffuser(
        mg_name,
        soil_transport_decay_depth=0.1,
        slope_crit=int_fric_ang,
        nterms=2,
        soil_transport_velocity=0.01,
        dynamic_dt=True,
        if_unstable="raise",
        courant_factor=0.9
    )
    
    if eroder_component == 'space':
        eroder = SpaceLargeScaleEroder(mg_name, K_sed=sed_space_coeff, K_br=bedrock_coeff)
    elif eroder_component == 'abrasion':
        eroder = GravelBedrockEroder(
            mg_name, 
            intermittency_factor=0.01, 
            sediment_porosity=phi,
            number_of_sediment_classes=1,
            plucking_coefficient=plucking_coeff, 
            transport_coefficient=sed_gbe_coeff,
            abrasion_coefficients=attrition_coeff,
            bedrock_abrasion_coefficient=br_abrasion_coeff,
            coarse_fractions_from_plucking=coarse_fractions,
        )

    hy = BedrockLandslider(
        mg_name, 
        angle_int_frict=int_fric_ang, 
        threshold_slope=ls_thresh, 
        cohesion_eff=cohesion,
        landslides_return_time=ls_return_time,
        landslides_on_boundary_nodes=False,
        phi=0.3, 
        fraction_fines_LS=0.5,
    )

    # Prepare data structure for output
    grid_states = []

    # Progress tracking setup
    start_time = time.time()
    
    def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
        if iteration == total: 
            print()

    def run_model_step(total_time, time_step, landslides):
        ndt = int(total_time // time_step)
        uplift_per_step = uplift * time_step
        for _ in range(ndt):
            mg_name.at_node['bedrock__elevation'][mg_name.core_nodes] += uplift_per_step
            mg_name.at_node['topographic__elevation'][:] = (mg_name.at_node["bedrock__elevation"] + mg_name.at_node["soil__depth"])
            ew.run_one_step()
            ddTd.run_one_step(time_step)
            fr.run_one_step()
            eroder.run_one_step(time_step)
            if landslides:
                hy.run_one_step(time_step)

    # Time loop
    current_time = 0
    while current_time < total_time:
        # Determine if this is a landslide year
        is_landslide_year = (current_time % ls_recur_int == 0)
        
        if is_landslide_year:
            # Run for 1 year with landslides
            run_model_step(1, 1, True)
            current_time += 1
        else:
            # Run for time_step years without landslides
            run_time = min(time_step, ls_recur_int - (current_time % ls_recur_int))
            run_model_step(run_time, time_step, False)
            current_time += run_time

        # Save output at specified intervals
        if current_time % output_interval == 0 or current_time >= total_time:
            grid_states.append({
                'time': current_time,
                'topographic__elevation': mg_name.at_node["topographic__elevation"].copy(),
                'soil__depth': mg_name.at_node["soil__depth"].copy(),
                'bedrock__elevation': mg_name.at_node["bedrock__elevation"].copy(),
                'drainage_area': mg_name.at_node["drainage_area"].copy(),
                'topographic__steepest_slope': mg_name.at_node["topographic__steepest_slope"].copy(),
                'is_landslide_year': is_landslide_year
            })

        # Update progress bar
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time * total_time / current_time
        time_remaining = estimated_total_time - elapsed_time
        
        print_progress_bar(current_time, total_time, 
                           prefix=f'Progress:', 
                           suffix=f'Complete. Time remaining: {time.strftime("%H:%M:%S", time.gmtime(time_remaining))}', 
                           length=50)

    # Save the final output file
    output_filename = f'{output_dir}/grid_states_periodic_landslides.pkl'
    with open(output_filename, 'wb') as f:
        pickle.dump(grid_states, f)

    print(f"\nSaved all grid states to {output_filename}")
    print(f"Total runtime: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

    return mg_name

# %% Example of how to run the model

mg_v32_a = read_netcdf("/Users/csdmsuser/Documents/Research/CU/GBE/Magnitude_Frequency_Experiments/grid_gbe_eq_midU.nc") 

scenario_num = 32
recurrence_interval = 'a'   # select recurrence interval
timestep = 10               # fixed timestep to use in years
total_time = 10000          # total simulation time in years

internal_angle_of_friction = 0.85
ls_return_time = 50

dir_path = f'/Users/csdmsuser/Documents/Research/CU/GBE/Magnitude_Frequency_Experiments/scenario{scenario_num}/{recurrence_interval}/'
os.chdir(dir_path)

recurrence_intervals = [50, 60, 70, 80, 90, 100, 250, 500, 1000]
if recurrence_interval == 'a':
    interval = recurrence_intervals[0]
elif recurrence_interval == 'b':
    interval = recurrence_intervals[1]
elif recurrence_interval == 'c':
    interval = recurrence_intervals[2]
elif recurrence_interval == 'd':
    interval = recurrence_intervals[3]
elif recurrence_interval == 'e':
    interval = recurrence_intervals[4]
elif recurrence_interval == 'f':
    interval = recurrence_intervals[5]
pass
    
# Calculate the numver of timesteps per recurrence interval
steps_per_interval = interval // timestep

# Calculate total number of full recurrence intervals
total_steps = int(total_time/interval)

mg_v32_a = run_model_with_periodic_landslides(
    scen_num=scenario_num,
    mg_name=mg_v32_a,
    total_time=total_time,
    time_step=timestep,
    rows=100, columns=100, node_spacing=30,
    eroder_component='abrasion',
    uplift=uplift_rates[3],
    bedrock_coeff=1.5e-5,
    sed_space_coeff=1e-5,
    plucking_coeff=3e-4,
    sed_gbe_coeff=0.041,
    attrition_coeff=0.005,
    br_abrasion_coeff=0.005,
    phi=0.1,
    cohesion=1e4,
    int_fric_ang=internal_angle_of_friction,
    ls_return_time=ls_return_time,  # This is used in the BedrockLandslider component
    ls_thresh=internal_angle_of_friction,
    ls_recur_int=interval,  # This is the parameter for landslide recurrence
    plotting=False,
    plotting_interval=10,
    new_grid=False,
    save_plots=False,
    output_format='pickle',
    output_interval=timestep,
    )