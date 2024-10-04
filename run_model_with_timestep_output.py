#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-10-04

@title: Landslide Model with Timestep Output
@description: This script contains a function to run a Landlab landscape evolution model landslides and save the model output at specified intervals.
@version: 1.0
@author: Susannah Morey
@contact: susannah.morey@gmail.com
@license: -----
"""

def run_model_with_output(scen_num, mg_name, total_time, time_step, rows, columns, node_spacing,
                          eroder_component, uplift, bedrock_coeff, sed_space_coeff,
                          plucking_coeff, sed_gbe_coeff, attrition_coeff, br_abrasion_coeff,
                          phi, cohesion, int_fric_ang, landslides, ls_return_time, ls_thresh,
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
    coarse_fractions    : float ; Fraction of coarse material in the model.
    plotting            : str ; Whether to plot the model output.
    plotting_interval   : int ; Number of timesteps between each plot.
    new_grid            : str ; Whether to create a new grid for the model.
    is_new_run          : str ; Whether this is a new run of the model.
    save_plots          : str ; Whether to save the plots.
    output_format       : str ; Format to save the output. Either 'netcdf' or 'pickle'.
    output_interval     : int ; Number of timesteps between each save of the grid state.
    
    Returns:
    -----------
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

    # Grid setup
    if new_grid == 'True':
        mg_name = RasterModelGrid((rows, columns), node_spacing)
        mg_name.axis_units = ('m', 'm')

        # Set Boundary Conditions
        mg_name.set_closed_boundaries_at_grid_edges(bottom_is_closed=False,
                                                    left_is_closed=False,
                                                    right_is_closed=False,
                                                    top_is_closed=False)
                                                    
        mg_name.add_zeros("topographic__elevation", at="node")
        mg_name.add_zeros("soil__depth", at="node")
        mg_name.add_zeros("bedrock__elevation", at="node")
        
        # Initialize topography
        np.random.seed(seed=5000)
        mg_name.at_node["topographic__elevation"] += np.random.rand(len(mg_name.node_y)) / 1000.0
        mg_name.at_node["bedrock__elevation"] = mg_name.at_node["topographic__elevation"].copy()
    else:
        mg_name = mg_name
        pass

    # Component initialization
    # Flow Router
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
    # Weatherer; values from Campforts et al. (2022)
    ew = ExponentialWeatherer(
        mg_name,
        soil_production_maximum_rate=3e-4,
        soil_production_decay_depth=0.44,
        )
    # Hillslope Diffusion; these values are from Campforts et al. (2022) and then mined from "model_basicHylands_smm.py".if not there
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
        eroder = GravelBedrockEroder(mg_name, 
                                    intermittency_factor = 0.01, 
                                    sediment_porosity = phi,
                                    number_of_sediment_classes = 1,
                                    plucking_coefficient = plucking_coeff, 
                                    transport_coefficient = sed_gbe_coeff,
                                    abrasion_coefficients = attrition_coeff,
                                    bedrock_abrasion_coefficient = br_abrasion_coeff,
                                    coarse_fractions_from_plucking = 0.05,
                                    )

    if landslides == 'True':
        hy = BedrockLandslider(mg_name, 
                                angle_int_frict=int_fric_ang, 
                                threshold_slope=ls_thresh, 
                                cohesion_eff=cohesion,
                                landslides_return_time=ls_return_time,
                                landslides_on_boundary_nodes=False,
                                phi = 0.3, 
                                fraction_fines_LS = 0.5,
                                )

    # Time loop setup
    ndt = int(total_time // time_step)
    uplift_per_step = uplift * time_step

    # Create output directory
    output_dir = f'scenario_{scen_num}_output'
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data structures for output
    if output_format == 'netcdf':
        # Initialize datasets for each variable we want to track
        ds = xr.Dataset(
            {
                "topographic__elevation": (["time", "y", "x"], np.zeros((ndt//output_interval + 1, rows, columns))),
                "soil__depth": (["time", "y", "x"], np.zeros((ndt//output_interval + 1, rows, columns))),
                "bedrock__elevation": (["time", "y", "x"], np.zeros((ndt//output_interval + 1, rows, columns))),
                "drainage_area": (["time", "y", "x"], np.zeros((ndt//output_interval + 1, rows, columns))),
                "topographic__steepest_slope": (["time", "y", "x"], np.zeros((ndt//output_interval + 1, rows, columns))),
            },
            coords={
                "time": np.arange(0, total_time + time_step, time_step * output_interval),
                "y": np.arange(rows) * node_spacing,
                "x": np.arange(columns) * node_spacing,
            }
        )
    elif output_format == 'pickle':
        grid_states = []

    # Progress tracking setup
    start_time = time.time()
    
    def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
        if iteration == total: 
            print()

    # Time loop
    for i in range(ndt + 1):
        # Uplift and update topography
        mg_name.at_node['bedrock__elevation'][mg_name.core_nodes] += uplift_per_step
        mg_name.at_node['topographic__elevation'][:] = (mg_name.at_node["bedrock__elevation"]
                                                        + mg_name.at_node["soil__depth"])

        # Run components
        ew.run_one_step()
        ddTd.run_one_step(time_step)
        fr.run_one_step()
        eroder.run_one_step(time_step)
        
        if landslides == 'True':
            hy.run_one_step(time_step)

        # Save output at specified intervals
        # if i % output_interval == 0:
        #     if output_format == 'netcdf':
        #         output_filename = f'{output_dir}/grid_state_step_{i}.nc'
        #         write_netcdf(output_filename, mg_name)
        #     elif output_format == 'pickle':
        #         output_filename = f'{output_dir}/grid_state_step_{i}.pkl'
        #         with open(output_filename, 'wb') as f:
        #             pickle.dump(mg_name, f)
        #     print(f"Saved grid state at step {i} to {output_filename}")

        # Save output at specified intervals
        if i % output_interval == 0:
            if output_format == 'netcdf':
                time_index = i // output_interval
                ds["topographic__elevation"][time_index] = mg_name.at_node["topographic__elevation"].reshape((rows, columns))
                ds["soil__depth"][time_index] = mg_name.at_node["soil__depth"].reshape((rows, columns))
                ds["bedrock__elevation"][time_index] = mg_name.at_node["bedrock__elevation"].reshape((rows, columns))
                ds["drainage_area"][time_index] = mg_name.at_node["drainage_area"].reshape((rows, columns))
                ds["topographic__steepest_slope"][time_index] = mg_name.at_node["topographic__steepest_slope"].reshape((rows, columns))
            elif output_format == 'pickle':
                grid_states.append({
                    'time': i * time_step,
                    'topographic__elevation': mg_name.at_node["topographic__elevation"].copy(),
                    'soil__depth': mg_name.at_node["soil__depth"].copy(),
                    'bedrock__elevation': mg_name.at_node["bedrock__elevation"].copy(),
                    'drainage_area': mg_name.at_node["drainage_area"].copy(),
                    'topographic__steepest_slope': mg_name.at_node["topographic__steepest_slope"].copy(),
                })

        # Update progress bar
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time * (ndt + 1) / (i + 1)
        time_remaining = estimated_total_time - elapsed_time
        
        print_progress_bar(i + 1, ndt + 1, 
                           prefix=f'Progress:', 
                           suffix=f'Complete. Time remaining: {time.strftime("%H:%M:%S", time.gmtime(time_remaining))}', 
                           length=50)

    # Save the final output file
    if output_format == 'netcdf':
        output_filename = f'{output_dir}/grid_states.nc'
        ds.to_netcdf(output_filename)
    elif output_format == 'pickle':
        output_filename = f'{output_dir}/grid_states.pkl'
        with open(output_filename, 'wb') as f:
            pickle.dump(grid_states, f)

    print(f"\nSaved all grid states to {output_filename}")
    print(f"Total runtime: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

    return mg_name

# %%   Example usage

import os
import time, warnings, copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

dir_path = '/Users/csdmsuser/Documents/Research/CU/GBE/Magnitude_Frequency_Experiments/'
os.chdir(dir_path)

import library_smm as lib
import metric_library as metric
import watershed_metric_calcs as wmc

# Landlab packages
from landlab import RasterModelGrid#, HexModelGrid
from landlab import imshowhs_grid#, imshow_grid
from landlab.io.netcdf import write_netcdf, read_netcdf
from landlab.components import (PriorityFloodFlowRouter,
                                SpaceLargeScaleEroder,
                                GravelBedrockEroder,
                                BedrockLandslider,
                                ExponentialWeatherer,
                                DepthDependentTaylorDiffuser,
                                SteepnessFinder,
                                ChiFinder,
                                ChannelProfiler,
                                )
import pickle
import time
from landlab.io.netcdf import read_netcdf
# other things
warnings.filterwarnings('ignore')
uplift_rates = np.linspace(1e-4,1e-3,8)
grid_eq = read_netcdf("/Users/csdmsuser/Documents/Research/CU/GBE/Magnitude_Frequency_Experiments/grid_gbe_eq_midU_100yrtimesteps.nc") 
time_step = 10 # years
internal_angle_of_friction = 0.85
ls_return_time = 50
total_time = 10000  # years

# Initialize timekeeping
total_steps = total_time // time_step
step_counter = 0
start_time = time.time()

# Run the model
mg_v32_a_test = lib.run_model_with_output(
    scen_num=scen_num, 
    mg_name=grid_eq, 
    total_time=total_time, 
    time_step=time_step, 
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
    landslides=True, 
    ls_return_time=ls_return_time, 
    ls_thresh=internal_angle_of_friction,
    plotting=False,
    plotting_interval=10,
    new_grid=False,
    is_new_run=True,
    save_plots=False,
    output_format='pickle',
    output_interval=time_step
)

# Calculate and print final runtime
end_time = time.time()
total_runtime = end_time - start_time
print(f"Total runtime: {lib.format_time(total_runtime)}")