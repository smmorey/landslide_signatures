import numpy as np
from scipy.ndimage import uniform_filter
from landlab import RasterModelGrid
from landlab.components import PriorityFloodFlowRouter, ChannelProfiler
from landlab.utils import get_watershed_mask
from scipy import stats
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import warnings
import os

# %%
class WatershedMetricsCalculator:
    """Class to handle calculation and analysis of watershed metrics."""
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    def __init__(self, grid_states_path):
        """Initialize calculator with path to grid states."""
        self.grid_states_path = grid_states_path  # Store the path
        self.load_grid_states(grid_states_path)
        self.scenario_num = self._extract_scenario_num(grid_states_path)
        self.metrics_registry = {}
        self.register_default_metrics()
        self.channel_profilers = {}
        self.grids = {}
        self.all_timestep_metrics = []
        
        # Add color scheme for watersheds
        self.watershed_colors = {
            'watershed_1': 'red',
            'watershed_2': 'blue',
            'watershed_3': 'purple',
            'watershed_4': 'yellow',
            'watershed_5': 'magenta',
            'watershed_6': 'gray'
        }
        
        # Define metrics that use diverging colormap
        self.diverging_metrics = [
            'profile_curvature', 'planform_curvature', 'ISED',
            'PES_profile_normal_curv', 'PES_planform_normal_curv',
            'PESe', 'PESD', 
            'xdem_curvature', 'xdem_profile_curvature', 'xdem_planform_curvature'
        ]
        
    def get_grid_at_timestep(self, timestep=-1):
        if not self.grid_states:
            raise ValueError("No grid states loaded")
                
        state = self.grid_states[timestep]
        grid = RasterModelGrid((100, 100), xy_spacing=30)
        
        # Add core fields
        field_mapping = {
            'topographic__elevation': state['topographic__elevation'],
            'soil__depth': state['soil__depth'],
            'bedrock__elevation': state['bedrock__elevation']
        }
        
        # Initialize landslide fields if landslides were enabled
        landslide_fields = {
            'landslide__erosion': np.zeros_like(state['topographic__elevation']),
            'landslide__deposition': np.zeros_like(state['topographic__elevation']),
            'cumulative_landslide_erosion': np.zeros_like(state['topographic__elevation']),
            'cumulative_landslide_deposition': np.zeros_like(state['topographic__elevation']),
            'landslide_size': np.zeros_like(state['topographic__elevation'])
        }
        field_mapping.update(landslide_fields)
        
        # Add all fields to grid
        for name, data in field_mapping.items():
            grid.add_field(name, data, at='node')
            
        return grid
    
    #plots slope vs drainage area for individual watersheds
    def plot_watershed_slope_area(self, timestep=-1, watersheds='all', plot_best_fit=False, 
                                polyfit_degree=1, figsize=(10, 6), use_xdem=False):
        """
        Plot slope vs drainage area for specified watersheds at a given timestep.
        
        Parameters
        ----------
        timestep : int, optional
            Timestep to analyze (default: -1, last timestep)
        watersheds : str or list, optional
            'all' to plot all watersheds, or list of watershed names
        plot_best_fit : bool, optional
            Whether to plot best-fit lines through the data
        polyfit_degree : int, optional
            Degree of polynomial for best-fit line if plot_best_fit=True
        figsize : tuple, optional
            Figure size (width, height)
        use_xdem : bool, optional
            Whether to use xdem_slope instead of regular slope field
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects
        """
        # Get timestep data
        if not hasattr(self, 'all_timestep_metrics') or not self.all_timestep_metrics:
            raise ValueError("No metrics data available. Run process_all_timesteps() first.")
        
        timestep_data = self.all_timestep_metrics[timestep]
        
        # Determine which watersheds to plot
        if watersheds == 'all':
            selected_watersheds = list(timestep_data['metrics'].keys())
        else:
            if isinstance(watersheds, str):
                watersheds = [watersheds]
            selected_watersheds = [w for w in watersheds if w in timestep_data['metrics']]
            if not selected_watersheds:
                raise ValueError(f"No valid watersheds found. Available watersheds: {list(timestep_data['metrics'].keys())}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each watershed
        for watershed in selected_watersheds:
            watershed_data = timestep_data['metrics'][watershed]
            
            # Get slope data
            slope_field = 'xdem_slope' if use_xdem else 'slope'
            if slope_field not in watershed_data:
                print(f"Warning: {slope_field} not found in watershed data")
                continue
                
            slopes = watershed_data[slope_field]
            areas = watershed_data['drainage_area']  # Now using per-cell drainage areas
            
            # Filter out invalid values
            valid_mask = (areas > 0) & (slopes > 0) & np.isfinite(areas) & np.isfinite(slopes)
            valid_areas = areas[valid_mask]
            valid_slopes = slopes[valid_mask]
            
            if len(valid_areas) == 0:
                print(f"No valid data for watershed {watershed}")
                continue
            
            # Plot data points
            color = self.watershed_colors.get(watershed, 'gray')
            ax.loglog(valid_areas, valid_slopes, '.', color=color, alpha=0.5, 
                     label=f'{watershed} data')
            
            if plot_best_fit:
                # Calculate best fit line in log space
                log_x = np.log10(valid_areas)
                log_y = np.log10(valid_slopes)
                coeffs = np.polyfit(log_x, log_y, polyfit_degree)
                
                # Create smooth line for plotting
                x_smooth = np.logspace(min(log_x), max(log_x), 100)
                log_y_fit = np.polyval(coeffs, np.log10(x_smooth))
                
                ax.loglog(x_smooth, 10**log_y_fit, '-', color=color, alpha=0.8,
                         label=f'{watershed} fit')
                
                # Calculate and display slope (theta) for linear fit
                if polyfit_degree == 1:
                    theta = -coeffs[0]
                    ax.text(0.05, 0.95 - selected_watersheds.index(watershed) * 0.05,
                           f'{watershed}: θ = {theta:.2f}',
                           transform=ax.transAxes, color=color)
        
        # Customize plot
        ax.set_xlabel('Drainage Area (m²)')
        ax.set_ylabel('Slope (m/m)')
        ax.grid(True, which='both', ls='-', alpha=0.2)
        
        # Get elapsed time and handle array case
        elapsed_time = timestep_data['elapsed_time']
        if isinstance(elapsed_time, np.ndarray):
            elapsed_time = elapsed_time[0]
        elapsed_time = float(elapsed_time)
        
        ax.set_title(f'Slope-Area Relationship\nScenario {self.scenario_num} - Time: {elapsed_time:,.0f} years')
        ax.legend()
        
        return fig, ax
    
    #compares slope-area relationships across different scenarios

    def compare_scenarios_slope_area(self, scenarios_data, watersheds='all', timestep=-1,
                                   plot_best_fit=False, polyfit_degree=1, figsize=(12, 8),
                                   use_xdem=False):
        """
        Compare slope-area relationships across different scenarios.
        
        Parameters
        ----------
        scenarios_data : dict
            Dictionary containing data for multiple scenarios
        watersheds : str or list, optional
            'all' to plot all watersheds, or list of watershed names
        timestep : int, optional
            Timestep to analyze (default: -1, last timestep)
        plot_best_fit : bool, optional
            Whether to plot best-fit lines through the data
        polyfit_degree : int, optional
            Degree of polynomial for best-fit line if plot_best_fit=True
        figsize : tuple, optional
            Figure size (width, height)
        use_xdem : bool, optional
            Whether to use xdem_slope instead of regular slope field
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects
        dict
            Dictionary containing fitted parameters for each scenario
        """
        fig, ax = plt.subplots(figsize=figsize)
        fit_params = {}
        
        # Create colormap for scenarios
        n_scenarios = len(scenarios_data)
        colors = plt.cm.viridis(np.linspace(0, 1, n_scenarios))
        
        # Process each scenario
        for idx, (scenario_name, scenario_data) in enumerate(scenarios_data.items()):
            timestep_data = scenario_data[timestep]
            
            # Determine watersheds to plot
            if watersheds == 'all':
                selected_watersheds = list(timestep_data['metrics'].keys())
            else:
                if isinstance(watersheds, str):
                    watersheds = [watersheds]
                selected_watersheds = [w for w in watersheds if w in timestep_data['metrics']]
            
            # Combine data from all selected watersheds
            all_slopes = []
            all_areas = []
            
            slope_field = 'xdem_slope' if use_xdem else 'slope'
            
            for watershed in selected_watersheds:
                watershed_data = timestep_data['metrics'][watershed]
                
                if slope_field not in watershed_data:
                    print(f"Warning: {slope_field} not found in watershed {watershed}")
                    continue
                    
                slopes = watershed_data[slope_field]
                areas = watershed_data['drainage_area']  # Now using per-cell drainage areas
                
                # Handle drainage area (single value or array)
                if isinstance(watershed_data['drainage_area'], (float, int)):
                    areas = np.full_like(slopes, watershed_data['drainage_area'])
                else:
                    areas = watershed_data['drainage_area']
                
                # Filter valid data
                valid_mask = (areas > 0) & (slopes > 0) & np.isfinite(areas) & np.isfinite(slopes)
                all_slopes.extend(slopes[valid_mask])
                all_areas.extend(areas[valid_mask])
            
            if not all_slopes:
                print(f"No valid data for scenario {scenario_name}")
                continue
            
            # Convert to arrays
            all_slopes = np.array(all_slopes)
            all_areas = np.array(all_areas)
            
            # Plot data points
            ax.loglog(all_areas, all_slopes, '.', color=colors[idx], alpha=0.3,
                     label=f'{scenario_name} data')
            
            if plot_best_fit:
                # Calculate best fit line in log space
                log_x = np.log10(all_areas)
                log_y = np.log10(all_slopes)
                coeffs = np.polyfit(log_x, log_y, polyfit_degree)
                fit_params[scenario_name] = coeffs
                
                # Create smooth line for plotting
                x_smooth = np.logspace(min(log_x), max(log_x), 100)
                log_y_fit = np.polyval(coeffs, np.log10(x_smooth))
                
                ax.loglog(x_smooth, 10**log_y_fit, '-', color=colors[idx], alpha=0.8,
                         label=f'{scenario_name} fit')
                
                # Display slope for linear fit
                if polyfit_degree == 1:
                    theta = -coeffs[0]
                    ax.text(0.05, 0.95 - idx * 0.05,
                           f'{scenario_name}: θ = {theta:.2f}',
                           transform=ax.transAxes, color=colors[idx])
        
        # Customize plot
        ax.set_xlabel('Drainage Area (m²)')
        ax.set_ylabel('Slope (m/m)')
        ax.grid(True, which='both', ls='-', alpha=0.2)
        ax.set_title('Slope-Area Relationships Across Scenarios')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig, ax, fit_params

    
    
    # plot slope-area plot for all scenarios at once
    
    def plot_slope_area_all_scenarios(calc, scenarios_data, timestep=-1, plot_best_fit=False):
        """Plot slope vs drainage area for all scenarios."""
        plt.figure(figsize=(10, 8))
        
        cmap_SA = cm.get_cmap('winter')
        cmap_SA2 = cm.get_cmap('viridis') 
        cmap_SA3 = cm.get_cmap('autumn')
        
        colors = {
            '1': [cmap_SA(0.2), cmap_SA(0.5), cmap_SA(0.8)],
            '2': [cmap_SA2(0.2), cmap_SA2(0.5), cmap_SA2(0.8)],
            '3': [cmap_SA3(0.2), cmap_SA3(0.5), cmap_SA3(0.8)]
        }
    
        for scenario_name, scenario_data in scenarios_data.items():
            # Get timestep data
            timestep_data = scenario_data[timestep]
            grid = calc.get_grid_at_timestep(timestep)
            
            # Get slope and area data
            slope = grid.at_node["topographic__steepest_slope"]
            area = grid.at_node["drainage_area"]
            cores = grid.core_nodes
            
            # Determine color based on scenario number
            series = scenario_name[-2]
            scenario_in_series = int(scenario_name[-1]) - 1
            color = colors[series][scenario_in_series]
            
            label = rf'Series {series} - $\phi$ = {0.85 - 0.27*scenario_in_series:.2f}'
            plt.loglog(area[cores], slope[cores], ".", color=color, alpha=0.3, label=label)
    
            if plot_best_fit:
                # Calculate best fit line in log space
                log_x = np.log10(area[cores])
                log_y = np.log10(slope[cores])
                coeffs = np.polyfit(log_x, log_y, 3)
                
                # Create smooth line
                line_x = np.linspace(min(log_x), max(log_x), 1000)
                line_y = np.polyval(coeffs, line_x)
                
                linestyle = ['-', '--', ':'][scenario_in_series]
                plt.loglog(10**line_x, 10**line_y, linestyle, color=color, alpha=0.8)
    
        plt.xlabel("Drainage area (m$^2$)")
        plt.ylabel("Slope (m/m)")
        plt.title('Slope-Area Relationships Across Scenarios')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_slope_area_all_scenarios_old(self, timestep=-1, polyfit_degree=1, plot_best_fit=False, scenarios=None):
        """
        Plots slope vs. drainage area for a given timestep for specified scenarios, with optional best-fit lines. 
        Drainage area is stored as the key "drainage_area" in each timestep dictionary within the list of grid_states.
        Slope is stored as the key "topographic__steepest_slope" in each timestep dictionary within the list of grid_states.
        """
        if scenarios is None:
            scenarios = range(len(self.grid_states))

        # Create a colormap for multiple scenarios
        cmap = cm.get_cmap("viridis", len(scenarios))
        colors = [cmap(i) for i in range(len(scenarios))]

        # Access the slope and area data for each scenario
        slope_area_data = []
        for idx, i in enumerate(scenarios):
            if i < len(self.grid_states):
                state = self.grid_states[i]
                if "topographic__steepest_slope" in state and "drainage_area" in state:
                    slope = state["topographic__steepest_slope"]
                    area = state["drainage_area"]
                    slope_area_data.append((slope, area, colors[idx]))
                else:
                    print(f"Warning: Missing required keys in scenario {i}")
            else:
                print(f"Warning: Scenario index {i} is out of range")

        # Plot the data
        fig, ax = plt.subplots()
        for slope, area, color in slope_area_data:
            # Filter out invalid data
            valid_indices = np.isfinite(np.log10(area)) & np.isfinite(slope)
            if np.sum(valid_indices) < 2:
                continue  # Skip if not enough valid data points

            valid_area = np.log10(area[valid_indices])
            valid_slope = slope[valid_indices]

            ax.scatter(area, slope, color=color, s=1)
            if plot_best_fit:
                z = np.polyfit(valid_area, valid_slope, polyfit_degree)
                p = np.poly1d(z)
                ax.plot(area, p(np.log10(area)), color=color)

        # Set plot labels and scales
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Drainage Area (m^2)")
        ax.set_ylabel("Slope (m/m)")
        ax.set_title("Slope vs. Drainage Area for Selected Scenarios")
        ax.legend(["Scenario " + str(i) for i in scenarios])
        plt.show()
        
        return fig, ax
    
    def plot_slope_vs_drainage_area(self, timestep=-1, figsize=(10, 6), save_path=None):
        """
        Plots slope vs. drainage area for the given timestep from grid_states.

        Parameters:
        - timestep: Index of the timestep to analyze. Defaults to -1 (last timestep).
        - figsize: Tuple indicating the size of the figure. Defaults to (10, 6).
        - save_path: Path to save the plot as an image. Defaults to None (no saving).
        """
        if not hasattr(self, 'grid_states') or not self.grid_states:
            raise ValueError("Grid states are not loaded. Ensure grid_states_path is valid.")
        
        # Get the state data for the selected timestep
        state = self.grid_states[timestep]
        
        # Extract drainage area and slope data
        drainage_area = state.get('drainage_area')
        slope = state.get('topographic__steepest_slope')
        
        # Check if the required keys exist and contain valid data
        if drainage_area is None or slope is None:
            print("Error: 'drainage_area' or 'topographic__steepest_slope' not found in the data.")
            return None, None

        # Mask invalid data (e.g., NaN values)
        valid_mask = np.isfinite(drainage_area) & np.isfinite(slope)
        drainage_area = drainage_area[valid_mask]
        slope = slope[valid_mask]
        
        # Plot the data
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(drainage_area, slope, alpha=0.7, c='blue', s=10)
        
        # Customize the plot
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Drainage Area')
        ax.set_ylabel('Topographic Steepest Slope')
        ax.set_title(f'Slope vs. Drainage Area at Timestep {timestep}')
        ax.grid(True, alpha=0.3)
        
        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig, ax
    
    def plot_slope_area_timestep(self, timestep=-1, polyfit_degree=1):
        """
        Plot slope vs. area for a specific timestep for all scenarios, with best-fit lines.
    
        Parameters:
        - timestep (int): The timestep index to plot (default: last timestep).
        - polyfit_degree (int): Degree of the polynomial for the best-fit line (default: linear).
        """
        # Create a colormap for multiple scenarios
        cmap = cm.get_cmap("viridis", len(self.scenarios))
        colors = [cmap(i) for i in range(len(self.scenarios))]
    
        plt.figure(figsize=(12, 8))
    
        for idx, (scenario_name, scenario_data) in enumerate(self.scenarios.items()):
            # Access slope and area data for the specified timestep
            try:
                area_data = np.array(scenario_data["area"][timestep])
                slope_data = np.array(scenario_data["slope"][timestep])
            except IndexError:
                print(f"Invalid timestep {timestep} for scenario {scenario_name}. Skipping.")
                continue
    
            # Filter out invalid values (NaNs, zeros, or negative values)
            valid_mask = (area_data > 0) & (slope_data > 0)
            area_data = area_data[valid_mask]
            slope_data = slope_data[valid_mask]
    
            if len(area_data) == 0 or len(slope_data) == 0:
                print(f"No valid data for scenario {scenario_name} at timestep {timestep}. Skipping.")
                continue
    
            # Plot slope vs. area data
            plt.loglog(
                area_data, slope_data, '.', alpha=0.6, color=colors[idx],
                label=f"{scenario_name} (Data)"
            )
    
            # Compute and overlay best-fit line
            log_area = np.log10(area_data)
            log_slope = np.log10(slope_data)
            coeffs = np.polyfit(log_area, log_slope, polyfit_degree)
            best_fit = np.polyval(coeffs, log_area)
            plt.loglog(
                area_data, 10**best_fit, '-', color=colors[idx],
                label=f"{scenario_name} (Fit)"
            )
    
        # Add labels, legend, and grid
        plt.xlabel("Drainage Area (m²)")
        plt.ylabel("Slope (m/m)")
        plt.title(f"Slope vs. Area at Timestep {timestep if timestep >= 0 else 'Last'}")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()

        
    def get_grid_at_timestep_old_v2(self, timestep=-1):
        """
        Get a Landlab grid initialized with data from a specific timestep.
        
        Parameters
        ----------
        timestep : int, optional
            Timestep to use for grid initialization (-1 for last timestep)
        
        Returns
        -------
        RasterModelGrid
            Landlab grid initialized with data from the specified timestep
        """
        if not self.grid_states:
            raise ValueError("No grid states loaded. Run process_all_timesteps() first.")
                
        state = self.grid_states[timestep]
        
        # Create grid
        grid = RasterModelGrid((100, 100), xy_spacing=30)  # Standard dimensions
        
        # Add fields from state
        field_mapping = {
            'topographic__elevation': 'topographic__elevation',
            'soil__depth': 'soil__depth',
            'bedrock__elevation': 'bedrock__elevation',
            'landslide__erosion': 'landslide__erosion',
            'landslide__deposition': 'landslide__deposition',
            'cumulative_landslide_erosion': 'cumulative_landslide_erosion',
            'cumulative_landslide_deposition': 'cumulative_landslide_deposition',
            'landslide_size': 'landslide_size'
        }
        
        for field_name, state_name in field_mapping.items():
            if state_name in state:
                grid.add_field(field_name, state[state_name], at='node')
        
        return grid
    


    def get_grid_at_timestep_old(self, timestep=-1):
        """
        Get a Landlab grid initialized with data from a specific timestep.
        
        Parameters
        ----------
        timestep : int, optional
            Timestep to use for grid initialization (-1 for last timestep)
        
        Returns
        -------
        RasterModelGrid
            Landlab grid initialized with data from the specified timestep
        """
        if not self.grid_states:
            raise ValueError("No grid states loaded. Run process_all_timesteps() first.")
            
        state = self.grid_states[timestep]
        
        # Create grid
        grid = RasterModelGrid((100, 100), xy_spacing=30)  # Standard dimensions
    
        field_mapping = {
            'topographic__elevation': 'topographic__elevation',
            'soil__depth': 'soil__depth',
            'bedrock__elevation': 'bedrock__elevation',
            'landslide__erosion': 'landslide__erosion',
            'landslide__deposition': 'landslide__deposition',
            'cumulative_landslide_erosion': 'cumulative_landslide_erosion',
            'cumulative_landslide_deposition': 'cumulative_landslide_deposition',
            'landslide_size': 'landslide_size'
        }
        
        for field_name, state_name in field_mapping.items():
            if state_name in state:
                grid.add_field(field_name, state[state_name], at='node')
                
        # Add bedrock elevation if not present
        if 'bedrock__elevation' not in grid.at_node:
            grid.add_field('bedrock__elevation', 
                         grid.at_node['topographic__elevation'] - 
                         grid.at_node['soil__depth'], 
                         at='node')
        
        return grid   
    @staticmethod
    def _extract_scenario_num(path):
        """Extract scenario number from path."""
        import re
        match = re.search(r'scenario(\d+)', path)
        return int(match.group(1)) if match else None
        
    def load_grid_states(self, path):
        """Load grid states from pickle file."""
        with open(path, 'rb') as f:
            self.grid_states = pickle.load(f)
            
    def register_metric(self, name, calculation_func, aggregation_func=np.mean):
        """
        Register a new metric calculation.
        
        Parameters
        ----------
        name :              str | Name of the metric
        calculation_func :  callable | Function to calculate the metric for a single grid
        aggregation_func :  callable, optional | Function to aggregate metric values (default: np.mean)
        """
        self.metrics_registry[name] = {
            'calc_func': calculation_func,
            'agg_func': aggregation_func
        }
        
    def register_default_metrics(self):
        """Register the default set of metrics."""
        self.register_metric('tri', self.calculate_tri)
        self.register_metric('rugosity', self.calculate_rugosity)
        
    def convert_landlab_to_xdem(self, grid, crs="EPSG:32722"):
        """
        Convert a Landlab grid to an xDEM DEM object.
        """
        import xdem
        import xarray as xr
        import tempfile
        import os
        
        # Extract elevation and reshape
        elevation = grid.at_node['topographic__elevation'].reshape(grid.shape)
        dx = grid.dx
        
        # Create xarray DataArray with proper coordinates
        y_coords = np.arange(grid.shape[0]) * dx
        x_coords = np.arange(grid.shape[1]) * dx
        
        dem_array = xr.DataArray(
            data=elevation,
            dims=('y', 'x'),
            coords={
                'y': y_coords,
                'x': x_coords
            }
        )
        
        # Add spatial reference
        dem_array.rio.write_crs(crs, inplace=True)
        dem_array.rio.set_spatial_dims('x', 'y', inplace=True)
        
        # Save to temporary file and create xDEM DEM object
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            temp_file = tmp.name
            dem_array.rio.to_raster(temp_file)
            
            try:
                dem = xdem.DEM(temp_file)
            finally:
                os.remove(temp_file)
                
        return dem

    def convert_xdem_to_landlab_format(self, xdem_array, grid_shape):
        """
        Convert an xDEM calculation result back to Landlab format.
        """
        if hasattr(xdem_array, 'data'):
            data = xdem_array.data
        elif hasattr(xdem_array, 'values'):
            data = xdem_array.values
        else:
            data = xdem_array
            
        # Ensure data matches grid shape
        if data.shape != grid_shape:
            raise ValueError(f"Shape mismatch: xDEM result {data.shape} != grid shape {grid_shape}")
            
        return data.flatten()  # Convert to 1D array for Landlab

    def calculate_xdem_metrics(self, grid, include_metrics=None):
        """Calculate xDEM metrics silently."""
        import xdem
        import warnings
        
        # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            dem = self.convert_landlab_to_xdem(grid)
            metrics = {}
            
            available_metrics = {
                'slope': lambda dem: xdem.terrain.slope(dem, method='Horn'),
                'aspect': xdem.terrain.aspect,
                'curvature': xdem.terrain.curvature,
                'planform_curvature': xdem.terrain.planform_curvature,
                'profile_curvature': xdem.terrain.profile_curvature,
                'tpi': xdem.terrain.topographic_position_index,
                'tri': xdem.terrain.terrain_ruggedness_index,
                'roughness': xdem.terrain.roughness,
                'rugosity': xdem.terrain.rugosity,
                'fractal_roughness': xdem.terrain.fractal_roughness
            }
            
            if include_metrics is None:
                include_metrics = list(available_metrics.keys())
            
            for metric_name in include_metrics:
                if metric_name in available_metrics:
                    try:
                        xdem_result = available_metrics[metric_name](dem)
                        metrics[metric_name] = self.convert_xdem_to_landlab_format(xdem_result, grid.shape)
                    except Exception as e:
                        print(f"Warning: Failed to calculate {metric_name}: {str(e)}")
        
        return metrics

# =============================================================================
#     def calculate_xdem_metrics_old(self, grid, include_metrics=None):
#         """
#         Calculate terrain metrics using xDEM for a given grid.
#         """
#         import xdem
#         
#         # Available metrics and their corresponding xDEM functions
#         available_metrics = {
#             'slope': lambda dem: xdem.terrain.slope(dem, method='Horn'),
#             'aspect': xdem.terrain.aspect,
#             'curvature': xdem.terrain.curvature,
#             'planform_curvature': xdem.terrain.planform_curvature,
#             'profile_curvature': xdem.terrain.profile_curvature,
#             'tpi': xdem.terrain.topographic_position_index,
#             'tri': xdem.terrain.terrain_ruggedness_index,
#             'roughness': xdem.terrain.roughness,
#             'rugosity': xdem.terrain.rugosity,
#             'fractal_roughness': xdem.terrain.fractal_roughness
#         }
#         
#         if include_metrics is None:
#             include_metrics = list(available_metrics.keys())
#             
#         # Convert to xDEM format
#         dem = self.convert_landlab_to_xdem(grid)
#         
#         # Calculate metrics
#         metrics = {}
#         for metric_name in include_metrics:
#             if metric_name not in available_metrics:
#                 print(f"Warning: {metric_name} not recognized, skipping...")
#                 continue
#                 
#             # Calculate metric using xDEM
#             xdem_result = available_metrics[metric_name](dem)
#             
#             # Convert back to Landlab format
#             metrics[metric_name] = self.convert_xdem_to_landlab_format(xdem_result, grid.shape)
#             
#         return metrics
#     
#     
# =============================================================================
    @staticmethod
    def calculate_gradient_components(elevation, dx):
        """
        Calculate slope gradient components using central difference.
        
        Parameters
        ----------
        elevation : ndarray | 2D array of elevation values
        dx :        float | Grid spacing
            
        Returns
        -------
        dict :      Dictionary containing slope components and magnitudes
        """
        # Calculate gradients
        dz_dx = np.zeros_like(elevation)
        dz_dy = np.zeros_like(elevation)
        dz_dx[:, 1:-1] = (elevation[:, 2:] - elevation[:, :-2]) / (2 * dx)
        dz_dy[1:-1, :] = (elevation[2:, :] - elevation[:-2, :]) / (2 * dx)
        
        # Calculate slope magnitude
        slope_squared = dz_dx**2 + dz_dy**2
        slope_magnitude = np.sqrt(slope_squared)
        
        return {
            'dz_dx': dz_dx,
            'dz_dy': dz_dy,
            'slope_squared': slope_squared,
            'slope_magnitude': slope_magnitude
        }
    
    @staticmethod
    def calculate_surface_derivatives(elevation, dx, gradient_components=None):
        """
        Calculate surface derivatives using pre-calculated gradients if available.
        
        Parameters
        ----------
        elevation :             ndarray | 2D array of elevation values
        dx :                    float | Grid spacing
        gradient_components :   dict, optional | Pre-calculated gradient components
            
        Returns
        -------
        dict :                  Dictionary containing all derivatives
        """
        if gradient_components is None:
            gradient_components = WatershedMetricsCalculator.calculate_gradient_components(elevation, dx)
        
        # Get first derivatives from components
        dz_dx = gradient_components['dz_dx']
        dz_dy = gradient_components['dz_dy']
        
        # Calculate second derivatives
        d2z_dx2 = np.zeros_like(elevation)
        d2z_dy2 = np.zeros_like(elevation)
        d2z_dxdy = np.zeros_like(elevation)
        
        d2z_dx2[:, 1:-1] = (elevation[:, 2:] - 2*elevation[:, 1:-1] + 
                           elevation[:, :-2]) / (dx**2)
        d2z_dy2[1:-1, :] = (elevation[2:, :] - 2*elevation[1:-1, :] + 
                           elevation[:-2, :]) / (dx**2)
        d2z_dxdy[1:-1, 1:-1] = ((elevation[2:, 2:] - elevation[2:, :-2] -
                                elevation[:-2, 2:] + elevation[:-2, :-2]) / 
                               (4 * dx**2))
        
        return {
            'dz_dx': dz_dx,
            'dz_dy': dz_dy,
            'd2z_dx2': d2z_dx2,
            'd2z_dy2': d2z_dy2,
            'd2z_dxdy': d2z_dxdy
        }
    
    @staticmethod
    def calculate_curvatures(elevation, dx, derivatives=None):
        """
        Calculate profile, planform, and regular curvature.
        
        Parameters
        ----------
        elevation :     ndarray | 2D array of elevation values
        dx :            float | Grid spacing
        derivatives :   dict, optional | Pre-calculated surface derivatives
            
        Returns
        -------
        dict
            Dictionary containing curvature values:
            - profile_curvature: curvature in steepest direction
            - planform_curvature: curvature transverse to slope
            - regular_curvature: overall surface curvature (mean curvature)
        """
        if derivatives is None:
            derivatives = WatershedMetricsCalculator.calculate_surface_derivatives(elevation, dx)
            
        # Get components
        dz_dx = derivatives['dz_dx']
        dz_dy = derivatives['dz_dy']
        d2z_dx2 = derivatives['d2z_dx2']
        d2z_dy2 = derivatives['d2z_dy2']
        d2z_dxdy = derivatives['d2z_dxdy']
        
        # Calculate slope components
        slope_squared = dz_dx**2 + dz_dy**2
        
        # Initialize curvature arrays
        profile_curv = np.zeros_like(elevation)
        planform_curv = np.zeros_like(elevation)
        regular_curv = np.zeros_like(elevation)
        
        # Calculate curvatures where slope is not zero
        valid_slopes = slope_squared > 0
        if np.any(valid_slopes):
            # Profile curvature
            profile_curv[valid_slopes] = (-(dz_dx[valid_slopes]**2 * d2z_dx2[valid_slopes] + 
                                          2*dz_dx[valid_slopes]*dz_dy[valid_slopes]*d2z_dxdy[valid_slopes] + 
                                          dz_dy[valid_slopes]**2 * d2z_dy2[valid_slopes]) / 
                                        (slope_squared[valid_slopes] * np.sqrt(1 + slope_squared[valid_slopes])))
            
            # Planform curvature
            planform_curv[valid_slopes] = (-(dz_dy[valid_slopes]**2 * d2z_dx2[valid_slopes] - 
                                           2*dz_dx[valid_slopes]*dz_dy[valid_slopes]*d2z_dxdy[valid_slopes] + 
                                           dz_dx[valid_slopes]**2 * d2z_dy2[valid_slopes]) / 
                                         (slope_squared[valid_slopes]**1.5))
            
            # Regular curvature (mean curvature)
            regular_curv[valid_slopes] = ((d2z_dx2[valid_slopes] + d2z_dy2[valid_slopes]) / 
                                        (2 * (1 + slope_squared[valid_slopes])**1.5))
        
        return {
            'profile_curvature': profile_curv,
            'planform_curvature': planform_curv,
            'regular_curvature': regular_curv
        }

    def calculate_pes_metrics(self, elevation, dx, metrics, rho=2600, g=9.81):
        """
        Calculate PES metrics using xDEM-calculated curvatures.
        
        Parameters
        ----------
        elevation : ndarray | 2D array of elevation values (not used directly anymore, kept for compatibility)
        dx :        float |  Grid spacing (not used directly anymore, kept for compatibility)
        metrics :   dict | Dictionary containing xDEM-calculated metrics including slope, profile and planform curvature
        rho :       float, optional | Density (kg/m^3), default 2600
        g :         float, optional | Gravitational acceleration (m/s^2), default 9.81
                
        Returns
        -------
        dict
            Dictionary containing:
            - PES: Potential Energy on Slope
            - PES_profile_normal_curv: Profile component
            - PES_planform_normal_curv: Planform component
            - PESe: Excess PES
            - PESD: PES difference (profile - planform)
            - ISED: Index of Slope Energy Disequilibrium
        """
        # Convert masked arrays to regular arrays, replacing masked values with NaN
        def unmask_array(arr):
            if hasattr(arr, 'mask'):
                return np.ma.filled(arr, fill_value=np.nan)
            return arr
        
        # Get slope from xDEM metrics and convert to regular array
        slope_magnitude = unmask_array(metrics['xdem_slope'])
        
        # Get curvatures from xDEM metrics and convert to regular arrays
        profile_curvature = unmask_array(metrics['xdem_profile_curvature'])
        planform_curvature = unmask_array(metrics['xdem_planform_curvature'])
        
        # Calculate PES components
        PES = rho * g * slope_magnitude
        PES_profile = rho * g * profile_curvature
        PES_planform = rho * g * planform_curvature
        PES_excess = PES_profile + PES_planform
        PESD = PES_profile - PES_planform
        
        # Calculate ISED, handling NaN values
        ISED = np.zeros_like(PES)
        valid_pes = (PES != 0) & np.isfinite(PES) & np.isfinite(PES_excess)
        ISED[valid_pes] = 100 * (PES_excess[valid_pes] / PES[valid_pes])
        
        return {
            'PES': PES,
            'PES_profile_normal_curv': PES_profile,
            'PES_planform_normal_curv': PES_planform,
            'PESe': PES_excess,
            'PESD': PESD,
            'ISED': ISED
        }
    
    
# =============================================================================
#     @staticmethod
#     def calculate_pes_metrics_old(elevation, dx, curvatures=None, rho=2600, g=9.81):
#         """
#         Calculate PES metrics using pre-calculated curvatures if available.
#         
#         Parameters
#         ----------
#         elevation : ndarray
#             2D array of elevation values
#         dx : float
#             Grid spacing
#         curvatures : dict, optional
#             Pre-calculated curvatures
#         rho : float, optional
#             Density (kg/m^3), default 2600
#         g : float, optional
#             Gravitational acceleration (m/s^2), default 9.81
#             
#         Returns
#         -------
#         dict
#             Dictionary containing:
#             - PES: Potential Energy on Slope
#             - PES_profile_normal_curv: Profile component
#             - PES_planform_normal_curv: Planform component
#             - PESe: Excess PES
#             - PESD: PES difference (profile - planform)
#             - ISED: Index of Slope Energy Disequilibrium
#         """
#         # Get gradient components
#         grad_comps = WatershedMetricsCalculator.calculate_gradient_components(elevation, dx)
#         slope_magnitude = grad_comps['slope_magnitude']
#         
#         # Get curvatures if not provided
#         if curvatures is None:
#             derivatives = WatershedMetricsCalculator.calculate_surface_derivatives(
#                 elevation, dx, gradient_components=grad_comps)
#             curvatures = WatershedMetricsCalculator.calculate_curvatures(
#                 elevation, dx, derivatives=derivatives)
#         
#         # Calculate PES components
#         PES = rho * g * slope_magnitude
#         PES_profile = rho * g * curvatures['profile_curvature']
#         PES_planform = rho * g * curvatures['planform_curvature']
#         PES_excess = PES_profile + PES_planform
#         PESD = PES_profile - PES_planform
#         
#         # Calculate ISED
#         ISED = np.zeros_like(PES)
#         valid_pes = PES != 0
#         ISED[valid_pes] = 100 * (PES_excess[valid_pes] / PES[valid_pes])
#         
#         return {
#             'PES': PES,
#             'PES_profile_normal_curv': PES_profile,
#             'PES_planform_normal_curv': PES_planform,
#             'PESe': PES_excess,
#             'PESD': PESD,
#             'ISED': ISED
#         }
#     
# =============================================================================
    @staticmethod
    def calculate_tri(elevation, window_size=5):
        """Calculate Topographic Roughness Index."""
        mean_square = uniform_filter(elevation ** 2, size=window_size, mode='nearest')
        square_mean = uniform_filter(elevation, window_size) ** 2
        return mean_square - square_mean
    
    @staticmethod
    def calculate_rugosity(grid, watershed_mask):
        """Calculate rugosity metrics."""
        elevation = grid.at_node['topographic__elevation'].reshape(grid.shape)
        watershed_elevations = elevation[watershed_mask.reshape(grid.shape)]
        dx, dy = grid.dx, grid.dy
        
        # Calculate areas
        planimetric_area = np.sum(watershed_mask) * dx * dy
        surface_area = 0.0
        rows, cols = grid.shape
        y_coords, x_coords = np.where(watershed_mask.reshape(grid.shape))
        
        for i in range(len(y_coords)):
            y, x = y_coords[i], x_coords[i]
            if y < rows-1 and x < cols-1:
                z00 = elevation[y, x]
                z10 = elevation[y+1, x]
                z01 = elevation[y, x+1]
                z11 = elevation[y+1, x+1]
                
                v1 = np.array([dx, 0, z10-z00])
                v2 = np.array([0, dy, z01-z00])
                v3 = np.array([dx, 0, z11-z01])
                v4 = np.array([0, dy, z11-z10])
                
                area1 = 0.5 * np.linalg.norm(np.cross(v1, v2))
                area2 = 0.5 * np.linalg.norm(np.cross(v3, v4))
                surface_area += area1 + area2
                
        return {
            'rugosity': surface_area / planimetric_area,
            'surface_area': surface_area,
            'planimetric_area': planimetric_area,
            'mean_elevation': np.mean(watershed_elevations),
            'elevation_range': np.ptp(watershed_elevations)
            }
    
    def plot_watershed_metric(self, timestep, metric_name, watersheds='all', grid_shape=(100, 100),
                        num_watersheds=6, show_boundaries=True, show_channels=True, figsize=(12, 8), save_path=None):
        """
        Plot metric values for selected watersheds in a single plot.
        
        Parameters
        ----------
        timestep :          int ; Timestep to plot
        metric_name :       str ; Name of metric to plot
        watersheds :        str or list, optional ; 'all' to plot all watersheds, or list of watershed names to plot
                                (e.g., ['watershed_1', 'watershed_3'])
        grid_shape :        tuple, optional ; Shape of the grid (rows, cols)
        show_boundaries :   bool, optional ; Whether to show watershed boundaries
        show_channels :     bool, optional ; Whether to show channel networks
        figsize :           tuple, optional ; Figure size in inches
        save_path :         str, optional ; If provided, save the figure to this path
        channel_width :     float, optional ; Width of channel lines
        channel_alpha :     float, optional ; Transparency of channel lines (0-1)
        """

        # Get required objects
        cp = self.get_channel_profiler(timestep)
        if show_channels and cp is None:
            print("Warning: No ChannelProfiler available for this timestep. Channels will not be shown.")
            show_channels = False
                
        timestep_data = self.get_timestep_metrics(timestep)
        
        # Handle watershed selection
        if watersheds == 'all':
            selected_watersheds = list(timestep_data['metrics'].keys())
        else:
            if isinstance(watersheds, str):
                watersheds = [watersheds]
            invalid_watersheds = [w for w in watersheds if w not in timestep_data['metrics']]
            if invalid_watersheds:
                available_watersheds = list(timestep_data['metrics'].keys())
                raise ValueError(f"Invalid watershed(s): {invalid_watersheds}\n"
                               f"Available watersheds: {available_watersheds}")
            selected_watersheds = watersheds
            
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = plt.axes([0.1, 0.1, 0.6, 0.8])
        
        # Set up colormap and normalization based on metric type
        if metric_name.startswith('xdem_'):
            # Handle xDEM metrics specifically
            base_metric = metric_name.replace('xdem_', '')
            if base_metric == 'slope':
                # Slope in degrees, use sequential colormap
                vmin, vmax = 0, 45  # typical range for slope in degrees
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.cm.YlOrRd
            elif 'curvature' in base_metric:
                # Curvature metrics should be diverging
                vmax = 10  # Typical range for curvature
                norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                cmap = plt.cm.RdBu_r
            else:
                # Default handling
                all_values = np.concatenate([
                    timestep_data['metrics'][w][metric_name] 
                    for w in selected_watersheds
                ])
                vmax = np.nanmax(np.abs(all_values))
                norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
                cmap = plt.cm.viridis
        else:
            # Original metric handling
            all_values = np.concatenate([
                timestep_data['metrics'][w][metric_name] 
                for w in selected_watersheds
            ])
            
            if metric_name in self.diverging_metrics:
                vmax = np.nanmax(np.abs(all_values))
                norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                cmap = plt.cm.RdBu_r
            else:
                vmax = np.nanmax(all_values)
                norm = mcolors.Normalize(vmin=0, vmax=vmax)
                cmap = plt.cm.viridis
        
        # Create custom colormap with white for NaN
        my_cmap = cmap.copy()
        my_cmap.set_bad(color='white')
        
        # Create combined grid for all watersheds
        combined_grid = np.full(grid_shape, np.nan)
        
        # Combine all watershed data
        for watershed in selected_watersheds:
            data = timestep_data['metrics'][watershed]
            values = data[metric_name]
            mask = data['mask']
            
            watershed_grid = np.full(grid_shape[0] * grid_shape[1], np.nan)
            watershed_grid[mask] = values
            watershed_grid = watershed_grid.reshape(grid_shape)
            
            valid_data = ~np.isnan(watershed_grid)
            combined_grid[valid_data] = watershed_grid[valid_data]
        
        # Plot combined data
        im = ax.imshow(combined_grid, cmap=my_cmap, norm=norm)
        
        # Add colorbar with appropriate label
        cax = plt.axes([0.75, 0.1, 0.02, 0.8])
        cbar = plt.colorbar(im, cax=cax)
        if metric_name.startswith('xdem_'):
            base_metric = metric_name.replace('xdem_', '')
            if base_metric == 'slope':
                cbar.set_label('Slope (degrees)')
            elif 'curvature' in base_metric:
                cbar.set_label('Curvature (1/100m)')
            else:
                cbar.set_label(metric_name)
        else:
            cbar.set_label(metric_name)
        
        # Add watershed boundaries and channels if requested
        legend_elements = []
        if show_boundaries or show_channels:
            for watershed in selected_watersheds:
                data = timestep_data['metrics'][watershed]
                mask = data['mask']
                
                if show_boundaries:
                    mask_2d = mask.reshape(grid_shape)
                    boundary = np.zeros_like(mask_2d)
                    if mask_2d.shape[0] > 2 and mask_2d.shape[1] > 2:
                        boundary[1:-1, 1:-1] = mask_2d[1:-1, 1:-1] & (
                            ~mask_2d[:-2, 1:-1] |
                            ~mask_2d[2:, 1:-1] |
                            ~mask_2d[1:-1, :-2] |
                            ~mask_2d[1:-1, 2:]
                        )
                        # Handle edges
                        boundary[0, :] = mask_2d[0, :] & ~mask_2d[1, :]
                        boundary[-1, :] = mask_2d[-1, :] & ~mask_2d[-2, :]
                        boundary[:, 0] = mask_2d[:, 0] & ~mask_2d[:, 1]
                        boundary[:, -1] = mask_2d[:, -1] & ~mask_2d[:, -2]
                    
                    if np.any(boundary):
                        color = self.watershed_colors.get(watershed, 'black')
                        ax.contour(boundary, colors=[color], levels=[0.5], linewidths=2, alpha=0.8)
                        legend_elements.append(Line2D([0], [0], color=color, label=watershed))
        
        # Add legend if needed
        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower left')
        
        # Get elapsed time and handle array case
        elapsed_time = timestep_data['elapsed_time']
        if isinstance(elapsed_time, np.ndarray):
            elapsed_time = elapsed_time[0]  # Take first value if it's an array
        elapsed_time = float(elapsed_time)  # Convert to Python float
        
        # Set title with scenario number
        title = f"Scenario {self.scenario_num} - {metric_name}\n"
        title += f"{len(selected_watersheds)} watersheds - Time: {elapsed_time:,.0f} years"
        plt.title(title, fontsize=14)
        
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
                
        plt.show()
        
    def plot_watershed_metric_old(self, timestep, metric_name, watersheds='all', grid_shape=(100, 100),
                        num_watersheds=6, show_boundaries=True, show_channels=True, figsize=(12, 8), save_path=None):
        """
        Plot metric values for selected watersheds in a single plot.
        
        Parameters
        ----------
        timestep :          int ; Timestep to plot
        metric_name :       str ; Name of metric to plot
        watersheds :        str or list, optional ; 'all' to plot all watersheds, or list of watershed names to plot
                                (e.g., ['watershed_1', 'watershed_3'])
        grid_shape :        tuple, optional ; Shape of the grid (rows, cols)
        show_boundaries :   bool, optional ; Whether to show watershed boundaries
        show_channels :     bool, optional ; Whether to show channel networks
        figsize :           tuple, optional ; Figure size in inches
        save_path :         str, optional ; If provided, save the figure to this path
        channel_width :     float, optional ; Width of channel lines
        channel_alpha :     float, optional ; Transparency of channel lines (0-1)
        """

        # Get required objects
        cp = self.get_channel_profiler(timestep)
        if show_channels and cp is None:
            print("Warning: No ChannelProfiler available for this timestep. Channels will not be shown.")
            show_channels = False
            
        timestep_data = self.get_timestep_metrics(timestep)
        
        # Handle watershed selection
        if watersheds == 'all':
            selected_watersheds = list(timestep_data['metrics'].keys())
        else:
            if isinstance(watersheds, str):
                watersheds = [watersheds]
            invalid_watersheds = [w for w in watersheds if w not in timestep_data['metrics']]
            if invalid_watersheds:
                available_watersheds = list(timestep_data['metrics'].keys())
                raise ValueError(f"Invalid watershed(s): {invalid_watersheds}\n"
                               f"Available watersheds: {available_watersheds}")
            selected_watersheds = watersheds
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = plt.axes([0.1, 0.1, 0.6, 0.8])
        
        # Set up colormap and normalization based on metric type
        if metric_name.startswith('xdem_'):
            # Handle xDEM metrics specifically
            base_metric = metric_name.replace('xdem_', '')
            if base_metric == 'slope':
                # Slope in degrees, use sequential colormap
                vmin, vmax = 0, 45  # typical range for slope in degrees
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.cm.YlOrRd
            elif 'curvature' in base_metric:
                # Curvature metrics should be diverging
                vmax = 10  # Typical range for curvature
                norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                cmap = plt.cm.RdBu_r
            else:
                # Default handling
                all_values = np.concatenate([
                    timestep_data['metrics'][w][metric_name] 
                    for w in selected_watersheds
                ])
                vmax = np.nanmax(np.abs(all_values))
                norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
                cmap = plt.cm.viridis
        else:
            # Original metric handling
            all_values = np.concatenate([
                timestep_data['metrics'][w][metric_name] 
                for w in selected_watersheds
            ])
            
            if metric_name in self.diverging_metrics:
                vmax = np.nanmax(np.abs(all_values))
                norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                cmap = plt.cm.RdBu_r
            else:
                vmax = np.nanmax(all_values)
                norm = mcolors.Normalize(vmin=0, vmax=vmax)
                cmap = plt.cm.viridis
        
        # Create custom colormap with white for NaN
        my_cmap = cmap.copy()
        my_cmap.set_bad(color='white')
        
        # Create combined grid for all watersheds
        combined_grid = np.full(grid_shape, np.nan)
        
        # Combine all watershed data
        for watershed in selected_watersheds:
            data = timestep_data['metrics'][watershed]
            values = data[metric_name]
            mask = data['mask']
            
            watershed_grid = np.full(grid_shape[0] * grid_shape[1], np.nan)
            watershed_grid[mask] = values
            watershed_grid = watershed_grid.reshape(grid_shape)
            
            valid_data = ~np.isnan(watershed_grid)
            combined_grid[valid_data] = watershed_grid[valid_data]
        
        # Plot combined data
        im = ax.imshow(combined_grid, cmap=my_cmap, norm=norm)
        
        # Add colorbar with appropriate label
        cax = plt.axes([0.75, 0.1, 0.02, 0.8])
        cbar = plt.colorbar(im, cax=cax)
        if metric_name.startswith('xdem_'):
            base_metric = metric_name.replace('xdem_', '')
            if base_metric == 'slope':
                cbar.set_label('Slope (degrees)')
            elif 'curvature' in base_metric:
                cbar.set_label('Curvature (1/100m)')
            else:
                cbar.set_label(metric_name)
        else:
            cbar.set_label(metric_name)
        
        # Add watershed boundaries and channels if requested
        legend_elements = []
        if show_boundaries or show_channels:
            for watershed in selected_watersheds:
                data = timestep_data['metrics'][watershed]
                mask = data['mask']
                
                if show_boundaries:
                    mask_2d = mask.reshape(grid_shape)
                    boundary = np.zeros_like(mask_2d)
                    if mask_2d.shape[0] > 2 and mask_2d.shape[1] > 2:
                        boundary[1:-1, 1:-1] = mask_2d[1:-1, 1:-1] & (
                            ~mask_2d[:-2, 1:-1] |
                            ~mask_2d[2:, 1:-1] |
                            ~mask_2d[1:-1, :-2] |
                            ~mask_2d[1:-1, 2:]
                        )
                        # Handle edges
                        boundary[0, :] = mask_2d[0, :] & ~mask_2d[1, :]
                        boundary[-1, :] = mask_2d[-1, :] & ~mask_2d[-2, :]
                        boundary[:, 0] = mask_2d[:, 0] & ~mask_2d[:, 1]
                        boundary[:, -1] = mask_2d[:, -1] & ~mask_2d[:, -2]
                    
                    if np.any(boundary):
                        color = self.watershed_colors.get(watershed, 'black')
                        ax.contour(boundary, colors=[color], levels=[0.5], linewidths=2, alpha=0.8)
                        legend_elements.append(Line2D([0], [0], color=color, label=watershed))
        
        # Add legend if needed
        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower left')
        
        # Set title with scenario number
        title = f"Scenario {self.scenario_num} - {metric_name}\n"
        title += f"{len(selected_watersheds)} watersheds - Time: {timestep_data['elapsed_time']:,} years"
        plt.title(title, fontsize=14)
        
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        plt.show()
            
    def get_timestep_metrics(self, timestep):
        """Get metrics data for a specific timestep."""
        if not hasattr(self, 'all_timestep_metrics') or not self.all_timestep_metrics:
            raise ValueError("No metrics data available. Did you run process_all_timesteps()?")
        try:
            return self.all_timestep_metrics[timestep]
        except (IndexError):
            raise ValueError(f"No metrics data available for timestep {timestep}"
                             f"Available timestesp: 0-{len(self.all_timestep_metrics)-1}")
        
    def plot_metric_timeseries(self, metric_name, watersheds='all', figsize=(10, 6), save_path=None):
        """Plot time evolution of a metric for selected watersheds.
        Parameters
        ----------
        metric_name :   str ; Name of metric to plot
        watersheds :    list or str, optional ; List of watershed names to plot, or 'all' for all watersheds
        figsize :       tuple, optional ; Figure size in inches
        save_path :     str, optional ; If provided, save the figure to this path
        """
        plt.figure(figsize=figsize)
        
        # Extract times and handle array case
        times = []
        for data in self.all_timestep_metrics:
            elapsed_time = data['elapsed_time']
            if isinstance(elapsed_time, np.ndarray):
                elapsed_time = elapsed_time[0]
            times.append(float(elapsed_time))
            
        if watersheds == 'all':
            watersheds = list(self.all_timestep_metrics[0]['metrics'].keys())
        
        for watershed in watersheds:
            values = [data['metrics'][watershed][metric_name].mean() 
                     for data in self.all_timestep_metrics]
            plt.plot(times, values, '-o', label=watershed, 
                    color=self.watershed_colors[watershed])
        
        plt.xlabel('Time (years)')
        plt.ylabel(f'Mean {metric_name}')
        plt.title(f'Time Evolution of {metric_name} by Watershed')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        plt.show()
        

        
    def process_grid_state(self, state, num_watersheds=6, initial_threshold=1000):
        """Process grid state silently."""
        grid_shape = (100, 100)
        dx = 30  # grid spacing
    
        # Create Landlab grid and add elevation field
        mg = RasterModelGrid(grid_shape, xy_spacing=dx)
        elevation = state['topographic__elevation'].reshape(grid_shape)
        mg.add_field('topographic__elevation', state['topographic__elevation'], at='node')
    
        # Initialize metrics dictionary
        metrics = {}
        
        # Calculate xDEM metrics first - this must succeed for PES calculations
        try:
            xdem_metrics = self.calculate_xdem_metrics(mg)
            # Store xDEM metrics directly in metrics dictionary without prefix
            for k, v in xdem_metrics.items():
                metrics[f'xdem_{k}'] = v
                metrics[k] = v  # Also store without prefix for PES calculations
        except Exception as e:
            print(f"Warning: Could not calculate xDEM metrics: {str(e)}")
            return None, None, None, None
    
        # Calculate other metrics that don't depend on xDEM
        metrics['tri'] = self.calculate_tri(elevation)
    
        # Calculate PES metrics using xDEM-derived values
        try:
            pes_metrics = self.calculate_pes_metrics(elevation, dx, metrics)
            metrics.update(pes_metrics)
        except Exception as e:
            print(f"Warning: Could not calculate PES metrics: {str(e)}")
            
        # Route flow 
        pfr = PriorityFloodFlowRouter(mg)
        pfr.run_one_step()
        
        # Add drainage area field before finding channels
        metrics['drainage_area'] = mg.at_node['drainage_area']  # Store the full drainage area array

        # Find channels
        threshold = initial_threshold
        min_threshold = 100
        cp = None
        while threshold >= min_threshold:
            try:
                cp = ChannelProfiler(mg, minimum_channel_threshold=threshold,
                                   number_of_watersheds=num_watersheds,
                                   main_channel_only=False)
                cp.run_one_step()
                break
            except ValueError:
                threshold //= 2
    
        time_step = len(self.channel_profilers)
        self.channel_profilers[time_step] = cp
        self.grids[time_step] = mg
    
        return mg, cp, metrics, state['time']

    
    def get_channel_profiler(self, timestep):
        """Get the ChannelProfiler object for a specific timestep."""
        return self.channel_profilers.get(timestep)
    
    def get_grid(self, timestep):
        """Get the grid object for a specific timestep."""
        return self.grids.get(timestep)
    
    def get_available_metrics(self):
        """
        Get all available metrics from the calculator.
        
        Returns
        -------
        list
            List of available metric names
        """
        if not hasattr(self, 'all_timestep_metrics') or not self.all_timestep_metrics:
            raise ValueError("No metrics data available. Did you run process_all_timesteps()?")
        
        first_watershed = list(self.all_timestep_metrics[0]['metrics'].values())[0]
        return [key for key in first_watershed.keys() 
                if key not in ['mask', 'outlet_id']]
    
    def process_and_plot_scenarios(self, scenarios, base_path, num_watersheds=6, 
                                 boundaries=False, channels=False, metrics='all', 
                                 save_plots=False, make_video=False, output_base_dir=None,
                                 use_xdem=False):
        """
        Process and plot metrics for multiple scenarios.
        
        Parameters
        ----------
        scenarios :         list | List of scenario numbers to process
        base_path :         str | Base path to scenario directories
        num_watersheds :    int, optional | Number of watersheds to process, default=6
        boundaries :        bool, optional | Whether to show watershed boundaries, default=False
        channels :          bool, optional | Whether to show channels, default=False
        metrics :           str or list, optional
                            Which metrics to plot. Options:
                                - 'all': plot all available metrics
                                - 'basic': plot basic metrics (tri, slope)
                                - 'curvature': plot curvature-related metrics
                                - 'energy': plot energy-related metrics (PES, ISED)
                                - 'xdem': plot xDEM-calculated metrics
                                - list: specific list of metrics to plot
        save_plots :        bool, optional | Whether to save plots, default=False
        make_video :        bool, optional | Whether to create videos, default=False
        output_base_dir :   str, optional | Directory for saving outputs
        use_xdem :          bool, optional | Whether to calculate xDEM metrics, default=False
        """
        from tqdm import tqdm
        import os
        import imageio
        
        # Define metric groups
        metric_groups = {
            'basic': ['tri', 'slope', 'drainage_area'],
            'curvature': ['profile_curvature', 'planform_curvature', 'regular_curvature'],
            'energy': ['PES', 'ISED', 'PES_planform_normal_curv', 'PES_profile_normal_curv', 
                      'PESD', 'PESe'],
            'xdem': ['xdem_curvature', 'xdem_planform_curvature', 'xdem_profile_curvature', 
                    'xdem_slope']
        }
        
        # Validate saving options
        if (save_plots or make_video) and output_base_dir is None:
            raise ValueError("output_base_dir must be provided if save_plots or make_video is True")
        
        # Define xDEM metrics to calculate if needed
        xdem_metrics_list = ['slope', 'curvature', 'planform_curvature', 'profile_curvature']
        
        for scen_num in tqdm(scenarios, desc="Processing scenarios"):
            print(f"\nProcessing scenario {scen_num}")
            path = os.path.join(base_path, f'scenario{scen_num}', 
                              f'scenario_{scen_num}_output/grid_states.pkl')
            
            calc = WatershedMetricsCalculator(path)
            
            # Only calculate xDEM metrics if needed
            should_use_xdem = use_xdem or (
                isinstance(metrics, str) and 
                (metrics in ['all', 'xdem'] or 
                 (isinstance(metrics, (list, tuple)) and 
                  any(m.startswith('xdem_') for m in metrics)))
            )
            
            calc.process_all_timesteps(num_watersheds=6, 
                                     use_xdem=should_use_xdem, 
                                     xdem_metrics=xdem_metrics_list)
            
            # Get available metrics for this scenario
            available_metrics = calc.get_available_metrics()
            
            # Determine which metrics to plot
            if isinstance(metrics, str):
                if metrics == 'all':
                    selected_metrics = available_metrics
                elif metrics in metric_groups:
                    # Only include xdem metrics if use_xdem is True
                    group_metrics = metric_groups[metrics]
                    if not use_xdem:
                        group_metrics = [m for m in group_metrics if not m.startswith('xdem_')]
                    selected_metrics = [m for m in group_metrics if m in available_metrics]
                    
                    if not selected_metrics:
                        print(f"Warning: No metrics from group '{metrics}' found")
                        continue
                    print(f"Plotting metrics from group '{metrics}': {selected_metrics}")
                else:
                    raise ValueError(f"Unknown metrics option: {metrics}. "
                                  f"Available options: 'all', {list(metric_groups.keys())}")
            elif isinstance(metrics, (list, tuple)):
                selected_metrics = [m for m in metrics if m in available_metrics]
                if not selected_metrics:
                    print(f"Warning: No valid metrics found for scenario {scen_num}")
                    continue
            
            print(f"Selected metrics for plotting: {selected_metrics}")
            
            # Plot each metric
            for metric in tqdm(selected_metrics, desc=f"Plotting metrics for scenario {scen_num}"):
                try:
                    if save_plots or make_video:
                        metric_dir = os.path.join(output_base_dir, metric)
                        os.makedirs(metric_dir, exist_ok=True)
                    
                    frames = []
                    # Plot each timestep
                    for timestep in range(len(calc.all_timestep_metrics)):
                        if save_plots or make_video:
                            save_path = os.path.join(metric_dir, f"timestep_{timestep:04d}.png")
                        else:
                            save_path = None
                        
                        calc.plot_watershed_metric(
                            timestep=timestep,
                            metric_name=metric,
                            show_boundaries=boundaries,
                            show_channels=channels,
                            save_path=save_path
                        )
                        
                        if make_video and save_path:
                            frames.append(imageio.imread(save_path))
                        
                        if not save_plots:
                            plt.close()
                            
                except Exception as e:
                    print(f"Error processing metric {metric}: {str(e)}")
                    continue
    
            # Make videos if requested
            if make_video:
                for metric in selected_metrics:
                    video_path = os.path.join(output_base_dir, f"{metric}_evolution.mp4")
                    frames = []
                    metric_dir = os.path.join(output_base_dir, metric)
                    
                    # Collect all frames
                    for timestep in range(len(calc.all_timestep_metrics)):
                        frame_path = os.path.join(metric_dir, f"timestep_{timestep:04d}.png")
                        if os.path.exists(frame_path):
                            frames.append(imageio.imread(frame_path))
                    
                    if frames:
                        imageio.mimsave(video_path, frames, fps=5)
                        print(f"Saved video: {video_path}")
                        
    
    """
    # Usage examples:
    scenarios = [10, 11, 12, 13, 21, 22, 23, 31, 32, 33]
    base_path = '/Users/csdmsuser/Documents/Research/CU/GBE/Magnitude_Frequency_Experiments'
    
    # Plot all metrics
    process_and_plot_scenarios(scenarios, base_path, metrics='all')
    
    # Plot only basic metrics
    process_and_plot_scenarios(scenarios, base_path, metrics='basic')
    
    # Plot only curvature metrics
    process_and_plot_scenarios(scenarios, base_path, metrics='curvature')
    
    # Plot only energy-related metrics
    process_and_plot_scenarios(scenarios, base_path, metrics='energy')
    
    # Plot specific metrics
    process_and_plot_scenarios(scenarios, base_path, 
                             metrics=['tri', 'slope', 'ISED'])
    
    # Plot basic metrics and save
    process_and_plot_scenarios(scenarios, base_path,
                             metrics='basic',
                             save_plots=True,
                             output_base_dir='output_plots')
    """
    def plot_scenario_pdfs(self, scenarios_data, metric_name, timestep=-1, watershed='watershed_1', 
                          scenarios='all', normalize_y=True, show_stats=True, show_hist=False,
                          scale='linear', focus_range=None, standardize=False,
                          figsize=(15, 10), save_path=None):
        """
        Plot PDFs with flexible scaling options for better visualization.
        Now includes support for xDEM metrics.
        
        Parameters
        ----------
        scenarios_data :    dict | Dictionary of scenario data, keyed by scenario name
        metric_name :       str | Name of the metric to analyze (including xdem_ prefix if applicable)
        timestep :          int, optional | Timestep to analyze (default: -1, last timestep)
        watershed :         str or list, optional | 'all' or list of watershed names to include
        scenarios :         str or list, optional | 'all' to plot all scenarios, or list of scenario names/numbers
        normalize_y :       bool, optional | Whether to normalize the y-axis to peak of 1 (default: True)
        show_stats :        bool, optional |  Whether to show summary statistics
        show_hist :         bool, optional | Whether to show underlying histograms
        scale :             str, optional | Scale for x-axis: 'linear', 'log', 'symlog'
        focus_range :       tuple, optional | (min, max) values to focus on in the plot
        standardize :       bool, optional | Whether to standardize values before plotting
        figsize :           tuple, optional | Figure size (width, height)
        save_path :         str, optional | Path to save the figure
        """
        from scipy import stats
        
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2 if show_stats else 1, width_ratios=[3, 1] if show_stats else [1])
        ax_pdf = plt.subplot(gs[:, 0])
        
        # Define colors based on second digit (scenario number within each group)
        cmap_SA2 = cm.get_cmap('viridis')
        series_colors = {
            '1': cmap_SA2(0.2),  # All x1 scenarios (11, 21, 31)
            '2': cmap_SA2(0.5),  # All x2 scenarios (12, 22, 32)
            '3': cmap_SA2(0.8)   # All x3 scenarios (13, 23, 33)
        }
        
        # Define line styles based on first digit (series)
        line_styles = {
            '1': '-',    # Series 1 (11, 12, 13): solid
            '2': '--',   # Series 2 (21, 22, 23): dashed
            '3': ':'     # Series 3 (31, 32, 33): dotted
        }
        
        # Process scenario selection
        if scenarios == 'all':
            selected_scenarios = list(scenarios_data.keys())
        else:
            if isinstance(scenarios[0], int):
                selected_scenarios = [f'scenario_{num}' for num in scenarios]
            else:
                selected_scenarios = scenarios
            
            missing_scenarios = [s for s in selected_scenarios if s not in scenarios_data]
            if missing_scenarios:
                raise ValueError(f"Scenarios not found in data: {missing_scenarios}")
    
        # Calculate global statistics if standardizing
        if standardize:
            all_values = []
            for scenario_data in scenarios_data.values():
                values = scenario_data[timestep]['metrics'][watershed][metric_name]
                valid_values = values[np.isfinite(values)]
                all_values.extend(valid_values)
            global_mean = np.mean(all_values)
            global_std = np.std(all_values)
    
        # Process each selected scenario
        all_stats = {}
        all_raw_stats = {}
        max_density = 0
        last_scenario_data = None
    
        # Sort scenarios to ensure scenario 10 is plotted first
        selected_scenarios = sorted(selected_scenarios, 
                                  key=lambda x: (x != 'scenario_10', x))
    
        for scenario_name in selected_scenarios:
            scenario_data = scenarios_data[scenario_name]
            last_scenario_data = scenario_data
            
            try:
                # Get values and filter out NaN/inf values
                values = scenario_data[timestep]['metrics'][watershed][metric_name]
                valid_mask = np.isfinite(values)
                if not np.any(valid_mask):
                    print(f"Warning: No valid data for {scenario_name}")
                    continue
                    
                values = values[valid_mask]
                raw_values = values.copy()
                
                # Standardize if requested
                if standardize:
                    values = (values - global_mean) / global_std
                
                # Calculate kernel density estimation
                kde = stats.gaussian_kde(values)
                
                # Determine x-range
                if focus_range is not None:
                    x_min, x_max = focus_range
                else:
                    all_values = []
                    for scenario_data in scenarios_data.values():
                        values = scenario_data[timestep]['metrics'][watershed][metric_name]
                        all_values.extend(values[np.isfinite(values)])
                    
                    max_abs = np.max(np.abs(all_values))
                    x_min, x_max = -max_abs * 1.1, max_abs * 1.1
                
                x_range = np.linspace(x_min, x_max, 200)
                density = kde(x_range)
                
                if normalize_y:
                    density = density / np.max(density)
                
                max_density = max(max_density, np.max(density))
                
                # Special handling for scenario 10
                if scenario_name == 'scenario_10':
                    color = 'black'
                    linestyle = '-'
                    label = f'Scenario 10'
                else:
                    # Get series and scenario numbers from name
                    series_num = scenario_name[-2]      # First digit
                    scenario_in_series = scenario_name[-1]  # Second digit
                    
                    # Use second digit for color, first digit for line style
                    color = series_colors[scenario_in_series]
                    linestyle = line_styles[series_num]
                    label = f'Scenario {series_num}{scenario_in_series}'
                
                # Plot PDF with appropriate color and line style
                ax_pdf.plot(x_range, density, linestyle=linestyle, color=color, 
                           label=label, linewidth=2)
    
                # Store statistics
                all_raw_stats[scenario_name] = {
                    'mean': np.mean(raw_values),
                    'median': np.median(raw_values),
                    'std': np.std(raw_values),
                    'p5': np.percentile(raw_values, 5),
                    'p95': np.percentile(raw_values, 95)
                }
                
                all_stats[scenario_name] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'p5': np.percentile(values, 5),
                    'p95': np.percentile(values, 95)
                }
                
            except KeyError as e:
                print(f"Warning: Could not process {scenario_name}: {str(e)}")
                continue
    
        # Set scale
        if scale == 'log':
            ax_pdf.set_xscale('log')
        elif scale == 'symlog':
            ax_pdf.set_xscale('symlog')
        
        # Customize plot
        if metric_name.startswith('xdem_'):
            base_metric = metric_name.replace('xdem_', '')
            if base_metric == 'slope':
                xlabel = 'Slope (degrees)'
            elif 'curvature' in base_metric:
                xlabel = 'Curvature (1/100m)'
            else:
                xlabel = metric_name
        else:
            xlabel = metric_name
        
        ax_pdf.set_xlabel(xlabel + (' (standardized)' if standardize else ''), fontsize=12)
        ax_pdf.set_ylabel('Normalized Probability Density' if normalize_y else 'Probability Density', 
                       fontsize=12)
        
        if last_scenario_data is not None:
            elapsed_time = last_scenario_data[timestep]['elapsed_time']
            if isinstance(elapsed_time, np.ndarray):
                elapsed_time = elapsed_time[0]
            elapsed_time = float(elapsed_time)
            
            ax_pdf.set_title(f'Distribution of {metric_name}\n{watershed} at Time: {elapsed_time:,.0f} years', 
                           fontsize=14)
        
        ax_pdf.grid(True, alpha=0.3)
        
        if normalize_y:
            ax_pdf.set_ylim(0, 1.1)
        
        # Add legend with custom sorting
        handles, labels = ax_pdf.get_legend_handles_labels()
        # Sort all entries except "Scenario 10"
        non_eq_indices = [i for i, label in enumerate(labels) if 'Scenario 10' not in label]
        eq_indices = [i for i, label in enumerate(labels) if 'Scenario 10' in label]
        
        # Sort by scenario number
        sorted_non_eq = sorted(non_eq_indices, 
                              key=lambda i: int(labels[i].split()[-1]))
        
        # Combine equilibrium and sorted non-equilibrium indices
        sorted_indices = eq_indices + sorted_non_eq
        
        handles = [handles[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        ax_pdf.legend(handles, labels, loc='upper right')
    
        # Add statistics if requested
        if show_stats:
            ax_stats = plt.subplot(gs[:, 1])
            stats_text = f"Summary Statistics for {watershed}:\n\n"
            
            # Sort scenarios for stats, keeping scenario 10 first
            sorted_scenarios = sorted(all_raw_stats.keys(), 
                                    key=lambda x: (x != 'scenario_10', 
                                                 x.split('_')[1][0] if len(x.split('_')[1]) > 1 else '0',
                                                 x.split('_')[1][1] if len(x.split('_')[1]) > 1 else '0'))
            
            for scenario_name in sorted_scenarios:
                stats = all_raw_stats[scenario_name]
                if scenario_name == 'scenario_10':
                    scenario_label = 'Scenario 10'
                else:
                    series_num = scenario_name[-2]
                    scenario_num = scenario_name[-1]
                    scenario_label = f'Scenario {series_num}{scenario_num}'
                    
                stats_text += f"\n{scenario_label}:\n"
                stats_text += f"  Mean: {stats['mean']:.4f}\n"
                stats_text += f"  Median: {stats['median']:.4f}\n"
                stats_text += f"  Std: {stats['std']:.4f}\n"
                stats_text += f"  5th-95th: [{stats['p5']:.4f}, {stats['p95']:.4f}]\n"
            
            ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                         verticalalignment='top', fontsize=10)
            ax_stats.axis('off')
            
        ax_pdf.tick_params(axis='x', labelsize=15)
        ax_pdf.tick_params(axis='y', labelsize=15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return all_raw_stats

    
# =============================================================================
#     def plot_scenario_pdfs_old(self, scenarios_data, metric_name, timestep=-1, watershed='watershed_1', 
#                           scenarios = 'all', normalize_y=True, show_stats=True, show_hist=False,
#                           scale='linear', focus_range=None, standardize=False,
#                           figsize=(15, 10), save_path=None):
#         """
#         Plot PDFs with flexible scaling options for better visualization.
#         
#         Parameters
#         ----------
#         scenarios_data :    dict ; Dictionary of scenario data, keyed by scenario name
#         metric_name :       str ;  Name of the metric to analyze
#         timestep :          int, optional ; Timestep to analyze (default: -1, last timestep)
#         watersheds :        str or list, optional ; all' or list of watershed names to include
#         scenarios :         str or list, optional ; 'all' to plot all scenarios, or list of scenario names to plot 
#                             (e.g., ['scenario_10', 'scenario_11']) or list of scenario numbers ([10, 11])
#         normalize_y :       bool, optional ; Whether to normalize the y-axis to peak of 1 (default: True)
#         show_stats :        bool, optional ; Whether to show summary statistics
#         show_hist :         bool, optional ; Whether to show underlying histograms
#         figsize :           tuple, optional ; Figure size (width, height)
#         save_path :         str, optional ;  Path to save the figure
#         scale :             str, optional ; Scale for x-axis: 'linear', 'log', 'symlog'
#         focus_range :       tuple, optional ; (min, max) values to focus on in the plot
#         standardize :       bool, optional ;  Whether to standardize values before plotting (z-score)
#         """
#         from scipy import stats
#         
#         # Create figure
#         fig = plt.figure(figsize=figsize)
#         gs = GridSpec(2, 2 if show_stats else 1, width_ratios=[3, 1] if show_stats else [1])
#         ax_pdf = plt.subplot(gs[:, 0])
#         
#         # Define colors for scenarios (unchanged)
#         scenario_colors = {
#             'scenario_10': '#1f77b4',
#             'scenario_11': '#2ca02c',
#             'scenario_12': '#9467bd',
#             'scenario_13': '#ff7f0e',
#             'scenario_21': '#e377c2',
#             'scenario_22': '#8c564b',
#             'scenario_23': '#17becf',
#             'scenario_31': '#bcbd22',
#             'scenario_32': '#7f7f7f',
#             'scenario_33': '#d62728'
#         }
#         # Process scenario selection
#         if scenarios == 'all':
#             selected_scenarios = list(scenarios_data.keys())
#         else:
#             if isinstance(scenarios[0], int):
#                 # Convert scenario numbers to full names
#                 selected_scenarios = [f'scenario_{num}' for num in scenarios]
#             else:
#                 selected_scenarios = scenarios
#             
#             # Verify all requested scenarios exist
#             missing_scenarios = [s for s in selected_scenarios if s not in scenarios_data]
#             if missing_scenarios:
#                 raise ValueError(f"Scenarios not found in data: {missing_scenarios}")
# 
#         # Initialize tracking variables
#         all_stats = {}
#         all_raw_stats = {}  # For storing pre-standardized statistics
#         max_density = 0
#         last_scenario_data = None
#         
#         # Calculate global statistics if standardizing
#         if standardize:
#             all_values = []
#             for scenario_data in scenarios_data.values():
#                 values = scenario_data[timestep]['metrics'][watershed][metric_name]
#                 all_values.extend(values)
#             global_mean = np.mean(all_values)
#             global_std = np.std(all_values)
#         
#         # Process each selected scenario
#         for scenario_name in selected_scenarios:
#             scenario_data = scenarios_data[scenario_name]
#             last_scenario_data = scenario_data
#             
#             try:
#                 # Get values
#                 values = scenario_data[timestep]['metrics'][watershed][metric_name]
#                 raw_values = values.copy()
#                 
#                 # Standardize if requested
#                 if standardize:
#                     values = (values - global_mean) / global_std
#                 
#                 # Calculate kernel density estimation
#                 kde = stats.gaussian_kde(values)
#                 
#                 # Determine x-range based on scale and focus_range
#                 if focus_range is not None:
#                     x_min, x_max = focus_range
#                 else:
#                     x_min, x_max = np.min(values), np.max(values)
#                 
#                 x_range = np.linspace(x_min, x_max, 200)
#                 density = kde(x_range)
#                 
#                 # Normalize density to peak of 1 if requested
#                 if normalize_y:
#                     density = density / np.max(density)
#                 
#                 max_density = max(max_density, np.max(density))
#                 
#                 # Plot PDF
#                 color = scenario_colors.get(scenario_name, '#333333')
#                 ax_pdf.plot(x_range, density, color=color, 
#                          label=scenario_name, alpha=0.7, linewidth=2)
#                 
#                 if show_hist:
#                     hist_counts, bins = np.histogram(values, bins=50, density=True)
#                     if normalize_y and np.max(hist_counts) > 0:
#                         weights = np.ones_like(values) / np.max(hist_counts)
#                     else:
#                         weights = None
#                     ax_pdf.hist(values, bins=50, weights=weights,
#                              alpha=0.1, color=color, edgecolor='none')
#                 
#                 # Store statistics
#                 all_raw_stats[scenario_name] = {
#                     'mean': np.mean(raw_values),
#                     'median': np.median(raw_values),
#                     'std': np.std(raw_values),
#                     'p5': np.percentile(raw_values, 5),
#                     'p95': np.percentile(raw_values, 95)
#                 }
#                 
#                 all_stats[scenario_name] = {
#                     'mean': np.mean(values),
#                     'median': np.median(values),
#                     'std': np.std(values),
#                     'p5': np.percentile(values, 5),
#                     'p95': np.percentile(values, 95)
#                 }
#                 
#             except KeyError as e:
#                 print(f"Warning: Could not process {scenario_name}: {str(e)}")
#                 continue
#     
#         # Set scale
#         if scale == 'log':
#             ax_pdf.set_xscale('log')
#         elif scale == 'symlog':
#             ax_pdf.set_xscale('symlog')
#         
#         # Customize plot
#         ax_pdf.set_xlabel(f'{metric_name}{" (standardized)" if standardize else ""}', fontsize=12)
#         ax_pdf.set_ylabel('Normalized Probability Density' if normalize_y else 'Probability Density', 
#                        fontsize=12)
#         
#         if last_scenario_data is not None:
#             ax_pdf.set_title(f'Distribution of {metric_name}\n{watershed} at Time: {last_scenario_data[timestep]["elapsed_time"]:,} years', 
#                            fontsize=14)
#         
#         ax_pdf.grid(True, alpha=0.3)
#         
#         if normalize_y:
#             ax_pdf.set_ylim(0, 1.1)
#         
#         # Add legend
#         handles, labels = ax_pdf.get_legend_handles_labels()
#         sorted_indices = sorted(range(len(labels)), key=lambda k: int(labels[k].split('_')[1]))
#         handles = [handles[i] for i in sorted_indices]
#         labels = [labels[i] for i in sorted_indices]
#         ax_pdf.legend(handles, labels, loc='upper right')
#         
#         # Add statistics
#         if show_stats:
#             ax_stats = plt.subplot(gs[:, 1])
#             stats_text = f"Summary Statistics for {watershed}:\n\n"
#             
#             sorted_scenarios = sorted(all_raw_stats.keys(), key=lambda x: int(x.split('_')[1]))
#             
#             for scenario_name in sorted_scenarios:
#                 stats = all_raw_stats[scenario_name]  # Use raw stats for display
#                 stats_text += f"\n{scenario_name}:\n"
#                 stats_text += f"  Mean: {stats['mean']:.4f}\n"
#                 stats_text += f"  Median: {stats['median']:.4f}\n"
#                 stats_text += f"  Std: {stats['std']:.4f}\n"
#                 stats_text += f"  5th-95th: [{stats['p5']:.4f}, {stats['p95']:.4f}]\n"
#             
#             ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
#                          verticalalignment='top', fontsize=10)
#             ax_stats.axis('off')
#         
#         plt.tight_layout()
#         
#         if save_path:
#             plt.savefig(save_path, bbox_inches='tight', dpi=300)
#         
#         plt.show()
#         
#         return all_raw_stats
# =============================================================================
    
    def compare_scenarios_over_time(self, scenarios_data, metric_name, watershed='watershed_1',
                                  normalize='zscore', num_time_points=10, figsize=(15, 10),
                                  save_path=None):
        """
        Create an animation or multi-panel plot showing PDF evolution over time.
        
        Parameters
        ----------
        scenarios_data :    dict | Dictionary of scenario data
        metric_name :       str | Name of the metric to analyze
        watershed :         str, optional | Which watershed to analyze (default: 'watershed_1')
        normalize :         str, optional | Normalization method ('peak', 'area', 'zscore', or None)
        num_time_points :   int, optional | Number of time points to show
        figsize :           tuple, optional | Figure size
        save_path :         str, optional | Path to save the figure
        """
        fig = plt.figure(figsize=figsize)
        
        # Create time points to sample
        for scenario_name, scenario_data in scenarios_data.items():
            total_steps = len(scenario_data)
            time_indices = np.linspace(0, total_steps-1, num_time_points, dtype=int)
            
            for idx, timestep in enumerate(time_indices, 1):
                ax = plt.subplot(int(np.ceil(num_time_points/3)), 3, idx)
                
                # Get elapsed time and handle array case
                elapsed_time = scenario_data[timestep]["elapsed_time"]
                if isinstance(elapsed_time, np.ndarray):
                    elapsed_time = elapsed_time[0]
                elapsed_time = float(elapsed_time)
                
                ax.set_title(f'Time: {elapsed_time:,.0f} years')
                
                # Get values for this timestep
                values = scenario_data[timestep]['metrics'][watershed][metric_name]
                
                # Normalize if requested
                if normalize == 'zscore':
                    values = (values - np.mean(values)) / np.std(values)
                elif normalize == 'peak':
                    values = values / np.max(np.abs(values))
                
                # Calculate and plot PDF
                kernel = stats.gaussian_kde(values)
                x_range = np.linspace(min(values), max(values), 200)
                density = kernel(x_range)
                
                if normalize == 'area':
                    density = density / np.trapz(density, x_range)
                elif normalize == 'peak':
                    density = density / np.max(density)
                
                ax.plot(x_range, density, label=scenario_name)
                ax.set_title(f'Time: {scenario_data[timestep]["elapsed_time"]:,} years')
                ax.grid(True, alpha=0.3)
                
                if idx % 3 == 1:
                    ax.set_ylabel('Probability Density')
                if idx > (num_time_points - 3):
                    ax.set_xlabel(f'{metric_name} {"(normalized)" if normalize else ""}')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.suptitle(f'Evolution of {metric_name} Distribution\nWatershed: {watershed}', 
                     fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()
    
    def extract_watershed_metrics(self, mg, cp, metrics):
        """Extract watershed metrics silently."""
        watershed_metrics = {}
        outlet_nodes = list(cp.data_structure.keys())
        
        for i, outlet_id in enumerate(outlet_nodes):
            watershed_mask = get_watershed_mask(mg, outlet_id)
            watershed_metrics[f'watershed_{i+1}'] = {
                'outlet_id': outlet_id,
                'mask': watershed_mask,
                'drainage_area': mg.at_node['drainage_area'][watershed_mask],  # Store per-cell drainage areas
                'total_drainage_area': np.sum(mg.at_node['drainage_area'][watershed_mask]) * (mg.dx ** 2)  # Keep total area too
            }
            
            for metric_name, metric_values in metrics.items():
                if isinstance(metric_values, np.ndarray):
                    watershed_metrics[f'watershed_{i+1}'][metric_name] = metric_values.flatten()[watershed_mask]
        
        return watershed_metrics
    
    def process_all_timesteps(self, num_watersheds=6, initial_threshold=1000, use_xdem=False, xdem_metrics=None):
        """
        Process all timesteps and store comprehensive metrics.
        """
        print(f"Starting timestep processing for Scenario {self.scenario_num}...")  # Initial message
        self.all_timestep_metrics = []
        self.processed_data = {}  # New structure for plotting
        
        # Process each timestep
        for time_step, state in enumerate(self.grid_states):
            mg, cp, metrics, elapsed_time = self.process_grid_state(state, num_watersheds, initial_threshold)
            
            if cp is not None:
                watershed_metrics = self.extract_watershed_metrics(mg, cp, metrics)
                
                # Add xDEM metrics if requested
                if use_xdem:
                    xdem_calcs = self.calculate_xdem_metrics(mg, include_metrics=xdem_metrics)
                    for watershed in watershed_metrics:
                        mask = watershed_metrics[watershed]['mask']
                        for metric_name, metric_values in xdem_calcs.items():
                            if hasattr(metric_values, 'filled'):
                                values = metric_values.filled(np.nan)[mask]
                            else:
                                values = metric_values[mask]
                            watershed_metrics[watershed][f'xdem_{metric_name}'] = values
                            
                            # Store in new structure
                            if f'xdem_{metric_name}' not in self.processed_data:
                                self.processed_data[f'xdem_{metric_name}'] = {}
                            if watershed not in self.processed_data[f'xdem_{metric_name}']:
                                self.processed_data[f'xdem_{metric_name}'][watershed] = []
                            self.processed_data[f'xdem_{metric_name}'][watershed].append({
                                'time': elapsed_time,
                                'values': values,
                                'mask': mask
                            })
                
                # Store traditional metrics in new structure
                for metric_name in metrics:
                    if metric_name not in self.processed_data:
                        self.processed_data[metric_name] = {}
                    for watershed in watershed_metrics:
                        if watershed not in self.processed_data[metric_name]:
                            self.processed_data[metric_name][watershed] = []
                        self.processed_data[metric_name][watershed].append({
                            'time': elapsed_time,
                            'values': watershed_metrics[watershed][metric_name],
                            'mask': watershed_metrics[watershed]['mask']
                        })
                
                # Maintain backward compatibility
                self.all_timestep_metrics.append({
                    'time_step': time_step,
                    'elapsed_time': elapsed_time,
                    'metrics': watershed_metrics
                })
            
            self._print_progress(time_step + 1, len(self.grid_states))
        print("\nProcessing complete!")  # Final message
        
        return self.all_timestep_metrics
    
    def plot_scenarios(self, metrics='all', num_watersheds=6, boundaries=False, channels=False, 
                      save_plots=False, make_video=False, output_base_dir=None):
        """Plot processed metrics with progress bar."""
        if not hasattr(self, 'processed_data'):
            raise ValueError("Must run process_all_timesteps before plotting")
        
        metric_groups = {
            'basic': ['tri', 'slope', 'drainage_area'],
            'curvature': ['profile_curvature', 'planform_curvature', 'regular_curvature'],
            'energy': ['PES', 'ISED', 'PES_planform_normal_curv', 'PES_profile_normal_curv', 
                      'PESD', 'PESe'],
            'xdem': [m for m in self.processed_data.keys() if m.startswith('xdem_')]
        }
        
        if isinstance(metrics, str):
            if metrics == 'all':
                selected_metrics = list(self.processed_data.keys())
            elif metrics in metric_groups:
                selected_metrics = [m for m in metric_groups[metrics] 
                                  if m in self.processed_data]
            else:
                raise ValueError(f"Unknown metrics option: {metrics}")
        else:
            selected_metrics = [m for m in metrics if m in self.processed_data]
        
        total_plots = len(selected_metrics) * len(self.all_timestep_metrics)
        plot_count = 0
        
        print(f"\nPlotting metrics for Scenario {self.scenario_num}...")
        
        for metric in selected_metrics:
            for timestep in range(len(self.all_timestep_metrics)):
                plot_count += 1
                self._print_progress(plot_count, total_plots, prefix='Plotting:')
                
                if save_plots:
                    save_path = os.path.join(output_base_dir, metric, 
                                           f"timestep_{timestep:04d}.png")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                else:
                    save_path = None
                
                self.plot_watershed_metric(
                    timestep=timestep,
                    metric_name=metric,
                    num_watersheds=num_watersheds,
                    show_boundaries=boundaries,
                    show_channels=channels,
                    save_path=save_path
                )
        
        print("\nPlotting complete!")    
        
    def plot_metric_comparison(self, metric_name, scenarios, timestep=-1, watershed='watershed_1', 
                             figsize=(20, 5), save_path=None, normalize_pdfs=True,
                             show_stats=False, standardize=False, focus_range=None,
                             base_path=None):
        """
        Create a single figure comparing spatial distributions and PDFs for a given metric.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric to analyze (e.g., 'tri', 'slope', 'ISED')
        scenarios : list
            List of scenario numbers to compare (e.g., [11, 12, 13])
        timestep : int
            Timestep to analyze (default: -1, last timestep)
        watershed : str
            Watershed to analyze for PDFs (default: 'watershed_1')
        figsize : tuple
            Figure size in inches (width, height)
        save_path : str
            Path to save the figure
        normalize_pdfs : bool
            Whether to normalize PDFs to peak of 1
        show_stats : bool
            Whether to show statistics with PDFs
        standardize : bool
            Whether to standardize values before plotting PDFs
        focus_range : tuple
            (min, max) range for PDFs
        base_path : str
            Base directory containing scenario folders
        """
        # Define colors for PDF lines based on second digit
        cmap_SA2 = cm.get_cmap('viridis')
        digit_colors = {
            '1': cmap_SA2(0.2),  # All x1 scenarios
            '2': cmap_SA2(0.5),  # All x2 scenarios
            '3': cmap_SA2(0.8)   # All x3 scenarios
        }
        
        # Define line styles based on first digit (series)
        series_styles = {
            '1': '-',    # Series 1: solid
            '2': '--',   # Series 2: dashed
            '3': ':'     # Series 3: dotted
        }
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Convert scenarios to correct format
        scenario_names = [f'scenario_{num}' for num in scenarios]
        scenarios_data = {}
        all_values = []
        
        # Determine base path if not provided
        if base_path is None:
            try:
                base_path = os.path.dirname(os.path.dirname(self.grid_states_path))
            except AttributeError:
                raise ValueError("base_path must be provided if grid_states_path is not available")
        
        # Load and process data for each scenario
        for scenario_num in scenarios:
            # Construct path for this scenario
            scenario_path = os.path.join(base_path, f'scenario{scenario_num}', 
                                       f'scenario_{scenario_num}_output/grid_states.pkl')
            
            if not os.path.exists(scenario_path):
                raise FileNotFoundError(f"Could not find scenario data at: {scenario_path}")
            
            # Create calculator for this scenario
            scenario_calc = WatershedMetricsCalculator(scenario_path)
            scenario_calc.process_all_timesteps()
            
            # Store processed data
            scenarios_data[f'scenario_{scenario_num}'] = scenario_calc.all_timestep_metrics
            
            # Collect values for colormap normalization
            timestep_data = scenario_calc.all_timestep_metrics[timestep]
            for watershed_data in timestep_data['metrics'].values():
                values = watershed_data[metric_name]
                all_values.extend(values[np.isfinite(values)])
        
        # Set up colormap
        if metric_name in self.diverging_metrics:
            vmax = np.nanmax(np.abs(all_values))
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            cmap = plt.cm.RdBu_r
        else:
            vmin, vmax = np.nanpercentile(all_values, [2, 98])
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.viridis
        
        # Plot spatial distributions
        for idx, scenario_num in enumerate(scenarios, 1):
            ax = plt.subplot(1, 4, idx)
            
            # Get data for this specific scenario
            scenario_data = scenarios_data[f'scenario_{scenario_num}']
            timestep_data = scenario_data[timestep]
            
            # Create combined grid for all watersheds
            combined_grid = np.full((100, 100), np.nan)
            
            for watershed_name, watershed_data in timestep_data['metrics'].items():
                data = watershed_data[metric_name]
                mask = watershed_data['mask']
                
                watershed_grid = np.full(100 * 100, np.nan)
                watershed_grid[mask] = data
                watershed_grid = watershed_grid.reshape((100, 100))
                
                valid_data = ~np.isnan(watershed_grid)
                combined_grid[valid_data] = watershed_grid[valid_data]
            
            # Plot with consistent colormap
            im = ax.imshow(combined_grid, cmap=cmap, norm=norm)
            plt.colorbar(im, ax=ax)
            
            # Plot watershed boundaries if available
            for watershed_name, watershed_data in timestep_data['metrics'].items():
                mask = watershed_data['mask'].reshape((100, 100))
                boundary = np.zeros_like(mask)
                boundary[1:-1, 1:-1] = mask[1:-1, 1:-1] & (
                    ~mask[:-2, 1:-1] | ~mask[2:, 1:-1] |
                    ~mask[1:-1, :-2] | ~mask[1:-1, 2:]
                )
                if np.any(boundary):
                    color = self.watershed_colors.get(watershed_name, 'black')
                    ax.contour(boundary, colors=[color], levels=[0.5], linewidths=1)
            
            ax.set_title(f'Scenario {scenario_num}')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Plot PDFs in last subplot
        ax_pdf = plt.subplot(1, 4, 4)
        
        # Sort scenarios to ensure scenario 10 is plotted first
        sorted_scenarios = sorted(scenarios_data.keys(), 
                                key=lambda x: (x != 'scenario_10', x))
    
        # Calculate PDFs for each scenario
        for scenario_name in sorted_scenarios:
            scenario_data = scenarios_data[scenario_name]
            values = scenario_data[timestep]['metrics'][watershed][metric_name]
            valid_values = values[np.isfinite(values)]
            
            if standardize:
                valid_values = (valid_values - np.mean(valid_values)) / np.std(valid_values)
            
            kde = stats.gaussian_kde(valid_values)
            if focus_range:
                x_range = np.linspace(focus_range[0], focus_range[1], 200)
            else:
                x_range = np.linspace(np.min(valid_values), np.max(valid_values), 200)
            
            density = kde(x_range)
            if normalize_pdfs:
                density = density / np.max(density)
            
            # Get color and style based on scenario number
            if scenario_name == 'scenario_10':
                color = 'black'
                style = '-'
            else:
                first_digit = scenario_name[-2]   # Series number
                second_digit = scenario_name[-1]  # Scenario within series
                color = digit_colors[second_digit]
                style = series_styles[first_digit]
            
            # Plot with appropriate color and line style
            ax_pdf.plot(x_range, density, style, color=color,
                       label=f'Scenario {scenario_name.split("_")[1]}', 
                       linewidth=2)
    
        # Add legend with smaller font and custom sorting
        handles, labels = ax_pdf.get_legend_handles_labels()
        # Sort all entries except "Scenario 10"
        non_eq_indices = [i for i, label in enumerate(labels) if 'Scenario 10' not in label]
        eq_indices = [i for i, label in enumerate(labels) if 'Scenario 10' in label]
        
        # Sort by scenario number
        sorted_non_eq = sorted(non_eq_indices, 
                              key=lambda i: int(labels[i].split()[-1]))
        
        # Combine equilibrium and sorted non-equilibrium indices
        sorted_indices = eq_indices + sorted_non_eq
        
        handles = [handles[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        ax_pdf.legend(handles, labels, loc='upper right', fontsize='small')
        
        # Get elapsed time for title - handle various cases
        elapsed_time = timestep_data['elapsed_time']
        if isinstance(elapsed_time, np.ndarray):
            # If it's an array, check if all values are the same
            if np.all(elapsed_time == elapsed_time[0]):
                elapsed_time = float(elapsed_time[0])
            else:
                # If values differ, use the maximum time
                elapsed_time = float(np.max(elapsed_time))
                print("Warning: Elapsed time varies across nodes, using maximum value")
        else:
            elapsed_time = float(elapsed_time)
        
        # Format title string
        title_str = f'Comparison of {metric_name} across scenarios\nTime: {elapsed_time:,.0f} years'
        plt.suptitle(title_str, fontsize=14, y=1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format="svg", dpi=300)

        return fig
    
    def plot_scenario_matrix(self, metric_name, scenarios, timestep=-1, watershed='watershed_1', 
                            base_path=None, figsize=(20, 20), save_path=None):
        """
        Create a matrix plot of scenarios with PDFs for rows and columns.
        Layout:
        - 3x3 grid of spatial plots (scenarios 11-33)
        - PDF plots for each row on the right
        - PDF plots for each column on the bottom
        """
        # Create figure with GridSpec
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 4, figure=fig)
    
        # Define colors and styles for PDFs
        cmap_SA2 = cm.get_cmap('viridis')
        digit_colors = {
            '1': cmap_SA2(0.2),
            '2': cmap_SA2(0.5),
            '3': cmap_SA2(0.8)
        }
        series_styles = {
            '1': '-',
            '2': '--',
            '3': ':'
        }
    
        # Determine base path if not provided
        if base_path is None:
            try:
                base_path = os.path.dirname(os.path.dirname(self.grid_states_path))
            except AttributeError:
                raise ValueError("base_path must be provided if grid_states_path is not available")
    
        # Load and process data for each scenario
        scenarios_data = {}
        all_values = []
        
        for scenario_num in scenarios:
            scenario_path = os.path.join(base_path, f'scenario{scenario_num}', 
                                       f'scenario_{scenario_num}_output/grid_states.pkl')
            
            if not os.path.exists(scenario_path):
                raise FileNotFoundError(f"Could not find scenario data at: {scenario_path}")
            
            # Create calculator for this scenario
            scenario_calc = WatershedMetricsCalculator(scenario_path)
            scenario_calc.process_all_timesteps()
            
            # Store processed data and collect values for colormap normalization
            scenarios_data[f'scenario_{scenario_num}'] = scenario_calc.all_timestep_metrics
            
            timestep_data = scenario_calc.all_timestep_metrics[timestep]
            for watershed_data in timestep_data['metrics'].values():
                values = watershed_data[metric_name]
                try:
                    # Convert values to numpy array if it isn't already
                    values = np.asarray(values, dtype=float)
                    valid_mask = np.isfinite(values)
                    all_values.extend(values[valid_mask])
                except Exception as e:
                    print(f"Warning: Error processing values for scenario {scenario_num}: {str(e)}")
    
        # Set up colormap for spatial plots
        if metric_name in self.diverging_metrics:
            vmax = np.nanmax(np.abs(all_values))
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            cmap = plt.cm.RdBu_r
        else:
            vmin, vmax = np.nanpercentile(all_values, [2, 98])
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.viridis
    
        # Plot spatial distributions
        for i in range(1, 4):
            for j in range(1, 4):
                ax = fig.add_subplot(gs[i-1, j-1])
                scenario_num = i * 10 + j
                scenario_name = f'scenario_{scenario_num}'
                
                # Get data for this specific scenario
                scenario_data = scenarios_data[scenario_name][timestep]
                
                # Create combined grid for all watersheds
                combined_grid = np.full((100, 100), np.nan)
                
                # Process each watershed
                for watershed_name, watershed_data in scenario_data['metrics'].items():
                    data = watershed_data[metric_name]
                    mask = watershed_data['mask']
                    
                    watershed_grid = np.full(100 * 100, np.nan)
                    watershed_grid[mask] = data
                    watershed_grid = watershed_grid.reshape((100, 100))
                    
                    valid_data = ~np.isnan(watershed_grid)
                    combined_grid[valid_data] = watershed_grid[valid_data]
                    
                    # Plot watershed boundary
                    mask_2d = mask.reshape((100, 100))
                    boundary = np.zeros_like(mask_2d)
                    boundary[1:-1, 1:-1] = mask_2d[1:-1, 1:-1] & (
                        ~mask_2d[:-2, 1:-1] | ~mask_2d[2:, 1:-1] |
                        ~mask_2d[1:-1, :-2] | ~mask_2d[1:-1, 2:]
                    )
                    color = self.watershed_colors.get(watershed_name, 'red')
                    ax.contour(boundary, colors=[color], levels=[0.5], linewidths=1)
                
                # Plot the combined grid
                im = ax.imshow(combined_grid, cmap=cmap, norm=norm)
                
                ax.set_title(f'Scenario {scenario_num}')
                ax.set_xticks([])
                ax.set_yticks([])
    
        # Plot row PDFs
        for i in range(1, 4):
            ax = fig.add_subplot(gs[i-1, -1])
            for j in range(1, 4):
                scenario_num = i * 10 + j
                scenario_name = f'scenario_{scenario_num}'
                
                try:
                    # Get data for watershed_1 only
                    values = scenarios_data[scenario_name][timestep]['metrics'][watershed][metric_name]
                    values = np.asarray(values, dtype=float)
                    valid_values = values[np.isfinite(values)]
                    
                    if len(valid_values) > 0:
                        kde = stats.gaussian_kde(valid_values)
                        x_range = np.linspace(min(valid_values), max(valid_values), 200)
                        density = kde(x_range)
                        density = density / np.max(density)
                        
                        ax.plot(x_range, density, series_styles[str(j)],
                               color=digit_colors[str(i)], 
                               label=f'Scenario {scenario_num}')
                except Exception as e:
                    print(f"Warning: Could not plot PDF for scenario {scenario_num}: {str(e)}")
            
            ax.legend(fontsize='small')
            ax.set_ylim(0, 1.1)
    
        # Plot column PDFs
        for j in range(1, 4):
            ax = fig.add_subplot(gs[-1, j-1])
            for i in range(1, 4):
                scenario_num = i * 10 + j
                scenario_name = f'scenario_{scenario_num}'
                
                try:
                    # Get data for watershed_1 only
                    values = scenarios_data[scenario_name][timestep]['metrics'][watershed][metric_name]
                    values = np.asarray(values, dtype=float)
                    valid_values = values[np.isfinite(values)]
                    
                    if len(valid_values) > 0:
                        kde = stats.gaussian_kde(valid_values)
                        x_range = np.linspace(min(valid_values), max(valid_values), 200)
                        density = kde(x_range)
                        density = density / np.max(density)
                        
                        ax.plot(x_range, density, series_styles[str(i)],
                               color=digit_colors[str(j)], 
                               label=f'Scenario {scenario_num}')
                except Exception as e:
                    print(f"Warning: Could not plot PDF for scenario {scenario_num}: {str(e)}")
            
            ax.legend(fontsize='small')
            ax.set_ylim(0, 1.1)
    
        # Add colorbar
        cax = fig.add_subplot(gs[-1, -1])
        plt.colorbar(im, cax=cax, orientation='horizontal', label=metric_name)
    
        # Overall title
        elapsed_time = scenario_data['elapsed_time']
        if isinstance(elapsed_time, np.ndarray):
            elapsed_time = elapsed_time[0]
        elapsed_time = float(elapsed_time)
        plt.suptitle(f'Comparison of {metric_name} across scenarios\nTime: {elapsed_time:,.0f} years',
                    fontsize=14, y=1.02)
    
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def plot_metric_comparison_old(self, metric_name, scenarios, timestep=-1, watershed='watershed_1', 
                             figsize=(20, 5), save_path=None, normalize_pdfs=True,
                             show_stats=False, standardize=False, focus_range=None,
                             base_path=None):
        """
        Create a single figure with subplots comparing spatial distributions and PDFs for a given metric.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric to analyze (e.g., 'tri', 'slope', 'ISED')
        scenarios : list
            List of scenario numbers to compare (e.g., [11, 12, 13])
        timestep : int, optional
            Timestep to analyze (default: -1, last timestep)
        watershed : str, optional
            Watershed to analyze for PDFs (default: 'watershed_1')
        figsize : tuple, optional
            Figure size in inches (width, height)
        save_path : str, optional
            Path to save the figure
        normalize_pdfs : bool, optional
            Whether to normalize PDFs to peak of 1 (default: True)
        show_stats : bool, optional
            Whether to show statistics with PDFs (default: False)
        standardize : bool, optional
            Whether to standardize values before plotting PDFs (default: False)
        focus_range : tuple, optional
            (min, max) range for PDFs (default: None, automatic)
        base_path : str, optional
            Base directory containing all scenario folders. If None, will try to determine from current path.
        """
        # Create single figure with subplots
        fig = plt.figure(figsize=figsize)
        
        # Convert scenarios to correct format
        scenario_names = [f'scenario_{num}' for num in scenarios]
        scenarios_data = {}
        all_values = []
        
        # Determine base path if not provided
        if base_path is None:
            try:
                base_path = os.path.dirname(os.path.dirname(self.grid_states_path))
            except AttributeError:
                raise ValueError("base_path must be provided if grid_states_path is not available")
        
        # Load and process data for each scenario
        for scenario_num in scenarios:
            # Construct path for this scenario
            scenario_path = os.path.join(base_path, f'scenario{scenario_num}', 
                                       f'scenario_{scenario_num}_output/grid_states.pkl')
            
            if not os.path.exists(scenario_path):
                raise FileNotFoundError(f"Could not find scenario data at: {scenario_path}")
            
            # Create calculator for this scenario
            scenario_calc = WatershedMetricsCalculator(scenario_path)
            scenario_calc.process_all_timesteps()
            
            # Store processed data
            scenarios_data[f'scenario_{scenario_num}'] = scenario_calc.all_timestep_metrics
            
            # Collect values for colormap normalization
            timestep_data = scenario_calc.all_timestep_metrics[timestep]
            for watershed_data in timestep_data['metrics'].values():
                values = watershed_data[metric_name]
                all_values.extend(values[np.isfinite(values)])
        
        # Set up colormap
        if metric_name in self.diverging_metrics:
            vmax = np.nanmax(np.abs(all_values))
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            cmap = plt.cm.RdBu_r
        else:
            vmin, vmax = np.nanpercentile(all_values, [2, 98])
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.viridis
        
        # Plot spatial distributions
        for idx, scenario_num in enumerate(scenarios, 1):
            ax = plt.subplot(1, 4, idx)
            
            # Get data for this specific scenario
            scenario_data = scenarios_data[f'scenario_{scenario_num}']
            timestep_data = scenario_data[timestep]
            
            # Create combined grid for all watersheds
            combined_grid = np.full((100, 100), np.nan)
            
            for watershed_name, watershed_data in timestep_data['metrics'].items():
                data = watershed_data[metric_name]
                mask = watershed_data['mask']
                
                watershed_grid = np.full(100 * 100, np.nan)
                watershed_grid[mask] = data
                watershed_grid = watershed_grid.reshape((100, 100))
                
                valid_data = ~np.isnan(watershed_grid)
                combined_grid[valid_data] = watershed_grid[valid_data]
            
            # Plot with consistent colormap
            im = ax.imshow(combined_grid, cmap=cmap, norm=norm)
            plt.colorbar(im, ax=ax)
            
            # Plot watershed boundaries
            for watershed_name, watershed_data in timestep_data['metrics'].items():
                mask = watershed_data['mask'].reshape((100, 100))
                boundary = np.zeros_like(mask)
                boundary[1:-1, 1:-1] = mask[1:-1, 1:-1] & (
                    ~mask[:-2, 1:-1] | ~mask[2:, 1:-1] |
                    ~mask[1:-1, :-2] | ~mask[1:-1, 2:]
                )
                if np.any(boundary):
                    color = self.watershed_colors.get(watershed_name, 'black')
                    ax.contour(boundary, colors=[color], levels=[0.5], linewidths=1)
            
            ax.set_title(f'Scenario {scenario_num}')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Plot PDFs in last subplot
        ax_pdf = plt.subplot(1, 4, 4)
        
        # Calculate PDFs for each scenario
        for scenario_name, scenario_data in scenarios_data.items():
            values = scenario_data[timestep]['metrics'][watershed][metric_name]
            valid_values = values[np.isfinite(values)]
            
            if standardize:
                valid_values = (valid_values - np.mean(valid_values)) / np.std(valid_values)
            
            kde = stats.gaussian_kde(valid_values)
            if focus_range:
                x_range = np.linspace(focus_range[0], focus_range[1], 200)
            else:
                x_range = np.linspace(np.min(valid_values), np.max(valid_values), 200)
            
            density = kde(x_range)
            if normalize_pdfs:
                density = density / np.max(density)
            
            color = plt.cm.tab10(list(scenarios_data.keys()).index(scenario_name) / len(scenarios_data))
            ax_pdf.plot(x_range, density, label=scenario_name, color=color)
        
        ax_pdf.set_xlabel(f'{metric_name} {"(standardized)" if standardize else ""}')
        ax_pdf.set_ylabel('Normalized Density' if normalize_pdfs else 'Density')
        ax_pdf.legend()
        ax_pdf.grid(True, alpha=0.3)
        
        # Add overall title
        plt.suptitle(
            f'Comparison of {metric_name} across scenarios\n'
            f'Time: {timestep_data["elapsed_time"]:,} years',
            fontsize=14, y=1.05
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return fig
# =============================================================================
#     def process_all_timesteps_old(self, num_watersheds=6, initial_threshold=1000, use_xdem=False, xdem_metrics=None):
#         """
#         Process all timesteps, optionally including xDEM calculations.
#         """
#         self.all_timestep_metrics = []
#         
#         for time_step, state in enumerate(self.grid_states):
#             # Original processing
#             mg, cp, metrics, elapsed_time = self.process_grid_state(
#                 state, num_watersheds, initial_threshold)
#             
#             if cp is not None:
#                 watershed_metrics = self.extract_watershed_metrics(mg, cp, metrics)
#                 
#                 # Add xDEM calculations if requested
#                 if use_xdem:
#                     xdem_calcs = self.calculate_xdem_metrics(mg, include_metrics=xdem_metrics)
#                     
#                     # Add xDEM metrics to each watershed
#                     for watershed in watershed_metrics:
#                         mask = watershed_metrics[watershed]['mask']
#                         for metric_name, metric_values in xdem_calcs.items():
#                             # Convert masked array to regular array and extract valid values
#                             if hasattr(metric_values, 'filled'):
#                                 values = metric_values.filled(np.nan)[mask]
#                             else:
#                                 values = metric_values[mask]
#                             watershed_metrics[watershed][f'xdem_{metric_name}'] = values
#                 
#                 self.all_timestep_metrics.append({
#                     'time_step': time_step,
#                     'elapsed_time': elapsed_time,
#                     'metrics': watershed_metrics
#                 })
#             
#             self._print_progress(time_step + 1, len(self.grid_states))
#             
#         return self.all_timestep_metrics
# =============================================================================
    
    def analyze_landslides_by_watershed(self, timestep):
        grid = self.get_grid_at_timestep(timestep)  # Get grid for this specific timestep
        timestep_data = self.get_timestep_metrics(timestep)
        
        landslide_stats = {}
        for watershed_name, watershed_data in timestep_data['metrics'].items():
            mask = watershed_data['mask']
            erosion = grid.at_node['landslide__erosion'][mask]
            erosion_volume = np.sum(erosion) * grid.dx * grid.dx
            
            landslide_stats[watershed_name] = {
                'current': {
                    'erosion_volume': erosion_volume
                }
            }
        
        return landslide_stats

    def analyze_landslides_by_watershed_old_V2(self, timestep=-1):
        """Analyze landslide statistics for each watershed at a given timestep."""
        timestep_data = self.get_timestep_metrics(timestep)
        grid = self.get_grid_at_timestep(timestep)
        dx = grid.dx
        
        landslide_stats = {}
        
        for watershed_name, watershed_data in timestep_data['metrics'].items():
            mask = watershed_data['mask']
            watershed_area = np.sum(mask) * dx * dx
            
            # Focus on erosion volumes only
            erosion = grid.at_node['landslide__erosion'][mask]
            erosion_volume = np.sum(erosion) * dx * dx
            
            print(f"\n{watershed_name}:")
            print(f"  Active erosion cells: {np.sum(erosion > 0)}")
            print(f"  Erosion volume: {erosion_volume:.2f} m³")
            
            landslide_stats[watershed_name] = {
                'watershed_area': watershed_area,
                'current': {
                    'erosion_volume': erosion_volume
                }
            }
        
        return landslide_stats

    def analyze_landslides_by_watershed_old(self, timestep=-1):
        """
        Analyze landslide statistics for each watershed at a given timestep.
        """
        timestep_data = self.get_timestep_metrics(timestep)
        grid = self.get_grid_at_timestep(timestep)
        dx = grid.dx  # grid spacing for area calculations
        
        landslide_stats = {}
        print("Debugging landslide volumes:")  # Debug print
        
        # Analyze each watershed
        for watershed_name, watershed_data in timestep_data['metrics'].items():
            mask = watershed_data['mask']
            watershed_area = np.sum(mask) * dx * dx  # total watershed area
            
            # Get landslide data for this watershed - using instantaneous fields
            erosion = grid.at_node['landslide__erosion'][mask]
            deposition = grid.at_node['landslide__deposition'][mask]
            
            # Calculate volumes
            erosion_volume = np.sum(erosion) * dx * dx
            deposition_volume = np.sum(deposition) * dx * dx
            
            print(f"{watershed_name}:")  # Debug print
            print(f"  Erosion volume: {erosion_volume}")  # Debug print
            print(f"  Deposition volume: {deposition_volume}")  # Debug print
            
            # Store statistics
            landslide_stats[watershed_name] = {
                'watershed_area': watershed_area,
                'current': {
                    'erosion_volume': erosion_volume,
                    'deposition_volume': deposition_volume,
                    'net_volume_change': deposition_volume - erosion_volume
                }
            }
            
            # Also store cumulative values if available, but check if they're zero first
            if 'cumulative_landslide_erosion' in grid.at_node:
                cumul_erosion = grid.at_node['cumulative_landslide_erosion'][mask]
                cumul_deposition = grid.at_node['cumulative_landslide_deposition'][mask]
                
                if np.any(cumul_erosion > 0) or np.any(cumul_deposition > 0):
                    landslide_stats[watershed_name]['cumulative'] = {
                        'erosion_volume': np.sum(cumul_erosion) * dx * dx,
                        'deposition_volume': np.sum(cumul_deposition) * dx * dx
                    }
        
        return landslide_stats    

    def plot_landslide_volume_comparison(self, scen_num, timestep=-1, figsize=(10, 6), save_path=None):
        """
        Create bar plot comparing landslide volumes across watersheds.
        
        Parameters
        ----------
        timestep : int, optional
            Timestep to analyze (default: -1, last timestep)
        figsize : tuple, optional
            Figure size
        save_path : str, optional
            Path to save figure
        """
        stats = self.analyze_landslides_by_watershed(timestep)
        
        # Prepare data
        watersheds = list(stats.keys())
        erosion = [-stats[w]['current']['erosion_volume'] for w in watersheds]  # Make erosion negative
        deposition = [stats[w]['current']['deposition_volume'] for w in watersheds]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Position bars
        x = np.arange(len(watersheds))
        width = 0.35
        
        # Create bars
        rects1 = ax.bar(x - width/2, erosion, width, label='Erosion', color='salmon', alpha=0.7)
        rects2 = ax.bar(x + width/2, deposition, width, label='Deposition', color='skyblue', alpha=0.7)
        
        # Add value labels on the bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height != 0:  # Only label non-zero bars
                    ax.annotate(f'{abs(height):.1e}',
                              xy=(rect.get_x() + rect.get_width()/2, height),
                              xytext=(0, 3 if height >= 0 else -3),  # 3 points vertical offset
                              textcoords="offset points",
                              ha='center', va='bottom' if height >= 0 else 'top',
                              rotation=45)
        
        autolabel(rects1)
        autolabel(rects2)
        
        # Customize plot
        ax.set_ylabel('Volume (m³)')
        ax.set_title(f'Cumulative Landslide Volumes by Watershed - Scenario {scen_num}')
        ax.set_xticks(x)
        ax.set_xticklabels(watersheds, rotation=45, ha='right')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_landslide_time_series(self, figsize=(12, 6), save_path=None):
        """
        Plot time series of cumulative landslide volumes for each watershed.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Initialize dictionaries to store time series data
        times = []
        watershed_data = {}
        
        # First collect all data
        for timestep in range(len(self.all_timestep_metrics)):
            stats = self.analyze_landslides_by_watershed(timestep)
            times.append(self.all_timestep_metrics[timestep]['elapsed_time'])
            
            for watershed in stats.keys():
                if watershed not in watershed_data:
                    watershed_data[watershed] = {
                        'erosion': [],
                        'deposition': []
                    }
                watershed_data[watershed]['erosion'].append(np.abs(stats[watershed]['cumulative']['erosion_volume']))
                watershed_data[watershed]['deposition'].append(stats[watershed]['cumulative']['deposition_volume'])
        
        # Convert times to numpy array
        times = np.array(times)
        
        # Plot data for each watershed
        for watershed, data in watershed_data.items():
            color = self.watershed_colors.get(watershed, 'gray')
            erosion = np.array(data['erosion'])
            deposition = np.array(data['deposition'])
            
            # Plot both erosion and deposition
            ax.plot(times, erosion, '--', color=color, alpha=0.5, label=f'{watershed} (erosion)')
            ax.plot(times, deposition, '-', color=color, label=f'{watershed} (deposition)')
        
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Cumulative Volume (m³)')
        ax.set_title('Landslide Volume Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_landslide_density_map(self, scen_num, timestep=-1, figsize=(10, 8), save_path=None):
        """
        Create map showing landslide density across watersheds.
        """
        grid = self.get_grid_at_timestep(timestep)
        stats = self.analyze_landslides_by_watershed(timestep)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create combined density map
        density_grid = np.zeros(grid.shape)
        
        for watershed, data in stats.items():
            watershed_data = self.all_timestep_metrics[timestep]['metrics'][watershed]
            mask = watershed_data['mask'].reshape(grid.shape)
            
            # Calculate density for this watershed
            erosion = grid.at_node['landslide__erosion'].reshape(grid.shape)
            deposition = grid.at_node['landslide__deposition'].reshape(grid.shape)
            
            # Combine erosion and deposition
            activity = ((erosion > 0) | (deposition > 0)).astype(float)
            density_grid[mask.reshape(grid.shape)] = activity[mask.reshape(grid.shape)]
        
        # Plot density map
        im = ax.imshow(density_grid, cmap='YlOrRd')
        plt.colorbar(im, ax=ax, label='Landslide Activity')
        
        # Add watershed boundaries
        for watershed, data in stats.items():
            watershed_data = self.all_timestep_metrics[timestep]['metrics'][watershed]
            mask = watershed_data['mask'].reshape(grid.shape)
            
            # Create watershed boundary
            boundary = np.zeros_like(mask)
            boundary[1:-1, 1:-1] = mask[1:-1, 1:-1] & (
                ~mask[:-2, 1:-1] | ~mask[2:, 1:-1] |
                ~mask[1:-1, :-2] | ~mask[1:-1, 2:]
            )
            
            # Plot boundary
            color = self.watershed_colors.get(watershed, 'black')
            ax.contour(boundary, colors=[color], levels=[0.5], linewidths=2)
        
        ax.set_title(f'Landslide Activity Density - Scenario {scen_num}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_landslide_size_distribution(self, scen_num, timestep=-1, figsize=(10, 6), save_path=None):
        """
        Plot probability density functions of landslide sizes for each watershed.
        """
        from scipy import stats
        
        fig, ax = plt.subplots(figsize=figsize)
        grid = self.get_grid_at_timestep(timestep)
        watershed_metrics = self.all_timestep_metrics[timestep]['metrics']
        
        for watershed, metrics in watershed_metrics.items():
            mask = metrics['mask']
            
            # Get landslide sizes (combine erosion and deposition)
            erosion = grid.at_node['landslide__erosion'][mask]
            deposition = grid.at_node['landslide__deposition'][mask]
            
            # Combine and filter non-zero values
            sizes = np.concatenate([erosion[erosion > 0], deposition[deposition > 0]])
            
            if len(sizes) > 0:
                # Calculate PDF
                kernel = stats.gaussian_kde(np.log10(sizes))
                x_range = np.linspace(np.log10(sizes.min()), np.log10(sizes.max()), 100)
                pdf = kernel(x_range)
                
                # Plot
                color = self.watershed_colors.get(watershed, 'gray')
                ax.plot(10**x_range, pdf, label=watershed, color=color)
        
        ax.set_xscale('log')
        ax.set_xlabel('Landslide Size (m³)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Landslide Size Distribution - Scenario {scen_num}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_landslide_summary(self, timestep=-1, figsize=(20, 15), save_path=None):
        """
        Create a comprehensive figure with all landslide visualizations.
        """
        fig = plt.figure(figsize=figsize)
        
        # Volume comparison
        ax1 = plt.subplot(2, 2, 1)
        self.plot_landslide_volume_comparison(timestep=timestep, ax=ax1)
        
        # Time series
        ax2 = plt.subplot(2, 2, 2)
        self.plot_landslide_time_series(ax=ax2)
        
        # Density map
        ax3 = plt.subplot(2, 2, 3)
        self.plot_landslide_density_map(timestep=timestep, ax=ax3)
        
        # Size distribution
        ax4 = plt.subplot(2, 2, 4)
        self.plot_landslide_size_distribution(timestep=timestep, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_volume_vs_metric(self, metric_name, timestep=-1, volume_type='total', 
                             figsize=(10, 6), save_path=None):
        """
        Create scatter plot of landslide volume vs specified metric for each watershed.
        
        Parameters
        ----------
        metric_name : str
            Name of metric to plot on y-axis (e.g., 'slope', 'drainage_area', 'tri', etc.)
        timestep : int, optional
            Timestep to analyze (default: -1, last timestep)
        volume_type : str, optional
            Type of volume to use ('total', 'erosion', or 'deposition')
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        fig, ax : matplotlib figure and axes objects
        dict : Dictionary containing the plotted data for further analysis
        """
        grid = self.get_grid_at_timestep(timestep)
        print("Available fields:", list(grid.at_node.keys())) 
        
        # Get landslide statistics and watershed metrics
        ls_stats = self.analyze_landslides_by_watershed(timestep)
        timestep_data = self.get_timestep_metrics(timestep)
        
        # Initialize data containers
        volumes = []
        metrics = []
        watershed_names = []
        
        # Collect data for each watershed
        for watershed_name, watershed_data in timestep_data['metrics'].items():
            # Get landslide volumes
            ls_data = ls_stats[watershed_name]['current']
            if volume_type == 'total':
                volume = ls_data['erosion_volume'] + ls_data['deposition_volume']
            elif volume_type == 'erosion':
                volume = ls_data['erosion_volume']
            elif volume_type == 'deposition':
                volume = ls_data['deposition_volume']
            else:
                raise ValueError("volume_type must be 'total', 'erosion', or 'deposition'")
            
            # Get metric value (mean for the watershed)
            metric_values = watershed_data[metric_name]
            metric_mean = np.mean(metric_values)
            
            volumes.append(volume)
            metrics.append(metric_mean)
            watershed_names.append(watershed_name)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot
        colors = [self.watershed_colors.get(w, 'gray') for w in watershed_names]
        scatter = ax.scatter(volumes, metrics, c=colors, s=100, alpha=0.7)
        
        # Add labels for each point
        for i, txt in enumerate(watershed_names):
            ax.annotate(txt, (volumes[i], metrics[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        # Add labels and title
        ax.set_xlabel(f'Landslide {volume_type.capitalize()} Volume (m³)')
        ax.set_ylabel(f'Mean {metric_name}')
        ax.set_title(f'Landslide Volume vs {metric_name} by Watershed\nScenario {self.scenario_num}')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
    # Optional log scale if data spans multiple orders of magnitude
        if volume_type == 'both':
            all_volumes = erosion_volumes + deposition_volumes
        elif volume_type == 'erosion':
            all_volumes = erosion_volumes
        else:
            all_volumes = deposition_volumes
            
        if len(all_volumes) > 0:  # Add check for empty sequences
            if max(all_volumes)/min(all_volumes) > 100:
                ax.set_xscale('log')
            if max(metrics)/min(metrics) > 100:
                ax.set_yscale('log')
        else:
            print("Warning: No valid landslide volumes found for plotting")
            ax.text(0.5, 0.5, "No landslide activity detected", 
                    ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Return figure, axes, and data for further use/comparison
        return fig, ax, {
            'watershed_names': watershed_names,
            'volumes': volumes,
            'metrics': metrics,
            'scenario': self.scenario_num,
            'timestep': timestep,
            'metric_name': metric_name,
            'volume_type': volume_type
        }


        
    def plot_scenarios_volume_vs_metric(self, metric_name, scenarios_data, watershed='watershed_1', timestep=-1,
                                      volume_type='both', figsize=(10, 6), save_path=None):
        """
        Create scatter plot comparing landslide volumes vs specified metric across different scenarios.
        
        
        Parameters
        ----------
        metric_name : str
            Name of the metric to analyze (e.g., 'tri', 'slope', 'ISED')
        scenarios_data : dict
            Dictionary containing data for multiple scenarios
        watershed : str, optional
            Watershed to analyze for PDFs (default: 'watershed_1')
        timestep : int, optional
            Timestep to analyze (default: -1, last timestep)
        volume_type : str, optional
            Type of volumes to plot. Options:
            - 'both': Plot both erosion and deposition (default)
            - 'erosion': Plot only erosion volumes
            - 'deposition': Plot only deposition volumes
        figsize : tuple, optional
            Figure size in inches (width, height)
        save_path : str, optional
            Path to save the figure
    
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        dict
            Dictionary containing the plotted data
        """
        
        # Initialize data containers
        erosion_volumes = []
        deposition_volumes = []
        metrics = []
        scenario_names = []
        
        # Define colors and styles based on scenario digits
        cmap_SA2 = plt.cm.get_cmap('viridis')
        digit_colors = {
            '1': cmap_SA2(0.2),  # All x1 scenarios
            '2': cmap_SA2(0.5),  # All x2 scenarios
            '3': cmap_SA2(0.8)   # All x3 scenarios
        }
        series_styles = {
            '1': '-',    # Series 1: solid
            '2': '--',   # Series 2: dashed
            '3': ':'     # Series 3: dotted
        }
        
        # Sort scenarios to ensure scenario 10 is plotted first
        sorted_scenarios = sorted(scenarios_data.keys(), 
                                key=lambda x: (x != 'scenario_10', x))
    
        # Process each scenario
        for scenario_name in sorted_scenarios:
            scenario_data = scenarios_data[scenario_name]
            timestep_data = scenario_data[timestep]
            
            # Get grid for this scenario at this timestep
            grid = self.get_grid_at_timestep(timestep)
            
            # Get metric data for the specified watershed
            watershed_data = timestep_data['metrics'][watershed]
            metric_values = watershed_data[metric_name]
            mask = watershed_data['mask']
            
            # Calculate metric mean (excluding invalid values)
            valid_values = metric_values[np.isfinite(metric_values)]
            if len(valid_values) > 0:
                metric_mean = np.mean(valid_values)
                
                # Get landslide volumes from grid fields
                erosion = grid.at_node['landslide__erosion'][mask]
                deposition = grid.at_node['landslide__deposition'][mask]
                
                # Calculate total volumes
                erosion_volume = np.sum(erosion) * grid.dx * grid.dx
                deposition_volume = np.sum(deposition) * grid.dx * grid.dx
                
                print(f"Scenario {scenario_name}, Watershed {watershed}:")
                print(f"  Erosion volume: {erosion_volume:.2e}")
                print(f"  Deposition volume: {deposition_volume:.2e}")
                
                # Only append data if there's landslide activity
                if erosion_volume > 0 or deposition_volume > 0:
                    erosion_volumes.append(erosion_volume)
                    deposition_volumes.append(deposition_volume)
                    metrics.append(metric_mean)
                    scenario_names.append(scenario_name)
                else:
                    print("  No landslide activity detected")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot based on volume_type
        if volume_type in ['both', 'erosion'] and erosion_volumes:
            for i, name in enumerate(scenario_names):
                # Get color and style based on scenario name
                if name == 'scenario_10':
                    color = 'black'
                    style = '-'
                else:
                    first_digit = name[-2]   # Series number
                    second_digit = name[-1]  # Scenario within series
                    color = digit_colors[second_digit]
                    style = series_styles[first_digit]
                
                marker = 'o'  # Circle for erosion
                ax.scatter(erosion_volumes[i], metrics[i], color=color, marker=marker,
                          s=100, alpha=0.7, label=f'{name} (erosion)')
        
        if volume_type in ['both', 'deposition'] and deposition_volumes:
            for i, name in enumerate(scenario_names):
                # Get color and style (same as above)
                if name == 'scenario_10':
                    color = 'black'
                    style = '-'
                else:
                    first_digit = name[-2]
                    second_digit = name[-1]
                    color = digit_colors[second_digit]
                    style = series_styles[first_digit]
                
                marker = '^'  # Triangle for deposition
                ax.scatter(deposition_volumes[i], metrics[i], color=color, marker=marker,
                          s=100, alpha=0.7, label=f'{name} (deposition)')
        
        # Check if we have any data to plot
        if not erosion_volumes and not deposition_volumes:
            print("Warning: No valid landslide volumes found for plotting")
            ax.text(0.5, 0.5, "No landslide activity detected", 
                    ha='center', va='center', transform=ax.transAxes)
        else:
            # Add labels and title
            ax.set_xlabel('Landslide Volume (m³)')
            ax.set_ylabel(f'Mean {metric_name}')
            ax.set_title(f'Landslide Volumes vs {metric_name}\n{watershed} Across Scenarios')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Set scales
            all_volumes = []
            if volume_type in ['both', 'erosion']:
                all_volumes.extend(erosion_volumes)
            if volume_type in ['both', 'deposition']:
                all_volumes.extend(deposition_volumes)
                
            if len(all_volumes) > 0:
                if max(all_volumes)/min(all_volumes) > 100:
                    ax.set_xscale('log')
                if max(metrics)/min(metrics) > 100:
                    ax.set_yscale('log')
            
            # Customize legend
            handles, labels = ax.get_legend_handles_labels()
            
            # Sort to ensure scenario 10 is first
            sorted_indices = []
            # First add scenario 10 indices if they exist
            sorted_indices.extend([i for i, l in enumerate(labels) if 'scenario_10' in l])
            # Then add other scenarios in numerical order
            sorted_indices.extend([i for i, l in enumerate(labels) if 'scenario_10' not in l])
            
            handles = [handles[i] for i in sorted_indices]
            labels = [labels[i] for i in sorted_indices]
            
            ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig, ax, {
            'scenario_names': scenario_names,
            'erosion_volumes': erosion_volumes,
            'deposition_volumes': deposition_volumes,
            'metrics': metrics,
            'watershed': watershed,
            'timestep': timestep,
            'metric_name': metric_name,
            'volume_type': volume_type
        }


    def plot_scenarios_volume_vs_metric_old_v3(self, metric_name, scenarios_data, watershed='watershed_1', timestep=-1,
                                      base_path='/Users/csdmsuser/Documents/Research/CU/GBE/Magnitude_Frequency_Experiments',
                                      figsize=(10, 6), save_path=None):
        # Initialize data containers
        volumes = []
        metrics = []
        scenario_names = []
        
        for scenario_name, scenario_data in scenarios_data.items():
            scen_num = scenario_name.split('_')[1]
            
            # Load and process grid states
            scenario_path = os.path.join(base_path, f'scenario{scen_num}', f'scenario_{scen_num}_output/grid_states.pkl')
            with open(scenario_path, 'rb') as f:
                grid_states = pickle.load(f)
                
            # Get state data
            state = grid_states[timestep]
            
            # Get metric data
            watershed_data = scenario_data[timestep]['metrics'][watershed]
            metric_values = watershed_data[metric_name]
            metric_mean = np.nanmean(metric_values[np.isfinite(metric_values)])
            
            # Get landslide data if it exists
            if 'landslide__erosion' in state:
                erosion = state['landslide__erosion']
                mask = watershed_data['mask']
                erosion_volume = np.sum(erosion[mask]) * 30 * 30  # Using grid spacing of 30m
            else:
                erosion_volume = 0
                
            if erosion_volume > 0 and np.isfinite(metric_mean):
                volumes.append(erosion_volume)
                metrics.append(metric_mean)
                scenario_names.append(scenario_name)
        
        # Only plot if we have valid data
        if volumes and metrics:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create scatter plot
            colors = plt.cm.tab10(np.linspace(0, 1, len(scenario_names)))
            scatter = ax.scatter(volumes, metrics, c=colors, s=100, alpha=0.7)
            
            # Add labels for each point
            for i, txt in enumerate(scenario_names):
                scen_num = txt.split('_')[1]
                ax.annotate(f'Scenario {scen_num}', (volumes[i], metrics[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel('Landslide Erosion Volume (m³)')
            ax.set_ylabel(f'Mean {metric_name}')
            ax.set_title(f'Landslide Erosion Volume vs {metric_name}\n{watershed} Across Scenarios')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig, ax, {
                'scenario_names': scenario_names,
                'volumes': volumes,
                'metrics': metrics
            }
        else:
            print("No valid data found for plotting")
            return None, None, None


    def plot_scenarios_volume_vs_metric_old_v2(self, metric_name, scenarios_data, watershed='watershed_1', timestep=-1,
                                      figsize=(10, 6), save_path=None):
        """
        Create scatter plot comparing landslide erosion volume vs specified metric across different scenarios 
        for a specific watershed.
        """
        # Initialize data containers
        volumes = []
        metrics = []
        scenario_names = []
        
        # Process each scenario
        for scenario_name, scenario_data in scenarios_data.items():
            # Get the grid for this scenario at the specified timestep
            grid = self.get_grid_at_timestep(timestep)
            
            # Calculate landslide statistics for this scenario
            ls_stats = self.analyze_landslides_by_watershed(timestep)
            
            # Get metric data for the specified watershed
            timestep_data = scenario_data[timestep]
            watershed_data = timestep_data['metrics'][watershed]
            
            # Get erosion volume only
            erosion_volume = ls_stats[watershed]['current']['erosion_volume']
            
            # Get metric value (mean for the watershed)
            metric_values = watershed_data[metric_name]
            valid_values = metric_values[np.isfinite(metric_values)]
            if len(valid_values) > 0:
                metric_mean = np.mean(valid_values)
                
                volumes.append(erosion_volume)
                metrics.append(metric_mean)
                scenario_names.append(scenario_name)
        
        # Create figure and plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot
        colors = plt.cm.tab10(np.linspace(0, 1, len(scenario_names)))
        scatter = ax.scatter(volumes, metrics, c=colors, s=100, alpha=0.7)
        
        # Add labels for each point
        for i, txt in enumerate(scenario_names):
            scen_num = txt.split('_')[-1] if '_' in txt else txt
            ax.annotate(f'Scenario {scen_num}', (volumes[i], metrics[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        # Add labels and title
        ax.set_xlabel('Landslide Erosion Volume (m³)')
        ax.set_ylabel(f'Mean {metric_name}')
        ax.set_title(f'Landslide Erosion Volume vs {metric_name}\n{watershed} Across Scenarios')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Set axis limits with padding
        x_min, x_max = min(volumes), max(volumes)
        y_min, y_max = min(metrics), max(metrics)
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax, {
            'scenario_names': scenario_names,
            'volumes': volumes,
            'metrics': metrics,
            'watershed': watershed,
            'timestep': timestep,
            'metric_name': metric_name
        }

    # First, let's add the proper function to the WatershedMetricsCalculator class
    def plot_scenarios_volume_vs_metric_old(self, metric_name, scenarios_data, watershed='watershed_1', timestep=-1,
                                      volume_type='total', figsize=(10, 6), save_path=None):
        """
        Create scatter plot comparing landslide volume vs specified metric across different scenarios 
        for a specific watershed.
        """
        # Initialize data containers
        volumes = []
        metrics = []
        scenario_names = []
        
        # Process each scenario
        for scenario_name, scenario_data in scenarios_data.items():
            # Get the grid for this scenario at the specified timestep
            grid = self.get_grid_at_timestep(timestep)
            
            # Calculate landslide statistics for this scenario
            ls_stats = self.analyze_landslides_by_watershed(timestep)
            
            # Get metric data for the specified watershed
            timestep_data = scenario_data[timestep]
            watershed_data = timestep_data['metrics'][watershed]
            
            # Get landslide volumes
            ls_data = ls_stats[watershed]['current']
            if volume_type == 'total':
                volume = ls_data['erosion_volume'] + ls_data['deposition_volume']
            elif volume_type == 'erosion':
                volume = ls_data['erosion_volume']
            elif volume_type == 'deposition':
                volume = ls_data['deposition_volume']
            else:
                raise ValueError("volume_type must be 'total', 'erosion', or 'deposition'")
            
            # Get metric value (mean for the watershed)
            metric_values = watershed_data[metric_name]
            metric_mean = np.mean(metric_values)
            
            volumes.append(volume)
            metrics.append(metric_mean)
            scenario_names.append(scenario_name)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot using the tab10 colormap for different scenarios
        colors = plt.cm.tab10(np.linspace(0, 1, len(scenario_names)))
        scatter = ax.scatter(volumes, metrics, c=colors, s=100, alpha=0.7)
        
        # Add labels for each point
        for i, txt in enumerate(scenario_names):
            # Extract scenario number for cleaner labeling
            scen_num = txt.split('_')[-1] if '_' in txt else txt
            ax.annotate(f'Scenario {scen_num}', (volumes[i], metrics[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        # Add labels and title
        ax.set_xlabel(f'Landslide {volume_type.capitalize()} Volume (m³)')
        ax.set_ylabel(f'Mean {metric_name}')
        ax.set_title(f'Landslide Volume vs {metric_name}\n{watershed} Across Scenarios')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Optional log scale if data spans multiple orders of magnitude
        if np.max(volumes)/np.min(volumes) > 100:
             ax.set_xscale('log')
        if np.max(metrics)/np.min(metrics) > 100:
             ax.set_yscale('log')
        
        #ax.set_xlim(46620, 46640)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax, {
            'scenario_names': scenario_names,
            'volumes': volumes,
            'metrics': metrics,
            'watershed': watershed,
            'timestep': timestep,
            'metric_name': metric_name,
            'volume_type': volume_type
        }

    @staticmethod
    def _print_progress(iteration, total, prefix='Processing:', length=50):
        """Print progress bar while suppressing other output."""
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = '█' * filled_length + '-' * (length - filled_length)
        # Use \r to overwrite the line and end='' to prevent newline
        print(f'\r{prefix} |{bar}| {percent}% Complete', end='')
        # Only print newline when complete
        if iteration == total:
            print()
    
# =============================================================================
#     @staticmethod
#     def _print_progress_old(iteration, total, prefix='Processing:', length=50):
#         """Print progress bar."""
#         percent = ("{0:.1f}").format(100 * (iteration / float(total)))
#         filled_length = int(length * iteration // total)
#         bar = '█' * filled_length + '-' * (length - filled_length)
#         print(f'\r{prefix} |{bar}| {percent}% Complete', end='\r')
#         if iteration == total:
#             print()
# =============================================================================
                
