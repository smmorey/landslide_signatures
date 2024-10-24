#! /usr/bin/env python
# coding: utf-8

# # Base class for a grid-based Landlab model
#
# *(Greg Tucker, University of Colorado Boulder)*
#

import sys

import numpy as np
from landlab import ModelGrid, create_grid, load_params
from landlab.io.native_landlab import load_grid, save_grid


def merge_user_and_default_params(user_params, default_params):
    """Merge default parameters into the user-parameter dictionary, adding
    defaults where user values are absent.

    Examples
    --------
    >>> u = {"a": 1, "d": {"da": 4}, "e": 5, "grid": {"RasterModelGrid": []}}
    >>> d = {"a": 2, "b": 3, "d": {"db": 6}, "grid": {"HexModelGrid": []}}
    >>> merge_user_and_default_params(u, d)
    >>> u["a"]
    1
    >>> u["b"]
    3
    >>> u["d"]
    {'da': 4, 'db': 6}
    >>> u["grid"]
    {'RasterModelGrid': []}
    """
    for k in default_params.keys():
        if k in default_params:
            if k not in user_params.keys():
                user_params[k] = default_params[k]
            elif isinstance(user_params[k], dict) and k != "grid":
                merge_user_and_default_params(user_params[k], default_params[k])


def get_or_create_node_field(grid, name, dtype="float64"):
    """Get handle to a grid field if it exists, otherwise create it."""
    try:
        return grid.at_node[name]
    except KeyError:
        return grid.add_zeros(name, at="node", dtype=dtype, clobber=True)


def _get_pause_time_list_and_next(time_info, clock_dict):
    """
    Given a float or iterable as ``time_info``, return a list of times to pause the
    simulation to perform an action.

    Return a list of times at which to pause the simulation, including an item at
    the end that is after the termination of the run so that there is always an item
    to be popped.

    Parameters
    ----------
    time_info : float or list of float
        Interval for pausing (if float) or list of individual times
    clock_dict : dict
        Contains ``start`` and ``stop`` as float items, with ``stop`` > ``start``

    Returns
    -------
    list : list of simulation times at which to pause for a given action
    float : the next time at which to pause
    """
    if isinstance(time_info, float):
        pause_times = list(
            np.arange(
                clock_dict["start"], clock_dict["stop"] + 2 * time_info, time_info
            )
        )
    elif isinstance(time_info, list):
        pause_times = time_info.copy()
        pause_times.append(clock_dict["stop"] + 1.0)
    next_pause = pause_times.pop(0)
    return pause_times, next_pause

def max_steady_state(old_values, new_values):
    return np.abs(np.max(old_values) - np.max(new_values))

def mean_steady_state(old_values, new_values):
    return np.abs(np.mean(old_values) - np.mean(new_values))

def max_local_steady_state(old_values, new_values):
    return np.max(np.abs(old_values-new_values))

STEADY_STATE_FUNCTIONS = {'max': max_steady_state,
                          'mean': mean_steady_state,
                          'max_local': max_local_steady_state}

def out_of_time(run_duration, start_time, current_time):
    if run_duration is None:
        return False
    else:
        stop_time = run_duration + start_time
        return current_time >= stop_time

def get_out_of_time_function(run_duration, start_time):
    return lambda current_time: out_of_time(run_duration, start_time, current_time)


class LandlabModel:
    """Base class for a generic Landlab grid-based model."""

    DEFAULT_PARAMS = {
        "grid": {
            "source": "create",
            "create_grid": {
                "RasterModelGrid": [
                    {"shape": (5, 5)},
                    {"spacing": 1.0},
                ],
            },
        },
        "clock": {"start": 0.0, "stop": 2.0, "step": 1.0},
        "output": {
            "plot_times": 10.0,  # float or list
            "save_times": 10.0,  # float or list
            "report_times": 1.0,  # float or list
            "save_path": "model_output",
            "clobber": True,
            "fields": None,
            "plot_to_file": True,
        },
    }

    def __init__(self, params={}):
        """Initialize the model."""
        merge_user_and_default_params(params, self.DEFAULT_PARAMS)
        self.setup_grid(params["grid"])
        self.setup_for_output(params)
        self.setup_run_control(params["runtime"])

    def setup_grid(self, grid_params):
        """Load or create the grid.

        Examples
        --------
        >>> p = {"grid": {"source": "create"}}
        >>> p["grid"]["create_grid"] = {
        ...     "RasterModelGrid": {
        ...         "shape": (4, 5),
        ...         "xy_spacing": 2.0
        ...     }
        ... }
        >>> sim = LandlabModel(params=p)
        >>> sim.grid.shape
        (4, 5)
        >>> from landlab.io.native_landlab import save_grid
        >>> save_grid(sim.grid, "test.grid", clobber=True)
        >>> p = {"grid": {"source": "file", "grid_file_name": "test.grid"}}
        >>> sim = LandlabModel(params=p)
        >>> sim.grid.shape
        (4, 5)
        >>> from landlab import RasterModelGrid
        >>> p = {"grid": {"source": "grid_object"}}
        >>> p["grid"]["grid_object"] = RasterModelGrid((3, 3))
        >>> sim = LandlabModel(params=p)
        >>> sim.grid.shape
        (3, 3)
        >>> from numpy.testing import assert_raises
        >>> p["grid"]["grid_object"] = "spam"
        >>> assert_raises(ValueError, LandlabModel, p)
        grid_object must be a Landlab grid.
        """
        if grid_params["source"] == "create":
            self.grid = create_grid(grid_params, section="create_grid")
        elif grid_params["source"] == "file":
            self.grid = load_grid(grid_params["grid_file_name"])
        elif grid_params["source"] == "grid_object":
            if isinstance(grid_params["grid_object"], ModelGrid):
                self.grid = grid_params["grid_object"]
            else:
                print("grid_object must be a Landlab grid.")
                raise ValueError

    def setup_for_output(self, params):
        """
        Setup variables for control of plotting and saving.

        Parameters
        ----------
        params : dict
            Parameter dictionary. Must include a key ``output`` with a dictionary
        containing values for ``plot_times``, ``save_times``, and ``report_times``.
        Each of these should be either a ``float`` or a ``list``. If a list, the value
        is interpreted as a list of model times for plotting, saving, or reporting.
        If a single float, the value is interpreted as the (regular) time
        interval (in model time) for plotting, saving, or reporting.
            Should also contain a key ``clock`` as a dictionary that has values
        for ``start`` and ``stop``.
        """
        op_params = params["output"]
        clock_params = params["clock"]

        self.plot_times, self.next_plot = _get_pause_time_list_and_next(
            op_params["plot_times"], clock_params
        )
        self.save_times, self.next_save = _get_pause_time_list_and_next(
            op_params["save_times"], clock_params
        )
        self.report_times, self.next_report = _get_pause_time_list_and_next(
            op_params["report_times"], clock_params
        )

        self.ndigits_for_save_files = int(np.ceil(np.log10(len(self.save_times) + 1)))
        self.save_num = 0  # current save file frame number
        self.save_path = op_params["save_path"]
        if op_params["plot_to_file"]:
            self.ndigits_for_plot_files = int(
                np.ceil(np.log10(len(self.plot_times) + 1))
            )
            self.plot_num = 0  # current plot image frame number
        self.display_params = params

    def setup_run_control(self, runtime_params):
        """Initialize variables related to control of run timing."""
        self.dt = 1
        try:
            clock_params = runtime_params['clock']
            self.dt = clock_params["step"]
            self.run_duration = clock_params["stop"] - clock_params["start"]
            self.current_time = clock_params["start"]
        except KeyError:
            self.run_duration = None
            self.current_time = 0
        try:
            steady_state_params = runtime_params['steady_state']
            self.steady_state = steady_state_params['steady_state']
            self.steady_state_type = steady_state_params['steady_state_type']
            self.steady_state_threshold = steady_state_params['steady_state_threshold']
            self.steady_state_interval = steady_state_params['steady_state_interval']
        except KeyError:
            self.steady_state = False
            self.steady_state_interval = self.run_duration
        self.steady_state_ammount = -1
        self.last_elevation = np.zeros(self.grid.number_of_nodes)
        

    def report(self, current_time):
        """Issue a text update on status."""
        print(self.__class__.__name__, "time =", current_time)

    def plot(self, current_time=0.0):
        """Virtual function for plotting; to be overridden."""
        print("Base class placeholder for plot() at time", current_time)

    def save_state(self, save_path, save_num, ndigits):
        """
        Save the grid and its fields.

        Override this function to add to or modify what gets saved.
        """
        save_grid(self.grid, save_path + str(save_num).zfill(ndigits) + ".grid")

    def check_if_steady_state(self):
        steady_state_condition = STEADY_STATE_FUNCTIONS[self.steady_state_type]
        self.steady_state_ammount = steady_state_condition(self.last_elevation, self.grid.at_node['topographic__elevation'])
        self.stady_state = self.steady_state_ammount < self.steady_state_threshold
        self.last_elevation = self.grid.at_node['topographic__elevation']
        
        
    def update(self, dt):
        """Advance the modelb by one time step of duration dt."""
        self.current_time += dt

    def update_until_steady_state(self, dt):
        while not self.steady_state:
            self.update(dt)

    def update_until(self, update_to_time, dt):
        """Iterate up to given time, using time-step duration dt."""
        remaining_time = update_to_time - self.current_time
        while remaining_time > 0.0:
            dt = min(dt, remaining_time)
            self.update(dt)
            remaining_time -= dt

    def run(self, run_duration=None, dt=None):
        """Run the model for given duration, or self.run_duration if none
        given.

        Includes file output of images and model state at user-specified
        intervals.
        """
        if run_duration is None:
            run_duration = self.run_duration
        if dt is None:
            dt = self.dt
        out_of_time = get_out_of_time_function(run_duration, self.current_time)
        while not out_of_time(self.current_time) and not self.steady_state:
            next_pause = self.current_time + self.steady_state_interval
            self.update_until(next_pause, dt)
            self.check_if_steady_state()

    def get_output(self):
        """Produce output report for storage.

        Override this function.
        """
        topo = self.grid.at_node['topographic__elevation']
        topo_max = np.max(topo)
        topo_min = np.min(topo)
        return {'output.model.endtime': self.current_time,
                'output.model.steadystate': self.steady_state,
                'output.model.steadystateammount': self.steady_state_ammount,
                'output.topography.max': topo_max,
                'output.topography.min': topo_min,
                'output.topography.range': topo_max-topo_min,
                'output.topography.mean': np.mean(topo),
                'output.topography.std': np.std(topo)}
        

if __name__ == "__main__":
    """Launch a run.

    Optional command-line argument is the name of a yaml-format text file with
    parameters. File should include sections for "grid_setup", "process",
    "run_control", and "output". Each of these should have the format shown in
    the defaults defined above in the class header.
    """
    if len(sys.argv) > 1:
        params = load_params(sys.argv[1])
        sim = LandlabModel(params)
    else:
        sim = LandlabModel()  # use default params
    sim.run()
