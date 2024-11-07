from landlab.components import (PriorityFloodFlowRouter,
                                ExponentialWeatherer,
                                DepthDependentTaylorDiffuser,
                                SpaceLargeScaleEroder,
                                GravelBedrockEroder,
                                BedrockLandslider
                                )
from model_base import LandlabModel
class PeriodicLandslider(LandlabModel):
    DEFAULT_PARAMS = {
        "grid": {
            "source": "create",
            "create_grid": {
                "RasterModelGrid": [
                    (41, 5),
                    {"xy_spacing": 5},
                    ],
                },
            },
        "clock": {"start": 0.0, "stop": 1000000, "step": 1250},
        "output": {
            "plot_times": [100, 100000, 1000000],
            "save_times": [1000001],
            "report_times": [1000001],
            "save_path": "model_run",
            "fields": None,
            "plot_to_file":  True,
            "save_interval": 10000
            },
        "baselevel": {
            "uplift_rate": 0.0001,
            },
        "flowrouter": {"runoff_rate": 13, "flow_metric": "D8",
                       "supress_out": True, "depression_handler": "fill",
                       "accumulat_flow": True, "seperate_hill_flow": True,
                       "accumulate_flow_hill": True},
        "weatherer": {"soil_production_maximum_rate": 3e-4,
                      "soil_production_decay_depth": 0.44},
        "diffuser": {"soil_transport_decay_depth": 0.1,
                      "slope_crit": 0.58,
                      "nterms": 2,
                      "soil_transport_velocity": 0.01,
                      "dynamic_dt": True,
                      "if_unstable": "raise",
                     "courant_factor": 0.9},
        "eroder": {"type": "abrasion",
                   "abrasion": {"intermittency_factor": 0.01,
                                "sediment_porosity": 0.1,
                                "plucking_rate": 5e-5,
                                "number_of_sediment_classes": 1,
                                "transport_coefficient": 0.041,
                                "abrasion_coefficients": 4e-3,
                                "bedrock_abrasion_coefficient": 4e-3,
                                "coarse_fractions_from_plucking": 0.05},
                   "space": {"K_sed": 1e-5,
                             "K_br": 1.5e-5}},

        "landslider": {"angle_init_frict": 0.58,
                       "threshold_slope": 0.58,
                       "cohesion_eff": 1e4,
                       "landslide_return_time": 100,
                       "landslides_on_boundary_nodes": False,
                       "phi": 0.3,
                       "fraction_fines_LS": 0.5}
        }

    def __init__(self, params={}):
        """Initialize the Model"""
        self.DEFAULT_PARAMS = DEFAULT_PARAMS
        super().__init__(params)
#ask susannah about model grid starts
        # existing equilibrium landscape
        if not ("topographic__elevation" in self.grid.at_node.keys()):
            self.grid.add_zeros("topographic__elevation", at="node")
        rng = np.random.default_rng(seed=int(params["seed"]))
        grid_noise= rng.random(self.grid.number_of_nodes)/10
        self.grid.at_node["topographic__elevation"] += grid_noise
        self.topo = self.grid.at_node["topographic__elevation"]

        self.uplift_rate = params["baselevel"]["uplift_rate"]

        self.flow_router = PriorityFloodFlowRouter(
            self.grid, surface = "topographic__elevation", **params["flowrouter"])
        self.flow_router.run_one_step()
        self.weatherer = ExponentialWeatherer(self.grid, **params["weatherer"])
        self.diffuser = DepthDependentTaylorDiffuser(self.grid, **params["diffuser"])
        eroder_type = params["eroder"]["type"]
        if eroder_type == "abrasion":
            self.eroder = GravelBedrockEroder(self.grid, **params["eroder"][eroder_type])
        elif eroder_type == "space":
            self.eroder = SpaceLargeScaleEroder(self.grid, **params["eroder"][eroder_type])
        self.landslider = BedrockLandslider(self.grid **params["landslider"])
        self.landslide_recur_int = params["landslider"]["landslide_return_time"] # what does this param do in the landslider?
        # this is about stoachasticy
        self.save_interval = params[output]["save_interval"]
        
    def update(self, dt):
        uplift = self.uplift_rate * dt # figure out whats up with time step during ls years
        # the smallest stable timestep for landlab is 1, this is supposed to be as close to an
        # instantaneous event as possible
        # we want to record if things are landslide years
        self.grid.at_node["bedrock__elevation"][self.grid.core_nodes] += uplift
        self.grid.at_node["topographic__elevation"][:] = (self.gird.at_node["bedrock__elevation"] + self.grid.at_node["soil__depth"])
        self.weatherer.run_one_step()
        self.diffuser.run_one_step(dt)
        self.flow_router.run_one_step()
        self.eroder.run_one_step(dt)
        # do we want recurrance interval to be set or stocastic?
        # we can make that stochastic
        
        if (self.current_time % self.landslide_recur_int == 0):
            self.landslider.run_one_step(dt)
        if (self.current_time % self.save_interval == 0):
            self.grid.add_field(f"t{self.current_time}_topographic__elevation",
                                self.grid.at_node["topographic__elevation"], at="node", copy=True)
            
