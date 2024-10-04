# A first pass for parameter space exploration of GBE + Hylands

# Ones I'm most interested in 
uplift_rates_range = [1e-4, 1e-3]
internal_angle_of_friction_range = [0.36, 0.85]
ls_return_time_range = [10, 2000]
timestep_range = [10, 500]
total_time = 2e4

# Secondary metrics; these numbers feel more random
plucking_coeff_range = [3e-5, 3e-3]     # 3e-4 seems to be realistic
sed_gbe_coeff_range = [0.031, 0.051]    # 0.041 seems to be realistic
attrition_coeff_range = [0.001, 0.01]   # 0.005 seems to be realistic
br_abrasion_coeff_range = [0.001, 0.01] # 0.005 seems to be realistic
phi_range = [0.05, 0.15]                # 0.1 seems to be realistic
cohesion_range = [1e4, 1e5]             # 1e4 seems to be realistic  