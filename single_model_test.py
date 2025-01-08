from landlab_ensemble import construct_model as cm
from landlab_ensemble import generate_ensembles as ge
from PeriodicLandslider import PeriodicLandslider
import os
TEST_DB_PATH = "test.db"
if os.path.exists(TEST_DB_PATH):
    os.remove(TEST_DB_PATH)
ge.create_model_db(TEST_DB_PATH, "SLURM_test_params.json")
cm.run_model(TEST_DB_PATH, PeriodicLandslider, 1, 1, "test_output/")
