from landlab_ensemble import generate_ensembles as ge
from landlab_ensemble import construct_model as cm
import argparse
import os
import importlib

def create(args):
    """Parse commandline arguments and run a model parameter database creation.
    x
    Arguments:
    template -- the path for the json file  to create the databse from
    output -- the path for the output database file
    """
    input_template = args.template
    output_db = args.output
    if not os.path.exists(input_template):
        raise argparse.ArgumentTypeError(f"The provided template file, '{input_template}' could not be found.")
    if os.path.exists(output_db):
        raise argparse.ArgumentTypeError(f"The provided output database file, '{output_db}' already exists.")
    ge.create_model_db(output_db, input_template)

def dispatch(args):
    """Create and run models from a parameter database.

    Arguments:
    database -- the path of the database to pull parameters from
    od -- a directory/other prefix for output model runs to be saved in
    filter -- some sort of sqlite condition for pulling runs from the database
    n -- number of maximum runs
    processes -- the number of processes (dask workers) to create for running models
    clean -- a boolean flag to remove unfinished runs from tables so they can be rerun
    """
    if not os.path.exists(args.database):
        raise argparse.ArgumentTypeError(f"The provided database file, `{args.database}` could not be found.")
    module, model = args.model.rsplit('.',1)
    model = getattr(importlib.import_module(module), model)
    if args.one:
        cm.run_model(args.database, model, args.batch_id, args.model_id, args.od)
        return
    dispatcher = cm.ModelDispatcher(args.database, model, args.od, args.filter, args.n, args.processes)
    if args.clean:
        dispatcher.clean_unfinished_runs()
    dispatcher.run_all()

def slurm_config(args):
    if not os.path.exists(args.database):
        raise argparse.ArgumentTypeError(f"The provided database file, `{args.database}` could not be found.")
    cm.generate_config_file_for_slurm(args.database, args.model, args.od,
                                      args.n, args.filter, args.slurm_csv,
                                      args.checkout_models)
    cm.generate_sbatch_file("Landlab Batch", args.num_tasks, args.cpus, args.n, args.slurm_csv, args.sbatch_file)
