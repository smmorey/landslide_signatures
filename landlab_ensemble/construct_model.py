import importlib
import builtins
import json
import sqlite3
import uuid
import time
import os
import csv

def _resolve_type(type_str):
    """This function returns the python class for a type string.
    
    Adapted from chatgpt.
    """
    if type_str == "<class 'NoneType'>":
        return type(None)
    module_class = type_str.split("'")[1].rsplit('.', 1)
    if len(module_class) == 1:
        module = builtins
        class_name = module_class[0]
    else:
        module_name, class_name = module_class
        module = importlib.import_module(module_name)
    return getattr(module, class_name)

def _ensure_type(value, ideal_type):
    """Given a string and a type, make sure that string is cast correctly.

    Slightly more complicated for two reasons.  First None needs to be
    handled slightly differently, and two, strings representing lists and
    dictionaries are better handled with json rather than naiive typecasting.
    """
    if ideal_type==type(None):
        return None
    elif ideal_type in (type([]), type({})):
        return json.loads(value.replace("'", "\""))
    else:
        return ideal_type(value)

def _expand_key_into_dict(key, value, current_dict):
    """Given a flattened key and associated value, unflatten the key, adding necessary dictionaries into the given dictionary.

    Given a dictionary {} and a key like "my_dict1.my_dict2.key" and a value: "value",  create the following structure:
    {"my_dict1": {my_dict2: {"key": value}}}.
    Works recursively.  Not agnostic as it expects heirarchy to be '.' seperated.
    """
    split_key = key.split('.',1)
    if len(split_key)==1:
        current_dict[key] = value
        return current_dict
    else:
        try:
            next_dict = current_dict[split_key[0]]
        except KeyError:
            next_dict = {}
            current_dict[split_key[0]] = next_dict
        return _expand_key_into_dict(split_key[1], value, next_dict)
    
def expand_dict(flat_dict):
    """Given a flat dictionary with keys representing heirarchy, restore a corresponding nested dictionary."""
    expanded_dict = {}
    for key, value in flat_dict.items():
        _expand_key_into_dict(key, value, expanded_dict)
    return expanded_dict

def row_to_params(row, columns, types):
    """Given a list of values from a table row, corresponding column names, and ideal types, return an associated dictionary.

    This is not agnostic, as it expect column names of interest to start with "model_param" and heirarchy to be '.'
    seperated.
    Args:
        row -- a list of values
        columns -- a list of names
        types -- a list of ideal types for each value
    """
    row_dict = dict(zip(columns, row))
    parameter_dictionary = {k.split('.', 1)[1]: _ensure_type(v, types[k]) for k,v in row_dict.items() if k.split('.')[0]=="model_param"}
    return expand_dict(parameter_dictionary)

def get_param_types(connection):
    """Pull parameter names and associated python type from the dimension table of a database connection."""
    cursor = connection.cursor()
    param_and_type = cursor.execute("SELECT param_name, python_type FROM model_param_dimension").fetchall()
    cursor.close()
    return {k: _resolve_type(v) for k,v in param_and_type}

class ModelSelector:
    """An object that iterates over runrun parameters in a parameter database.

    The ModelSelector object connects to a specific database, queries for unrun parameter
    combinations, and constructsdictionaries from them.

    Attributes:
        database -- the database path to grab parameters from
        filter_statement -- an additional filter for queries
        select_statement -- the statement that selects from the database
        columns -- the columns of the model parameter database
        limit -- the maximum ammount of parameters to return
        current -- the current number of parameters returend
    """
    def __init__(self, database, filter=None, limit=None):
        """Initializes the ModelSelector with a specific database.

        Args:
            database -- the database path to connect to
            filter -- the sql condition to filter by
            limit -- the maximum number of rows to return
        """
        self.database = database
        if filter:
            self.filter_statement = "%s AND model_run_id IS NULL"
        else:
            self.filter_statement = "model_run_id IS NULL"
        self.connection = sqlite3.connect(database, check_same_thread=False)
        self.parameter_types = get_param_types(self.connection)
        cursor = self.connection.cursor()
        self.select_statement = "SELECT run_param_id, * FROM model_run_params WHERE %s" % (self.filter_statement)#, limit_statement))
        cursor.execute(self.select_statement)
        self.columns = [c[0] for c in cursor.description[1:]]
        cursor.close()
        self.limit = limit
        self.current = 0

    def __iter__(self):
        """Returns the object as it is an iterator."""
        return self

    def __next__(self):
        """Calls the next function, returns the a set of parameters from the database."""
        return self.next()

    def next(self):
        """Returns a set of parameters from the database."""
        if self.limit is not None and self.current > self.limit:
            raise StopIteration
        cursor = self.connection.cursor()
        results = cursor.execute(self.select_statement).fetchone()
        cursor.close()
        if results is None:
            raise StopIteration
        run_id = results[0]
        model_parameters = results[1:]
        param_dict = row_to_params(model_parameters, self.columns, self.parameter_types)
        self.current += 1
        return run_id, param_dict

    def empty(self):
        """Returns True if there are no more parameters for this iterator to return."""
        if self.limit is not None and self.current > self.limit:
            return True
        else:
            cursor = self.connection.cursor()
            results = cursor.execute(self.select_statement).fetchone()
            cursor.close()
            if results is None:
                return True
            else:
                return False

    
def make_and_run_model(model_class, batch_id, model_run_id, param_dict, out_dir):
    """Creates a new instantiation of a landlab model, runs it, and saves the output as a netcdf."""
    model = model_class(param_dict)
    model.batch_id = batch_id
    model.run_id = model_run_id
    model.run()
    end_time = time.time()
    output_f = "%s%s.nc" % (out_dir, model.run_id)
    model.grid.save(output_f, names=model.grid_fields_to_save)
    outputs = model.get_output()
    outputs['model_batch_id'] = model.batch_id
    outputs['model_run_id'] = model.run_id
    outputs['end_time'] = end_time
    return outputs

def run_model(database, model_class, batch_id, run_param_id, output_dir):
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL;")
    cursor = connection.cursor()
    outputs = cursor.execute("SELECT * FROM model_run_outputs")
    valid_output_names = [d[0] for d in outputs.description]
    select_statement = f"SELECT run_param_id, * FROM model_run_params WHERE run_param_id = {run_param_id}"
    results = cursor.execute(select_statement).fetchone()
    columns = [c[0] for c in cursor.description[1:]]
    model_parameters = results[1:]
    parameter_types = get_param_types(connection)
    param_dict = row_to_params(model_parameters, columns, parameter_types)
    model_run_id = str(uuid.uuid4())
    start_time = time.time()
    retries = 5
    for attmept in range(retries):
        try:
            update_statement = "UPDATE model_run_params SET model_batch_id = \"%s\", model_run_id = \"%s\" WHERE run_param_id = %s" % (batch_id, model_run_id, run_param_id)
            cursor.execute(update_statement)
            metadata_insert_statement = "INSERT INTO model_run_metadata (model_run_id, model_batch_id, model_start_time) VALUES (\"%s\", \"%s\", %f)" % (model_run_id, batch_id, start_time)
            cursor.execute(metadata_insert_statement)
            connection.commit()
            outputs = make_and_run_model(model_class, batch_id, model_run_id, param_dict, output_dir)
            metadata_update_statement = "UPDATE model_run_metadata SET model_end_time = %f WHERE model_run_id = \"%s\"" %(outputs['end_time'], outputs['model_run_id'])
            cursor.execute(metadata_update_statement)
            valid_outputs = {key: outputs[key] for key in outputs.keys() if key in valid_output_names}
            columns = str(tuple(valid_outputs.keys()))
            values = str(tuple(valid_outputs.values()))
            query_str = "INSERT INTO model_run_outputs %s VALUES %s;" % (columns, values)
            cursor.execute(query_str)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                time.sleep(2 ** attempt)
            else:
                raise e
        finally:
            connection.commit()
            cursor.close()
    if attempt == retries-1:
        print("Database is locked, could not write to database.")  
            

def generate_config_file_for_slurm(database, model, output_directory, number_of_runs=100, filter=None, filename="slurm_runs.csv", checkout_models=False):
    selector = ModelSelector(database, filter)
    batch_id = str(uuid.uuid4())
    if checkout_models:
        connection = sqlite3.connect(database)
        cursor = connection.cursor()
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, ['database', 'model', 'output_directory', 'batch_id', 'param_id'], delimiter=',')
        #writer.writeheader()
        for _ in range(number_of_runs):
            try:
                id = selector.next()[0]
                if checkout_models:
                    update_statement = f"UPDATE model_run_params SET model_batch_id = \"FOR_SLURM\", model_run_id = \"FOR_SLURM\" WHERE run_param_id = {id}"
                    cursor.execute(update_statement)
                     
                writer.writerow({'database': database,
                                'model': model,
                                'output_directory': output_directory,
                                'batch_id': batch_id,
                                'param_id': id})
            except StopIteration:
                break
    if checkout_models:
        connection.commit()
        cursor.close()

def generate_sbatch_file(job_name, ntasks, cpus, runs, config_path="slurm_runs.csv", slurm_path="landlab_batch_for_slurm.sh"):
    slurm_file_str = f"""
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus}
#SBATCH --array=1-{runs}

config={config_path}

row=$(awk 'FNR == $SLURM_ARRAY_TASK_ID {{print}}' {config_path})
IFS=',' read -r -a vals <<< "$row"

python model_control.py dispatch --one -d ${{vals[0]}} -m ${{vals[1]}} -od ${{vals[2]}} -b ${{vals[3]}} -mid ${{vals[4]}}

                     """
    with open(slurm_path, 'w') as file:
        file.write(slurm_file_str)
    

class ModelDispatcher:
    """An object that creates and runs landlab models from a parameter database.

    Attributes:
        database -- the path to the parameter database
        model_class -- the Landlab model object
        parameter_list -- a ModelSelector for the database
        batch_id -- a unique ID corresponding to this instantiation of the ModelDispatcher
        out_dir -- a directory to save completed model runs to
        filter -- a sqlite statement to filter parameters by
        parameter_types -- a dictionary where the keys are model parameter columns and the values
                           are the python types of the parameter
        self.client -- a daskclient for multiprocessing
        self.processes -- the number of processes to run simultaneously
    """
    def __init__(self, database, model_class, out_dir="", filter=None, limit=None, processes=None):
        """Creates a ModelDispatcher"""
        self.database = database
        self.model_class = model_class
        self.parameter_list = ModelSelector(database, filter, limit)
        self.batch_id = uuid.uuid4()
        self.out_dir = out_dir
        self.filter = filter
        connection = sqlite3.connect(database, check_same_thread=False)
        self.parameter_types = get_param_types(connection)
        if processes is not None:
            try:
                from dask.distributed import Client
                self.client = Client(threads_per_worker=1, n_workers=processes)
            except ImportError:
                print("Dask is required for multiprocessing at this time.  Install Dask in this python environment or use in single process mode.")
                os._exit(os.EX_UNAVAILABLE)
        self.processes = processes
        cursor = connection.cursor()
        outputs = cursor.execute("SELECT * FROM model_run_outputs")
        self.valid_outputs = [d[0] for d in outputs.description]
        cursor.close()

    def run_a_model(self):
        """Grabs the next set of parameters, creates a model, and runs it."""
        try:
            run_id, param_dict = self.parameter_list.next()
            self.dispatch_model(run_id, param_dict)
        except StopIteration:
            self.end_batch()

    def end_batch(self):
        """Little handler for when there are no more parameters"""
        print("no more to run")

    def run_all(self):
        """Runs all the models in the ModelSelector.

        Dispatches to dask client if in multiprocessign mode.
        """
        if self.processes is not None:
            self.run_models_on_dask()
        else:
            for run_id, param_dict in self.parameter_list:
                self.dispatch_model(run_id, param_dict)
        self.end_batch()

    def get_unfinished_runs(self):
        """Finds all the runs in the database that were started but never finished."""
        # should this be moved to the ModelSelector class?
        connection = sqlite3.connect(self.database, check_same_thread=False)
        cursor = connection.cursor()
        selection_statement = "SELECT model_run_id FROM model_run_metadata WHERE model_start_time IS NOT NULL AND model_end_time IS NULL"
        if self.filter:
            selection_statement = "%s AND %s" % (selection_statement, self.filter)
        unfinished = [r[0] for r in cursor.execute(selection_statement).fetchall()]
        cursor.close()
        if len(unfinished)==0:
            return None
        else:
            return unfinished

    def reset_model(self, model_run_id, clear_metadata=True):
        """Takes a given model run, sets it as unrun, and deletes the corresponding  metadata table entry."""
        connection = sqlite3.connect(self.database, check_same_thread=False)
        cursor = connection.cursor()
        update_statement = "UPDATE model_run_params SET model_run_id = NULL, model_batch_id = NULL WHERE model_run_id = \"%s\"" % model_run_id
        cursor.execute(update_statement)
        if clear_metadata:
            delete_statement = "DELETE FROM model_run_metadata WHERE model_run_id = \"%s\"" % model_run_id
            cursor.execute(delete_statement)
        connection.commit()
        cursor.close()

    def clean_unfinished_runs(self, clear_metadata=True):
        """Resets all model runs that started but never finished"""
        unfinished_runs = self.get_unfinished_runs()
        if unfinished_runs:
            for model_run in unfinished_runs:
                self.reset_model(model_run, clear_metadata)

    def run_models_on_dask(self):
        """Runs all the models by dispatching them to the corresponding dask client."""
        # First we do some setup, with two variables to keep track of things.
        model_runs = []
        parameter_list_empty = False
        # Now we "seed" the dask client with two model runs for every worker (assuming there are enough).
        for _ in range(2*self.processes):
            try:
                run_id, param_dict = self.parameter_list.next()
                model_run = self.dispatch_model_to_dask(run_id, param_dict)
                model_runs.append(model_run)
            except StopIteration:
                break
        # Now we loop, checking for finished runs and handeling the database work needed, and adding new runs.
        while True:
            # First we try finding any finished models in the model_runs list
            try:
                index = [model.status for model in model_runs].index('finished') # this is what should pop the ValueError
                # if we get one, pop it out, and record it in the database
                finished_run = model_runs.pop(index)
                self.record_finished_run(finished_run.result())
                # check to see if we have no more parameters to run, and no more model runs to wait for
                if parameter_list_empty and len(model_runs)==0:
                    # if we do, break out of the loop and end the function
                    break
                try:
                    # if so, try go grab a new parameter and add that to the dask queue to run
                    run_id, param_dict = self.parameter_list.next()
                    model_run = self.dispatch_model_to_dask(run_id, param_dict)
                    model_runs.append(model_run)
                except StopIteration:
                    # if we've run out of model parameters toggle our boolean so we just wait for our existing runs to finish
                    parameter_list_empty = True
            except ValueError:
                # if no models are finished running, check again
                # should we add a timer to wait here?
                pass

                 
    def record_finished_run(self, outputs):
        """For a given run_id set that run end time in the metadata table."""
        connection = sqlite3.connect(self.database, check_same_thread=False)
        cursor = connection.cursor()
        metadata_update_statement = "UPDATE model_run_metadata SET model_end_time = %f WHERE model_run_id = \"%s\"" %(outputs['end_time'], outputs['model_run_id'])
        cursor.execute(metadata_update_statement)
        valid_outputs = {key: outputs[key] for key in outputs.keys() if key in self.valid_outputs}
        columns = str(tuple(valid_outputs.keys()))
        values = str(tuple(valid_outputs.values()))
        query_str = "INSERT INTO model_run_outputs %s VALUES %s;" % (columns, values)
        cursor.execute(query_str)
        connection.commit()
        cursor.close()
                
    def dispatch_model_to_dask(self, run_id, param_dict):
        """For a given model parameter list, submit the model creation and run to dask."""
        model_run_id = str(uuid.uuid4())
        start_time = time.time()
        self.set_model_as_in_progress(self.batch_id, model_run_id, run_id, start_time)
        model_run = self.client.submit(make_and_run_model, self.model_class, self.batch_id, model_run_id, param_dict, self.out_dir)
        return model_run

    def set_model_as_in_progress(self, model_batch_id, model_run_id, param_run_id, start_time):
        """Update the metadata and parameter table to indicate that this given parameter set is running or queued for running."""
        connection = sqlite3.connect(self.database, check_same_thread=False)
        cursor = connection.cursor()
        update_statement = "UPDATE model_run_params SET model_batch_id = \"%s\", model_run_id = \"%s\" WHERE run_param_id = %d" % (model_batch_id, model_run_id, param_run_id)
        cursor.execute(update_statement)
        start_time = time.time()
        metadata_insert_statement = "INSERT INTO model_run_metadata (model_run_id, model_batch_id, model_start_time) VALUES (\"%s\", \"%s\", %f)" % (model_run_id, model_batch_id, start_time)
        cursor.execute(metadata_insert_statement)
        connection.commit()
        cursor.close()

    def dispatch_model(self, run_id, param_dict):
        """Create and run a model.  Used for single process mode."""
        # Probably could be rewritten to use some of the functions used in the dask mode.
        print("dispatching model %d" % run_id)
        connection = sqlite3.connect(self.database, check_same_thread=False)
        cursor = connection.cursor()
        model_run_id = str(uuid.uuid4())
        model = self.model_class(param_dict)
        model.batch_id = self.batch_id
        model.run_id = model_run_id
        update_statement = "UPDATE model_run_params SET model_batch_id = \"%s\", model_run_id = \"%s\" WHERE run_param_id = %d" % (model.batch_id, model.run_id, run_id)
        cursor.execute(update_statement)
        start_time = time.time()
        metadata_insert_statement = "INSERT INTO model_run_metadata (model_run_id, model_batch_id, model_start_time) VALUES (\"%s\", \"%s\", %f)" % (model.run_id, model.batch_id, start_time)
        cursor.execute(metadata_insert_statement)
        connection.commit()
        cursor.close()
        model.update_until(model.run_duration, model.dt)
        end_time = time.time()
        cursor = connection.cursor()
        metadata_update_statement = "UPDATE model_run_metadata SET model_end_time = %f WHERE model_run_id = \"%s\"" %(end_time, str(model.run_id))
        cursor.execute(metadata_update_statement)
        connection.commit()
        output_f = "%s%s.nc" % (self.out_dir, model.run_id)
        model.grid.save(output_f)

    
