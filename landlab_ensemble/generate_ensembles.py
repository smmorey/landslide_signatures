import json
import re
import numpy as np
from copy import deepcopy
import sqlite3
from collections.abc import MutableMapping

MODEL_PARAM_TABLE_SQL_START = """
CREATE TABLE model_run_params (
    run_param_id INTEGER PRIMARY KEY AUTOINCREMENT,
"""
MODEL_PARAM_TABLE_SQL_END = """
    model_run_id TEXT,
    model_batch_id TEXT
);
"""

MODEL_RUN_TABLE_SQL = """
CREATE TABLE model_run_metadata (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_run_id TEXT,
    model_batch_id TEXT,
    model_start_time REAL,
    model_end_time REAL
);
"""

PARAM_DIM_TABLE_SQL = """
CREATE TABLE model_param_dimension (
    param_name TEXT,
    python_type TEXT
);
                      """

OUTPUT_TABLE_SQL = """
CREATE TABLE model_run_outputs (
    output_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_run_id TEXT,
    model_batch_id TEXT,
"""

def _flatten_dict_gen(d, parent_key, sep):
    """Takes a dictionary and returns a new "flat" generator with old heirarchy represented in key (from StackOverflow)."""
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v

def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.'):
    """Takes a dictionary and returns a new "flat" dictionary with the old heirarchy represented in the key.
    This function taken from StackOverflow.  Assuming a parent key of "EXAMPLE" and a sep of ".", the
    following behavior is expected:
    {'key_1': 'value',
     'my_dict': {'key_2': 'value 2',
                 'dict_2': {'key_3': 'value_3'}}}
    would return the following dictionary:
    {'EXAMPLE.key_1': 'value',
     'EXAMPLE.my_dict.key_2': 'value_2',
     'EXAMPLE.my_dict.dict_2.key_3': 'value_3'}
    """
    return dict(_flatten_dict_gen(d, parent_key, sep))

def python_type_to_sql_type(value):
    """Maps python type to type of sqlite column that will be used to store it."""
    if isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "REAL"
    else:
        return "TEXT"

def generate_model_param_table_sql(run_params):
    """Returns a sqlite string for a table with a column for every model parameter."""
    table_creation_sql = MODEL_PARAM_TABLE_SQL_START
    flat_run_params = flatten_dict(run_params, "model_param")
    for param, value in flat_run_params.items():
        param_type = python_type_to_sql_type(value)
        table_creation_sql += "\"%s\" %s," % (param, param_type)
    table_creation_sql += MODEL_PARAM_TABLE_SQL_END
    return table_creation_sql

def generate_model_param_dim_table_sql(run_params):
    """Returns a sqlite string to insert parameter and value information into the dimension table.""" 
    flat_run_params = flatten_dict(run_params, "model_param")
    insertion_string = str([(key, str(type(value))) for key, value in flat_run_params.items()])[1:-1]
    insert_sql = "INSERT INTO model_param_dimension (param_name, python_type) VALUES %s" % insertion_string
    return insert_sql

def generate_model_output_table_sql(run_params):
    outputs = run_params['output_fields']
    table_creation_sql = OUTPUT_TABLE_SQL
    table_creation_sql += ", ".join(["\"%s\" REAL" % output for output in outputs])
    table_creation_sql += ");"
    return table_creation_sql
        

def generate_model_run_db(db_path, params):
    """Generates a sqlite table with parameter information."""
    sqliteConnection = sqlite3.connect(db_path)
    cursor = sqliteConnection.cursor()
    model_param_sql = generate_model_param_table_sql(params)
    cursor.execute(model_param_sql)
    cursor.execute(MODEL_RUN_TABLE_SQL)
    cursor.execute(PARAM_DIM_TABLE_SQL)
    param_dim_sql = generate_model_param_dim_table_sql(params)
    cursor.execute(param_dim_sql)
    output_sql = generate_model_output_table_sql(params)
    cursor.execute(output_sql)
    sqliteConnection.commit()
    cursor.close()

DYNAMIC_PARAM_KEYS = ["ITERATIVE", "RANDOM"]
ITER_PARAM_RE = re.compile(r"ITERATIVE\s+(\w+)\s+(\{.*\})")
RANDOM_PARAM_RE = re.compile(r"RANDOM\s+(\w+)\s+(\{.*\})")
VALID_GENERATORS = {"linspace": np.linspace,
                    "arange": np.arange,
                    "logspace": np.logspace,
                    "geomspace": np.geomspace}
#                    "randint": np.random.randint,
#                    "randfloat"}

def get_dynamic_params(paramaters):
    """Given a parameter dictionary, it returns all keys that are iterable.

    This function flattens the dictionary so all keys are returned in a "flat"
    format that represents the heirarchy of the keys (see flatten_dict).  It
    returns all keys that have a value that starts with the string "ITERATIVE"
    """
    fparams = flatten_dict(paramaters)#, "model_param")
    dynamic_keys = []
    for key, value in fparams.items():
        if isinstance(value, str) and value.split(" ")[0].upper() in DYNAMIC_PARAM_KEYS:
            param_type = value.split(" ")[0].upper()
            dynamic_keys.append((key, param_type))
    return dynamic_keys

def generate_iterative_parameter_array(interative_param_value):
    """For an appropriatly defined iterative parameter value, returns the corresponding numpy array.

    This function expects an iterative parameter to be defined with the following pattern:
    "ITERATIVE <numpy generator function> {"<argument_name>": <argument value>,...}"
    It extracts the generator function and arguments based on this pattern, checks that the generator
    function is an allowed function, and then runs that function with the corresponding arguments
    """
    match = ITER_PARAM_RE.match(interative_param_value)
    function = VALID_GENERATORS[match.group(1)]
    args = json.loads(match.group(2))
    return function(**args)

def generate_random_parameter_array(random_param_value, rng=np.random.default_rng()):
    """For an appropriatly defined random parameter valye, return a random numpy array.

    This function expects a random parameter to defined with the following pattern:
    "RANDOM" <numpy generator function {"<argument_name>": <argument value>, ...}"
    It extracts the generator function and the arguments for that function.  It also
    allows for two arguments that do not belong to the generator function: "scaler"
    and "shifter" which allow the user to specify an ammount to multipy or add to
    the randomly generated number.
    """
    match = RANDOM_PARAM_RE.match(random_param_value)
    function = getattr(rng, match.group(1))
    args = json.loads(match.group(2))
    scaler = args.pop("scaler", 1)
    shifter = args.pop("shifter", 0)
    if 'size' not in args:
        args['size'] == (1)
    return scaler*function(**args)+shifter

class ModelParams:
    """An object that iterates over all possible parameter combinations.

    The ModelParams pbject expects a paramter dictionary where some parameters are
    defined as "ITERATIVE" (see documentation for generate_parameter_array).  This
    iterator then will return every possible paramter combination based on the iterative
    parameters.

    Attributes:
        parameters -- the parameter dictionary
        iterative_params -- the keys of the parameters that are iterative
        iterative_parameter_values -- all possible combinations of iterative parameters
        current -- the current parameter combination to be returned
    
    """
    
    def __init__(self, parameters):
        """Initializes the iteratator based on a parameter array.

        This finds the iterative parameters and creates a matrix where each row represents

        a possible parameter combination.
        Args:
            parameters -- a parameter dictionary
        """
        self.parameters = parameters
        self.dynamic_params = get_dynamic_params(parameters)
        self.rng = np.random.default_rng()
        flat_params = flatten_dict(parameters)#, "model_param")
        parameter_arrays = [self.generate_parameter_array(flat_params[param[0]], param[1]) for param in self.dynamic_params if param[1] in ("ITERATIVE", "RANDOM")]
        self.iterative_parameter_values = np.array(np.meshgrid(*parameter_arrays)).T.reshape(-1,len(parameter_arrays))
        self.current = 0

    def generate_parameter_array(self, parameter, dynamic_type):
        """Generates the parameter array"""
        if dynamic_type == "ITERATIVE":
            return generate_iterative_parameter_array(parameter)
        elif dynamic_type == "RANDOM":
            return generate_random_parameter_array(parameter, self.rng)
        else:
            raise ValueError

    def __iter__(self):
        """Returns the object as it is an iterator."""
        return self
        
    def __next__(self):
        """Calls the next function, which returns the next combination of parameters as a dictionary"""
        return self.next()
        
    def next(self):
        """" Returns the next combination of parameters as a dictionary"""
        if self.current >= len(self.iterative_parameter_values):
            raise StopIteration
        iterative_param_values = self.iterative_parameter_values[self.current]
        params_to_return = deepcopy(self.parameters)
        for i, param in enumerate(self.dynamic_params):
            param = param[0]
            param_val = iterative_param_values[i]
            working_params = params_to_return
            for key in param.split('.')[:-1]:
                working_params = working_params[key]
            working_params[param.split('.')[-1]] = param_val
        self.current += 1
        return params_to_return

def insert_model_run(cursor, params):
    """Flattens a parameter dictionary and inserts it with the provided sqlite cursor."""
    flat_params = flatten_dict(params, "model_param")
    columns = str(tuple(flat_params.keys()))#.replace('[', '(').replace(']', ')')
    values = []
    for value in flat_params.values():
        if not isinstance(value, (int, float)):
            values.append(str(value))
        else:
            values.append(value)
    values = str(tuple(values))#.replace('[', '(').replace(']', ')')
    query_str = "INSERT INTO model_run_params %s VALUES %s;" % (columns, values)
    cursor.execute(query_str)

def create_model_db(db_path, param_path):
    """Given a model parameter json file path, creates and fills an associated model database."""
    with open(param_path, 'r') as param_f:
        params = json.load(param_f)
    model_params = ModelParams(params)
    param = model_params.next()
    generate_model_run_db(db_path, param)
    sqliteConnection = sqlite3.connect(db_path)
    cursor = sqliteConnection.cursor()
    insert_model_run(cursor, param)
    for param in model_params:
        insert_model_run(cursor, param)
    sqliteConnection.commit()
    cursor.close()
