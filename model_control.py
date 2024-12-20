from cli_functions import create, dispatch, slurm_config
import uuid
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="a CLI for generate model parameter databases and running landlab models based on them",
        usage=""" model_control <command> [<args>]

              The possible commands are:
              createdb    Create a sqlite database based on a model configuration file
              dispatch    Create and run landlab models based on a parameter database
              """)
    subparsers = parser.add_subparsers()
    parse_create = subparsers.add_parser("createdb")
    parse_dispatch = subparsers.add_parser("dispatch")
    parse_slurm = subparsers.add_parser("slurmitup")
    parse_create.add_argument('-t', '--template')
    parse_create.add_argument('-o', '--output')
    parse_create.set_defaults(func=create)

    parse_dispatch.add_argument('-d', '--database')
    parse_dispatch.add_argument('--one', action='store_true')
    parse_dispatch.add_argument('-m', '--model')
    parse_dispatch.add_argument('-f', '--filter')
    parse_dispatch.add_argument('-n', type=int)
    parse_dispatch.add_argument('-p', '--processes', type=int)
    parse_dispatch.add_argument('-od')
    parse_dispatch.add_argument('-c', '--clean', action='store_true')
    parse_dispatch.add_argument('-b', '--batch_id', default=uuid.uuid4())
    parse_dispatch.add_argument('-mid', '--model_id')
    
    parse_dispatch.set_defaults(func=dispatch)

    parse_slurm.add_argument('-d', '--database')
    parse_slurm.add_argument('-m', '--model')
    parse_slurm.add_argument('-od')
    parse_slurm.add_argument('-n', type=int)
    parse_slurm.add_argument('-f', '--filter')
    parse_slurm.add_argument('-scsv', '--slurm_csv')
    parse_slurm.add_argument('--checkout_models', action='store_true')
    parse_slurm.add_argument('-ntsks', '--num_tasks')
    parse_slurm.add_argument('--cpus')
    parse_slurm.add_argument('--sbatch_file')

    parse_slurm.set_defaults(func=slurm_config)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
