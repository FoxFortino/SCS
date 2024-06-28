import argparse
from shutil import rmtree
from os import mkdir
from os import remove
from os.path import join
from os.path import isdir
from os.path import isfile
from glob import glob
import json

import numpy as np

import hp_sets
from learn import load_json
from learn import run_SCS
import data_plotting as dplt


def main(args):
    print(args)
    
    array_index = args.SLURM_ARRAY_TASK_ID
    hp_set = args.hp_set
    if args.SLURM_RESTART_COUNT == "":
        restart_count = 0
    else:
        restart_count = int(args.SLURM_RESTART_COUNT)
    hp, PG = get_param_grid(array_index, args.hp_set)
    
    dir_batch_model = args.dir_batch_model
    dir_model = join(dir_batch_model, f"{array_index:0>3}_model")
    dir_model_backup = join(dir_model, "backup")
    dir_model_data = join(dir_model, "data")
    dir_model_results = join(dir_model, "results")
    

    resume = check_dirs(dir_batch_model, array_index, restart_count, args.reuse_data)
    print(f"Resume: {resume}")
    
    run_SCS(
        dir_model,
        args.R,
        hp,
        file_raw_data=args.file_raw_data,
        resume=resume
    )


def get_metric_grid(PG, model_dirs):
    keys, vals = get_varying_hp(PG)
    assert "retry" in keys
    
    grids = np.meshgrid(*vals, indexing="ij")
    metric_grid_shape = (*grids[0].shape, 2, 3)
    metric_grid = np.full(metric_grid_shape, np.nan)

    for dir_model in model_dirs:
        file_model_results = join(dir_model, "results", "results.json")
        file_model_hp = join(dir_model, "hp.json")
        
        hp = load_json(file_model_hp)
        results = load_json(file_model_results)
        
        params = []
        for key in keys:
            params.append(hp[key])

        param_inds = [val == grid for val, grid in zip(params, grids)]
        index = np.where(np.logical_and.reduce(param_inds))
        index = np.hstack(index)
        
        metric_grid[(*index, 0, 0)] = results["trn"]["loss"]
        metric_grid[(*index, 0, 1)] = results["trn"]["ca"]
        metric_grid[(*index, 0, 2)] = results["trn"]["f1"]
        metric_grid[(*index, 1, 0)] = results["tst"]["loss"]
        metric_grid[(*index, 1, 1)] = results["tst"]["ca"]
        metric_grid[(*index, 1, 2)] = results["tst"]["f1"]

    return metric_grid, keys, vals


def get_varying_hp(PG):
    pg0 = PG.param_grid[0]
    
    keys = []
    vals = []
    for key, val in pg0.items():
        if len(val) > 1:
            keys.append(key)
            vals.append(val)
        
    return keys, vals
    
    
def check_batch_completion(dir_batch_model, PG):
    num_models = len(PG)
    
    # Check if there are as many directories in dir_batch_model as num_models.
    model_dirs = glob(join(dir_batch_model, "[0-9][0-9][0-9]_model/"))
    for dir_model in model_dirs:
        file_model_curves = join(dir_model, "results", "curves.pdf")
        if not isfile(file_model_curves):
            print("XXXXXXXXXXXX FALSE XXXXXXXXX")
            return False

    return model_dirs


def check_dirs(dir_batch_model, array_index, restart_count, reuse_data):
    assert isdir(dir_batch_model), f"'{dir_batch_model}' does not exist."
    
    dir_model = join(dir_batch_model, f"{array_index:0>3}_model")
    dir_model_backup = join(dir_model, "backup")
    dir_model_data = join(dir_model, "data")
    dir_model_results = join(dir_model, "results")
    
    file_model = join(dir_model, "model.hdf5")
    file_model_history = join(dir_model, "history.log")
    file_model_results = join(dir_model_results, "results.json")
    file_model_hp = join(dir_model, "hp.json")
    file_model_curves = join(dir_model_results, "curves.pdf")
    
    file_df_trn = join(dir_model_data, "df_trn.parquet")
    file_df_tst = join(dir_model_data, "df_tst.parquet")
    
    if isdir(dir_model) and (restart_count == 0) and reuse_data:
        resume = True
        assert isfile(file_df_trn)
        assert isfile(file_df_tst)

        if isfile(file_model): remove(file_model)
        if isfile(file_model_history): remove(file_model_history)
        if isfile(file_model_hp): remove(file_model_hp)
        if isfile(file_model_results): remove(file_model_results)
        if not isdir(dir_model_backup): mkdir(dir_model_backup)
        if not isdir(dir_model_results): mkdir(dir_model_results)
    
    elif isdir(dir_model) and (restart_count == 0):
        resume = False
        rmtree(dir_model)
        mkdir(dir_model)
        mkdir(dir_model_backup)
        mkdir(dir_model_data)
        mkdir(dir_model_results)
        
    elif isdir(dir_model) and (restart_count != 0):
        resume = True
    
    elif not isdir(dir_model):
        resume = False
        mkdir(dir_model)
        mkdir(dir_model_backup)
        mkdir(dir_model_data)
        mkdir(dir_model_results)

    return resume


def get_param_grid(array_index, hp_set):
    PG = eval(f"hp_sets.{hp_set}()")
    if array_index >= len(PG):
        sys.exit("Array index out of range.")
    hp = PG[int(array_index)]
    return hp, PG


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, required=True)
    parser.add_argument("--dir_batch_model", required=True)
    parser.add_argument("--SLURM_ARRAY_TASK_ID", type=int, required=True)
    parser.add_argument("--SLURM_RESTART_COUNT", required=True)
    parser.add_argument("--hp_set", required=True)
    parser.add_argument("--reuse_data", action="store_true")
    parser.add_argument(
        "--file_raw_data",
        default="/home/2649/repos/SCS/data/sn_data.parquet",
    )
    args = parser.parse_args()
    main(args)