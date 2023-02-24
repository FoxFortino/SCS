import shutil
from pprint import pprint
import sys
import os
from os.path import isfile
from os.path import isdir
from os.path import join

from absl import app
from absl import flags
import numpy as np

import learn
import data_loading as dl
import data_degrading as dd
import data_preparation as dp
import data_augmentation as da
import model_searches


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "dir_library",
    "/lustre/lrspec/users/2649/spectralib_v1",
    "Specify the directory where the degraded spectra for each R exist."
)
flags.DEFINE_string(
    "dir_models",
    "/lustre/lrspec/users/2649/models/transformer_testing",
    "Specify the directory where the resulting models will be saved."
)
flags.DEFINE_integer(
    "array_index",
    None,
    "Array index given by $SLURM_ARRAY_TASK_ID."
)
flags.DEFINE_string(
    "num_requeue",
    None,
    "Number of restarts given by $SLURM_RESTART_COUNT."
)
flags.DEFINE_string(
    "R_or_C",
    "R",
    "Whether to use rebinned (R) or convolved (C) data."
)
flags.DEFINE_string(
    "model_search_name",
    None,
    "The function name from `model_searches.py` that you want to use."
)


def main(argv):
    del argv

    restart_fit = True if (FLAGS.num_requeue == "") else False
    print(f"`FLAGS.num_requeue`: {FLAGS.num_requeue}")
    print(f"`restart_fit`: {restart_fit}")

    PG = eval(f"model_searches.{FLAGS.model_search_name}()")
    if FLAGS.array_index >= len(PG):
        sys.exit(f"Array index was {FLAGS.array_index} but Parameter Grid is size {len(PG)}.")

    data_dir_original = FLAGS.data_dir_original
    dir_library = FLAGS.dir_library
    dir_models = FLAGS.dir_models

    model_dir = join(dir_models, f"{FLAGS.R}_{FLAGS.model_search_name}_{FLAGS.array_index}")
    backup_dir = join(model_dir, "backup")
    model_dir_data = join(model_dir, "data")

    # Directory making
    if os.path.isdir(model_dir) and restart_fit:
        shutil.rmtree(model_dir)
        print(f"Removed {model_dir}")

    print(f"Created {model_dir}")
    os.mkdir(model_dir)
    print(f"Created {backup_dir}")
    os.mkdir(backup_dir)
    print(f"Created {model_dir_data}")
    os.mkdir(model_dir_data)

    hp = PG[int(FLAGS.array_index)]
    print("Passing the following parameters to the `learn.train`:")
    pprint(hp)

    file_trn = join(model_dir_data, f"sn_data_trn.{FLAGS.R_or_C}PA.parquet")
    file_tst = join(model_dir_data, f"sn_data_tst.{FLAGS.R_or_C}P.parquet")

    learn.train(
        FLAGS.R,
        model_dir,
        data_dir_original,
        model_dir_data,
        model_dir_data,
        model_dir_data,
        model_dir_data,
        file_trn,
        file_tst,
        hp,
        restart_fit=restart_fit,
        num_epochs=100000,
        batch_size=32,
        verbose=2)

if __name__ == "__main__":
    app.run(main)
