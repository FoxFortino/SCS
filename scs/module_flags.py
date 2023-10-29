from absl import app
from absl import flags


FLAGS = flags.FLAGS
# Flags that specify directories.
flags.DEFINE_string(
    "file_data_raw",
    "/home/2649/repos/SCS/data/sn_parquet",
    "The absolute path for the raw data, `sn_data.parquet`.",
)
flags.DEFINE_string(
    "spectralib",
    "/lustre/lrspec/users/2649/spectralib_v1",
    "Directory to create a new folder which will contain all of the data,"
)



# Flags for batch jobs
flags.DEFINE_string(
    "hp_set",
    None,
    "The name of a function in `hp_sets.py` which specifies the set of"
    " hyper-parameters to be used.",
)
flags.DEFINE_integer(
    "array_index",
    None,
    "Array index given by $SLURM_ARRAY_TASK_ID.",
)
flags.DEFINE_string(
    "num_requeue",
    None,
    "Number of restarts given by $SLURM_RESTART_COUNT.",
)

# Flags for miscellaneous parameters.
flags.DEFINE_integer(
    "R",
    None,
    "The spectroscopic resolution to degrade the data to.",
)
flags.DEFINE_integer(
    "random_state",
    1415,
    "NumPy random seed.",
)


def main(argv):
    del argv
    return


if __name__ == "__main__":
    app.run(main)
