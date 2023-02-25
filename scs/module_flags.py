from absl import app
from absl import flags


FLAGS = flags.FLAGS
# Flags that specify directories.
flags.DEFINE_string(
    "dir_models",
    "/lustre/lrspec/users/2649/models/transformer_testing",
    "Directory to create a new folder which will contain all of the data,"
    " models, and results from a training session.",
)
flags.DEFINE_string(
    "dir_raw",
    "/home/2649/repos/SCS/data/",
    "The absolute path for the directory where `sn_data.parquet` is currently"
    " located.",
)
flags.DEFINE_string(
    "dir_degraded",
    None,
    "The absolute path for the directory where `sn_data.C.parquet` and"
    " `sn_data.R.parquet` is to be saved.",
)
flags.DEFINE_string(
    "dir_preprocessed",
    None,
    "The absolute path for the directory where `sn_data.CP.parquet` and"
    " `sn_data.RP.parquet` is to be saved.",
)
flags.DEFINE_string(
    "dir_split",
    None,
    "The absolute path for the directory where `sn_data_trn.CP.parquet`,"
    " `sn_data_tst.CP.parquet`, `sn_data_trn.RP.parquet` and"
    " `sn_data_tst.CP.parquet` is to be saved.",
)
flags.DEFINE_string(
    "dir_augmented",
    None,
    "The absolute path for the directory where `sn_data_trn.CPA.parquet` and"
    " `sn_data_trn.RPA.parquet` is to be saved.",
)

# Flags for preprocessing parameters.
flags.DEFINE_float(
    "phase_range_lower",
    -20,
    "Spectra with phases below this value will be removed from the dataset in"
    " data preprocessing.",
)
flags.DEFINE_float(
    "phase_range_upper",
    50,
    "Spectra with phases above this value will be removed from the dataset in"
    " data preprocessing.",
)
flags.DEFINE_float(
    "ptp_range_lower",
    0.1,
    "Spectra with a flux range (`flux_max - flux_min`) below this value will"
    " be removed from the dataset during preprocessing. 'ptp' stands for"
    " 'peak-to-peak'.",
)
flags.DEFINE_float(
    "ptp_range_upper",
    100,
    "Spectra with a flux range (`flux_max - flux_min`) above this value will"
    " be removed from the dataset during preprocessing. 'ptp' stands for"
    " 'peak-to-peak'.",
)
flags.DEFINE_float(
    "wvl_range_lower",
    4500,
    "The lower limit of wavelength ranges to be included in the dataset. Flux"
    " values corresponding to wavelengths below this value will be set to"
    " zero for all spectra.",
)
flags.DEFINE_float(
    "wvl_range_upper",
    7000,
    "The upper limit of wavelength ranges to be included in the dataset. Flux"
    " values corresponding to wavelengths above this value will be set to"
    " zero for all spectra.",
)

# Flags for train-test split parameters.
flags.DEFINE_float(
    "train_frac",
    0.50,
    "Fraction of the data to allocate for the training set. The rest will be"
    " reserved for the testing set.",
)

# Flags for data augmentation parameters.
flags.DEFINE_float(
    "noise_scale",
    0.1,
    "The standard deviation of the noise to augment the data with. This number"
    " is a fraction of the standard deviation of the flux values in the data.",
)
flags.DEFINE_float(
    "spike_scale",
    3,
    "Scaling factor for the standard deviation of the spikes to augment the"
    " data with.",
)
flags.DEFINE_integer(
    "max_spikes",
    5,
    "Maximum number of spikes to augment a spectrum with.",
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
