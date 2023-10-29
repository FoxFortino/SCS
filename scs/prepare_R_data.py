from absl import app
from absl import flags
from absl import logging

# Python standard modules
import sys
import os
from os import mkdir
from os.path import join
from os.path import isdir
from os.path import isfile
from os.path import abspath
from shutil import rmtree

# Community Packages
import numpy as np
import pandas as pd

# My packages
import data_degrading as dd
import data_preparation as dp
import data_augmentation as da


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "R",
    None,
    "The spectroscopic resolution to degrade the data to.",
)
flags.DEFINE_string(
    "file_raw_data",
    "/home/2649/repos/SCS/data/sn_data.parquet",
    "The absolute path for the raw data, `sn_data.parquet`.",
)
flags.DEFINE_string(
    "spectralib",
    "/lustre/lrspec/users/2649/spectralib_v1",
    "Directory which contains the folders that hold the data for each R."
)
flags.DEFINE_bool(
    "remake_data",
    True,
    "Whether or not to completely redo the data or just exit if the data_R_dir folder exists."
)


def main(argv):
    del argv

    print(f"Preparing spectral data for R = {FLAGS.R}")
    df_RPA_trn, df_RP_tst = prepare_R_data(FLAGS.R, FLAGS.file_raw_data)

    data_R_dir = join(FLAGS.spectralib, f"{FLAGS.R}")
    print(f"Checking if directory {abspath(data_R_dir)} exists.")
    print("If `remake_data` is True, then it will be deleted and remade.")
    if isdir(data_R_dir):
        if FLAGS.remake_data:
            rmtree(data_R_dir)
            mkdir(data_R_dir)

    else:
        mkdir(data_R_dir)
        
    file_df_trn = join(data_R_dir, "df_trn.parquet")
    file_df_tst = join(data_R_dir, "df_tst.parquet")

    print(f"Saving '{file_df_trn}' and '{file_df_tst}'.")
    df_RPA_trn.to_parquet(file_df_trn)
    df_RP_tst.to_parquet(file_df_tst)


def prepare_R_data(
    R: int,
    file_raw_data: os.PathLike,
    phase_range=(-20, 50),
    ptp_range=(0.1, 100),
    wvl_range=(4500, 7000),
    train_frac=0.80,
    noise_scale=0.1,
    spike_scale=1.0,
    max_spikes=3,
    random_state=1415,
):
    print(f"Numpy random state: {random_state}")

    # Load the raw data
    print(f"Reading in raw data file: {abspath(file_raw_data)}")
    df_raw = pd.read_parquet(file_raw_data)

    # Degrade the data and only keep the rebinned dataset. Don't save it yet.
    print(f"Degrading the dataset to R = {R}")
    _, df_R = dd.degrade_dataframe(R, df_raw)

    # Preprocess the dataset
    print("Preprocessing the dataset.")
    print(f"Phase Range (in days): {phase_range}")
    print(f"Peak-to-Peak Range (in spectral units): {ptp_range}")
    print(f"Wavelength Range (in Angstroms): {wvl_range}")
    df_RP = dp.preproccess_dataframe(
        df_R,
        phase_range=phase_range,
        ptp_range=ptp_range,
        wvl_range=wvl_range,
    )

    # Perform the special train-test split
    print("Perform a special train-test split on the dataset.")
    print("This train-test split splits the dataset by SNe, not by spectra.")
    print(f"Fraction of SNe in the training set: {train_frac}")
    rng = np.random.RandomState(random_state)
    df_RP_trn, df_RP_tst = dp.split_data(df_RP, train_frac, rng)

    # Augment the training set
    print("Augmenting the dataset.")
    print(f"Scale of noise to augment the data with: {noise_scale}")
    print(f"Scale of the spikes to augment the data with: {spike_scale}")
    print(f"Maximum spikes to augment the dataset with: {max_spikes}")
    df_RPA_trn = da.augment(
        df_RP_trn,
        rng, 
        wvl_range=wvl_range,
        noise_scale=noise_scale,
        spike_scale=spike_scale,
        max_spikes=max_spikes
    )

    print("Data preparation complete.")
    return df_RPA_trn, df_RP_tst


if __name__ == "__main__":
    app.run(main)
