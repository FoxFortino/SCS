import os
from os.path import isdir
from os.path import dirname

import numpy as np
from numpy import typing as npt
import pandas as pd


def load_dataset(file_df):
    df = pd.read_parquet(file_df)
    return df

def save_noise_dataset(data, testing=False):
    if testing:
        np.save("../data/raw/snnoise_TESTING.npy", data)
    else:
        np.save("../data/raw/snnoise.npy", data)

def read_noise_dataset():
    return np.load("../data/raw/snnoise.npy")
def save_clean_dataset(df, testing=False):
    if testing:
        df.to_parquet("../data/raw/sn_clean_TESTING.parquet")
    else:
        df.to_parquet("../data/raw/sn_clean.parquet")

def extract_dataframe(sn_data):
    """
    Extract both metadata and flux data from a dataframe.
    """
    # Extract the row indices from the dataframe. These correspond to the SN
    # name of the spectrum at each row.
    index = sn_data.index

    # Extract the sub-dataframe that contains only the columns corresponding
    # to flux values. We do this specifically with a regex expression that
    # takes only the columns that start with a number.
    df_fluxes = sn_data.filter(regex="\d+")
    fluxes = df_fluxes.to_numpy(dtype=float)

    # Extract the columns that identify the flux columns. These will also be
    # the wavelengths at for each flux value, but as a string.
    flux_columns = df_fluxes.columns
    wvl = flux_columns.to_numpy(dtype=float)

    # In the even that more non-flux columns are added to these dataframes, we
    # find all of the columns representing the metadata (such as SN class,
    # spectral phase, etc.) by extracting all columns apart from
    # `flux_columns`.
    metadata_columns = sn_data.columns.difference(flux_columns)
    df_metadata = sn_data[metadata_columns]

    return (index, wvl,
            flux_columns, metadata_columns,
            df_fluxes, df_metadata,
            fluxes)


def preproccess_dataframe(
    sn_data,
    phase_range=(-20, 50),
    ptp_range=(0.1, 100),
    wvl_range=(4500, 7000),
    save_path=None,
):
    if (save_path is not None) and (not isdir(dirname(save_path))):
        raise FileNotFoundError(f"Directory '{dirname(save_path)}' does not exist.")

    # The function below neatly and reproducibly extracts all of the relevant 
    # subsets of the dataframe.
    data = extract_dataframe(sn_data)
    wvl0 = data[1]  # Wavelength array
    flux0_columns = data[2]  # Columns that index the fluxes in the dataframe
    fluxes0 = data[6]  # Only the flux values in a numpy array

    # Spectra with a spectral phase outside of `phase_range`.
    bad_ind = sn_data["Spectral Phase"] < phase_range[0]
    bad_ind |= sn_data["Spectral Phase"] > phase_range[1]

    # Remove the spectra with a range that is too small or too large.
    ptp = np.ptp(fluxes0, axis=1)
    bad_ind |= ptp < ptp_range[0]
    bad_ind |= ptp > ptp_range[1]

    # Remove rows from fluxes0 according to bad_ind
    fluxes0 = fluxes0[~bad_ind]

    # Standardize the dataset to zero mean and standard deviation of 1.
    flux_means = np.mean(fluxes0, axis=1)[..., None]
    flux_stds = np.std(fluxes0, axis=1)[..., None]
    standardized_fluxes0 = (fluxes0 - flux_means) / flux_stds

    # Set all flux values outside of `wvl_range` to 0.
    standardized_fluxes0[:, (wvl0 < wvl_range[0]) | (wvl0 > wvl_range[1])] = 0

    # Set the standardized flux data into the dataframe.
    sn_data.loc[~bad_ind, flux0_columns] = standardized_fluxes0

    # Remove the rows that we have pruned above.
    sn_data = sn_data.loc[~bad_ind]

    if save_path is not None:
        sn_data.to_parquet(save_path)
        print(f"Saved: {save_path}")

    return sn_data


def split_data(
    sn_data,
    train_frac,
    rng,
    save_path_trn=None,
    save_path_tst=None,
):
    if (save_path_trn is not None) and (save_path_tst is not None):
        if not isdir(dirname(save_path_tst)):
            raise FileNotFoundError(f"Directory '{dirname(save_path_tst)}' does not exist.")
        if not isdir(dirname(save_path_trn)):
            raise FileNotFoundError(f"Directory '{dirname(save_path_trn)}' does not exist.")

    sn_data_split = sn_data.groupby(by=["SN Subtype"],
                                    axis=0,
                                    group_keys=True).apply(df_split,
                                                           train_frac,
                                                           rng)
    training_set = sn_data_split["Training Set"] & ~sn_data_split["Exclude"]
    testing_set = ~sn_data_split["Training Set"] & ~sn_data_split["Exclude"]
    sn_data_trn = sn_data_split.loc[training_set]
    sn_data_tst = sn_data_split.loc[testing_set]

    sn_data_trn.reset_index(level="SN Subtype", drop=True, inplace=True)
    sn_data_tst.reset_index(level="SN Subtype", drop=True, inplace=True)

    if (save_path_trn is not None) and (save_path_tst is not None):
        sn_data_trn.to_parquet(save_path_trn)
        print(f"Saved: {save_path_trn}")
        sn_data_tst.to_parquet(save_path_tst)
        print(f"Saved: {save_path_tst}")

    return sn_data_trn, sn_data_tst


def df_split(x, train_frac, rng):
    x["Exclude"] = False
    x["Training Set"] = False

    sn_names = x.index.unique().to_list()
    num_supernova = len(sn_names)
    if num_supernova == 1:
        x["Exclude"] = True
        return x

    num_train = int(np.ceil(num_supernova * train_frac))
    if num_supernova - num_train == 0:
        num_train -= 1

    inds = rng.choice(sn_names,
                      size=num_train,
                      replace=False)
    x.loc[inds, "Training Set"] = True
    return x
