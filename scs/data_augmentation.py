from os.path import isdir
from os.path import dirname

import numpy as np
from numpy import typing as npt
import pandas as pd
from scipy import stats

import data_preparation as dp


def augment(
    sn_data,
    rng,
    wvl_range,
    noise_scale,
    spike_scale,
    max_spikes,
    save_path=None,
):
    if (save_path is not None) and (not isdir(dirname(save_path))):
        raise FileNotFoundError(f"Directory '{dirname(save_path)}' does not exist.")

    data = dp.extract_dataframe(sn_data)
    wvl0 = data[1]
    flux_columns = data[2]
    fluxes = data[6]

    sn_types, num_spectra = np.unique(sn_data["SN Subtype"], return_counts=True)
    num_augments = (np.max(num_spectra) - num_spectra) / num_spectra
    num_augments = np.ceil(num_augments).astype(int) + 1

    sn_type_df_list = []
    for sn_type, num_augment in zip(sn_types, num_augments):
        df_copy = sn_data.copy(deep=True)
        sn_type_mask = sn_data["SN Subtype"] == sn_type
        sn_type_df = df_copy[sn_type_mask]

        sn_type_df_rep = sn_type_df.iloc[np.tile(np.arange(sn_type_df.shape[0]), num_augment)].copy(deep=True)

        data = dp.extract_dataframe(sn_type_df_rep)
        index = data[0]
        wvl0 = data[1]
        flux_columns = data[2]
        metadata_columns = data[3]
        df_fluxes = data[4]
        df_metadata = data[5]
        fluxes = data[6]

        # Mask the arrays so we only augment the region specified by `wvl_range`.
        wvl_range_mask = (wvl0 < wvl_range[0]) | (wvl0 > wvl_range[1])
        masked_fluxes = fluxes[:, ~wvl_range_mask]
        masked_wvl0 = wvl0[~wvl_range_mask]

        # Generate noise to augment the data
        masked_noise = gen_noise(masked_fluxes, noise_scale, rng)

        # Generate spikes to augment the data
        masked_spikes = gen_spikes(masked_fluxes, masked_wvl0, spike_scale, max_spikes, rng)

        # Add the augmented noise and spikes to the fluxes.
        fluxes[:, ~wvl_range_mask] = masked_fluxes + masked_noise + masked_spikes
        sn_type_df_rep.loc[:, flux_columns] = fluxes
        sn_type_df_list.append(sn_type_df_rep)

    sn_data_augmented = pd.concat(sn_type_df_list, axis=0)

    if save_path is not None:
        sn_data_augmented.to_parquet(save_path)
        print(f"Saved: {save_path}")

    return sn_data_augmented


def gen_noise(spectrum, noise_scale, rng):
    stddev = spectrum.std()
    scale = stddev * noise_scale

    noise = stats.norm.rvs(loc=0, scale=scale, size=spectrum.size, random_state=rng)

    return noise
gen_noise = np.vectorize(gen_noise, signature="(n),(),()->(n)")


def gen_spikes(spectrum, wvl, spike_scale, max_spikes, rng):
    num_spikes = stats.randint.rvs(
        low=0,
        high=max_spikes,
        size=1,
        random_state=rng)[0]

    spike_wvls = stats.uniform.rvs(
        loc=wvl[0],
        scale=wvl[-1] - wvl[0],
        size=num_spikes,
        random_state=rng)

    spike_inds = np.digitize(spike_wvls, wvl)

    spike_dir = stats.binom.rvs(
        n=1,
        p=0.50,
        size=num_spikes,
        random_state=rng)
    spike_dir[spike_dir == 0] = -1

    std = spectrum.std()
    scale = std * spike_scale
    spike_mag = stats.norm.rvs(
        loc=0,
        scale=scale,
        size=num_spikes,
        random_state=rng)

    spikes = np.zeros_like(spectrum)
    spikes[spike_inds] = spike_mag * spike_dir

    return spikes
gen_spikes = np.vectorize(gen_spikes, signature="(n),(m),(),(),()->(n)")
