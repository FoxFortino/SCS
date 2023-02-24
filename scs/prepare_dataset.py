"""
Below is an explanation of the different default file names usedd and assumed
in this script. Each one represents a Pandas DataFrame that has been written
to a .parquet object with the pyarrow package.

    sn_data.parquet
        The original spectra.

    sn_data.C.parquet
        The spectra after it has been convolved in the aritifcial degradation,
        but it has not been rebinned to the new set of wavelengths defined by
        the new spectroscopic resolution, R. Generated after
        `data_degrading.degrade_dataframe` is performed on sn_data.parquet.
    sn_data.R.parquet
        The spectra after it has been artificially degraded and rebinned to
        the new set of wavelengths defined by the new spectroscopic
        resolution, R. Generated after `data_degrading.degrade_dataframe` is
        performed on sn_data.parquet.

    sn_data.CP.parquet
        The data from sn_data.C.parquet after it has been preprocessed with
        `data_preparation.preprocess_dataframe`.
    sn_data.RP.parquet
        The data from sn_data.R.parquet after it has been preprocessed with
        `data_preparation.preprocess_dataframe`.

    sn_data_trn.CP.parquet, sn_data_tst.CP.parquet
        The data from sn_data.CP.parquet after is has been split into a
        training and a testing set with `data_preparation.split_data`.
    sn_data_trn.RP.parquet, sn_data_tst.RP.parquet
        The data from sn_data.RP.parquet after is has been split into a
        training and a testing set with `data_preparation.split_data`.

    sn_data_trn.CPA.parquet
        The data from sn_data_trn.CP.parquet after it hasa been augmented with
        `data_augmenation.augment`.
    sn_data_trn.RPA.parquet
        The data from sn_data_trn.RP.parquet after it hasa been augmented with
        `data_augmenation.augment`.
"""
import os
from os.path import isfile
from os.path import isdir
from os.path import join

from absl import app
from absl import flags
import numpy as np

import data_loading as dl
import data_degrading as dd
import data_preparation as dp
import data_augmentation as da


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "R",
    None,
    "The spectroscopic resolution to degrade the data to.",
    required=True)
flags.DEFINE_string(
    "data_dir_original",
    "/home/2649/repos/SCS/data/",
    "The absolute path for the directory where sn_data.parquet is currently located.")
flags.DEFINE_string(
    "data_dir_degraded",
    "/home/2649/repos/SCS/data/",
    "The absolute path for the directory where sn_data.C.parquet and sn_data.R.parquet is to be saved.")
flags.DEFINE_string(
    "data_dir_preprocessed",
    "/home/2649/repos/SCS/data/",
    "The absolute path for the directory where sn_data.CP.parquet and sn_data.RP.parquet is to be saved.")
flags.DEFINE_string(
    "data_dir_train_test",
    "/home/2649/repos/SCS/data/",
    "The absolute path for the directory where sn_data_trn.CP.parquet, sn_data_tst.CP.parquet, sn_data_trn.RP.parquet and sn_data_tst.CP.parquet is to be saved.")
flags.DEFINE_string(
    "data_dir_augmented",
    "/home/2649/repos/SCS/data/",
    "The absolute path for the directory where sn_data_trn.CPA.parquet and sn_data_trn.RPA.parquet is to be saved.")
flags.DEFINE_float(
    "phase_range_lower",
    -20,
    "The lower limit of spectral phases to be included in the dataset.")
flags.DEFINE_float(
    "phase_range_upper",
    50,
    "The upper limit of spectral phases to be included in the dataset.")
flags.DEFINE_float(
    "ptp_range_lower",
    0.1,
    "The lower limit of spectral ranges (flux_max - flux_min) to be included in the dataset.")
flags.DEFINE_float(
    "ptp_range_upper",
    100,
    "The upper limit of spectral ranges (flux_max - flux_min) to be included in the dataset.")
flags.DEFINE_float(
    "wvl_range_lower",
    4500,
    "The lower limit of wavelength ranges to be included in the dataset. Flux values corresponding to wavelengths beyond this range will be set to zero.")
flags.DEFINE_float(
    "wvl_range_upper",
    7000,
    "The upper limit of wavelength ranges to be included in the dataset. Flux values corresponding to wavelengths beyond this range will be set to zero.")
flags.DEFINE_float(
    "train_frac",
    0.50,
    "Fraction of the data to allocate for the training set. The rest will be reserved for the testing set.")
flags.DEFINE_float(
    "noise_scale",
    0.25,
    "The standard deviation of the noise to augment the data with. This number is a fraction of the standard deviation of the flux values in the data.")
flags.DEFINE_float(
    "spike_scale",
    3,
    "Scaling factor for the standard deviation of the spikes to augment the data with.")
flags.DEFINE_integer(
    "max_spikes",
    5,
    "Maximum number of spikes to augment a spectrum with.")
flags.DEFINE_integer(
    "random_state",
    1415,
    "NumPy random seed.")


def main(argv):
    del argv
    sn_data_file = os.path.join(FLAGS.data_dir_original, "sn_data.parquet")
    phase_range = (FLAGS.phase_range_lower, FLAGS.phase_range_upper)
    ptp_range = (FLAGS.ptp_range_lower, FLAGS.ptp_range_upper)
    wvl_range = (FLAGS.wvl_range_lower, FLAGS.wvl_range_upper)

    prepare_dataset(
        FLAGS.R,
        sn_data_file,
        FLAGS.data_dir_degraded,
        FLAGS.data_dir_preprocessed,
        FLAGS.data_dir_train_test,
        FLAGS.data_dir_augmented,
        phase_range,
        ptp_range,
        wvl_range,
        FLAGS.train_frac,
        FLAGS.noise_scale,
        FLAGS.spike_scale,
        FLAGS.max_spikes,
        FLAGS.random_state,
    )


def prepare_dataset(
    R: int,
    sn_data_original_file: os.PathLike,
    data_dir_degraded: os.PathLike,
    data_dir_preprocessed: os.PathLike,
    data_dir_train_test: os.PathLike,
    data_dir_augmented: os.PathLike,
    phase_range: tuple[float, float],
    ptp_range: tuple[float, float],
    wvl_range: tuple[float, float],
    train_frac: float,
    noise_scale: float,
    spike_scale: float,
    max_spikes: int,
    random_state: int = 1415,
    redo_preprocess: bool = False,
    redo_split: bool = False,
    redo_augment: bool = False,
):
    # TODO(FoxFortino): Revisit how all of these checks are formatted. Consider that asserts should only 
    if not isfile(sn_data_original_file):
        raise FileNotFoundError(f"The specified file '{sn_data_original_file}' for `sn_data_original_file` does not exist.")
    if not isdir(data_dir_degraded):
        raise FileNotFoundError(f"The specified directory '{data_dir_degraded}' for `data_dir_degraded` does not exist.")
    if not isdir(data_dir_preprocessed):
        raise FileNotFoundError(f"The specified directory '{data_dir_preprocessed}' for `data_dir_preprocessed` does not exist.")
    if not isdir(data_dir_train_test):
        raise FileNotFoundError(f"The specified directory '{data_dir_train_test}' for `data_dir_train_test` does not exist.")
    if not isdir(data_dir_augmented):
        raise FileNotFoundError(f"The specified directory '{data_dir_augmented}' for `data_dir_augmented` does not exist.")

    # Conformity of `phase_range`
    assert (isinstance(phase_range, tuple)), f"`phase_range` must be a tuple but is {type(phase_range)}."
    assert (len(phase_range) == 2), f"`phase_range` must be length 2 but is {len(phase_range)}."
    assert (phase_range[0] < phase_range[1]), f"The first element of `phase_range` must be smaller than the second element but `phase_range` is {phase_range}."

    # Conformity of `ptp_range`
    assert (isinstance(ptp_range, tuple)), f"`ptp_range` must be a tuple but is {type(ptp_range)}."
    assert (len(ptp_range) == 2), f"`ptp_range` must be length 2 but is {len(ptp_range)}."
    assert (ptp_range[0] < ptp_range[1]), f"The first element of `ptp_range` must be smaller than the second element but `ptp_range` is {ptp_range}."

    # Conformity of `wvl_range`
    assert (isinstance(wvl_range, tuple)), f"`wvl_range` must be a tuple but is {type(wvl_range)}."
    assert (len(wvl_range) == 2), f"`wvl_range` must be length 2 but is {len(wvl_range)}."
    assert (wvl_range[0] < wvl_range[1]), f"The first element of `wvl_range` must be smaller than the second element but `wvl_range` is {wvl_range}."

    # Conformity of `train_frac`
    assert (isinstance(train_frac, float)), f"`train_frac` must be a float but is {type(train_frac)}."
    assert (0 < train_frac < 1), f"`train_frac` must be between 0 and 1 but is {train_frac}."

    # Conformity of `noise_scale`
    assert (isinstance(noise_scale, float)), f"`noise_scale` must be a float but is {type(noise_scale)}."
    assert (0 < noise_scale), f"`noise_scale` must be greater than 0 sssbut is {noise_scale}."

    rng = np.random.RandomState(random_state)
    df_original = dl.load_sn_data(sn_data_original_file)

    file_C = join(data_dir_degraded, "sn_data.C.parquet")
    file_CP = join(data_dir_preprocessed, "sn_data.CP.parquet")
    file_CP_trn = join(data_dir_train_test, "sn_data_trn.CP.parquet")
    file_CP_tst = join(data_dir_train_test, "sn_data_tst.CP.parquet")
    file_CPA_trn = join(data_dir_augmented, "sn_data_trn.CPA.parquet")

    file_R = join(data_dir_degraded, "sn_data.R.parquet")
    file_RP = join(data_dir_preprocessed, "sn_data.RP.parquet")
    file_RP_trn = join(data_dir_train_test, "sn_data_trn.RP.parquet")
    file_RP_tst = join(data_dir_train_test, "sn_data_tst.RP.parquet")
    file_RPA_trn = join(data_dir_augmented, "sn_data_trn.RPA.parquet")

    do_degrade = True
    if isfile(file_C) and isfile(file_R):
        do_degrade = False  # Degraded already done
        df_C = dl.load_sn_data(file_C)
        df_R = dl.load_sn_data(file_R)
        print()

    do_preprocess = True
    if isfile(file_CP) and isfile(file_RP) and (not redo_preprocess):
        do_preprocess = False  # Preprocessing already done
        df_CP = dl.load_sn_data(file_CP)
        df_RP = dl.load_sn_data(file_RP)
        print()

    do_split = True
    if (
        isfile(file_CP_trn)
        and isfile(file_CP_tst)
        and isfile(file_RP_trn)
        and isfile(file_RP_tst)
        and (not redo_split)
    ):
        do_split = False  # Train-test split already done
        df_CP_trn = dl.load_sn_data(file_CP_trn)
        df_CP_tst = dl.load_sn_data(file_CP_tst)
        df_RP_trn = dl.load_sn_data(file_RP_trn)
        df_RP_tst = dl.load_sn_data(file_RP_tst)
        print()

    do_augment = True
    if isfile(file_CPA_trn) and isfile(file_RPA_trn) and (not redo_augment):
        do_augment = False  # Data augmentation already done
        df_CPA_trn = dl.load_sn_data(file_CPA_trn)
        df_RPA_trn = dl.load_sn_data(file_RPA_trn)
        print()

    # Degrade the original spectra. This operation produces two new
    # dataframes. The `_C` or 'convolved' dataframe contains the spectral data
    # just after it has been convolved in the artificial spectral degradation
    # process and the flux values correspond to the wavelength bins of the
    # original data. The `_R` or `rebinned` data contains the spectral data
    # after the fluxes have been rebinned to the new set of wavelengths
    # defined by the chose spectral resolution, R.
    if do_degrade:
        print(f"Artificially degrading the following dataset to a spectral resolution of {R}:")
        print(f"    {sn_data_original_file}")
        df_C, df_R = dd.degrade_dataframe(
            R,
            df_original,
            save_path_C=file_C,
            save_path_R=file_R)
        print(f"Degrading completed.")
        print()

    # Preprocess both of the dataframes from above and save them.
    # Preprocessing involves (not in this order) standardizing the data to
    # mean zero and standard deviation one, setting all flux values outside of
    # the wavelengths defined by `wvl_range` to be zero, remove spectra that
    # have a phase outside of `phase_range`, and remove spectra that have a
    # flux range (flux_max - flux_min) outside of `ptp_range`.
    if do_preprocess:
        print(f"Preprocesing the following dataset files:")
        print(f"    {file_C}")
        print(f"    {file_R}")
        df_CP = dp.preproccess_dataframe(
            df_C,
            phase_range=phase_range,
            ptp_range=ptp_range,
            wvl_range=wvl_range,
            save_path=file_CP)
        df_RP = dp.preproccess_dataframe(
            df_R,
            phase_range=phase_range,
            ptp_range=ptp_range,
            wvl_range=wvl_range,
            save_path=file_RP)
        print(f"Preprocessing complete.")
        print()

    # Perform the special train-test split on both the convolved and rebinned
    # datasets. NOTE THAT THE SPLIT WILL NOT NECCESSARILY BE THE SAME BETWEEN
    # THEM.
    if do_split:
        print(f"Performing special train-test split on the following dataset files:")
        print(f"    {file_CP}")
        print(f"    {file_RP}")
        df_CP_trn, df_CP_tst = dp.split_data(
            df_CP,
            train_frac,
            rng,
            save_path_trn=file_CP_trn,
            save_path_tst=file_CP_tst)
        df_RP_trn, df_RP_tst = dp.split_data(
            df_RP,
            train_frac,
            rng,
            save_path_trn=file_RP_trn,
            save_path_tst=file_RP_tst,)
        print(f"Train-test split complete.")
        print()

    # Augment the training data for both the convolved and rebinned datasets.
    if do_augment:
        print(f"Performing data augmentation on the following dataset files:")
        print(f"    {file_CP_trn}")
        print(f"    {file_CP_tst}")
        df_CPA_trn = da.augment(
            df_CP_trn,
            rng,
            wvl_range=wvl_range,
            noise_scale=noise_scale,
            spike_scale=spike_scale,
            max_spikes=max_spikes,
            save_path=file_CPA_trn)
        df_RPA_trn = da.augment(
            df_RP_trn,
            rng,
            wvl_range=wvl_range,
            noise_scale=noise_scale,
            spike_scale=spike_scale,
            max_spikes=max_spikes,
            save_path=file_RPA_trn)
        print(f"Data augmentation complete.")
        print()


if __name__ == "__main__":
    app.run(main)
