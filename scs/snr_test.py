import sys
from os.path import isfile

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.signal import savgol_filter

from tensorflow.keras import callbacks
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Nadam

from tensorflow_addons.metrics import F1Score

# My packages
sys.path.insert(0, "../scs/")
import data_degrading as dd
import data_preparation as dp
import data_augmentation as da
from prepare_datasets_for_training import extract
import data_plotting as dplt
import scs_config

sys.path.insert(0, "../scs/models/")
import feed_forward
import transformer_encoder


def get_noise_scale_arr():
    noise_scale_arr = np.linspace(0, 10, num=101)
    return noise_scale_arr


def load_original_dataset():
    file_df_raw = "../data/raw/sn_data.parquet"
    df_raw = pd.read_parquet(file_df_raw)
    return df_raw


def inject_noise(df_raw, rng, noise_scale):
    data = dp.extract_dataframe(df_raw)
    index, wvl, flux_columns, metadata_columns, df_fluxes, df_metadata, fluxes = data
    fluxes_noise = fluxes + gen_noise(fluxes, noise_scale, rng)
    df_raw[flux_columns] = fluxes_noise
    return df_raw


def degrade_data(df_raw, R):
    df_C, df_R = dd.degrade_dataframe(R, df_raw)
    return df_C, df_R


def clean_data(df_C, df_R, phase_range, ptp_range, wvl_range):
    df_CP = dp.preproccess_dataframe(
        df_C,
        phase_range=phase_range,
        ptp_range=ptp_range,
        wvl_range=wvl_range,
    )
    df_RP = dp.preproccess_dataframe(
        df_R,
        phase_range=phase_range,
        ptp_range=ptp_range,
        wvl_range=wvl_range,
    )
    return df_CP, df_RP


def split_train_test(df_CP, df_RP, train_frac, rng):
    df_CP_trn, df_CP_tst = dp.split_data(df_CP, train_frac, rng)
    df_RP_trn, df_RP_tst = dp.split_data(df_RP, train_frac, rng)
    return df_CP_trn, df_CP_tst, df_RP_trn, df_RP_tst


def augment_training_set(df_CP_trn, df_RP_trn, rng, wvl_range, spike_scale, max_spikes):
    df_CPA_trn = da.augment(
        df_CP_trn,
        rng,
        wvl_range=wvl_range,
        noise_scale=0,
        spike_scale=spike_scale,
        max_spikes=max_spikes,
    )
    df_RPA_trn = da.augment(
        df_RP_trn,
        rng,
        wvl_range=wvl_range,
        noise_scale=0,
        spike_scale=spike_scale,
        max_spikes=max_spikes,
    )
    return df_CPA_trn, df_RPA_trn


def get_model(input_shape, num_classes):
    model = feed_forward.model(
        input_shape,
        num_classes,
        [1024, 1024, 1024],
        activation="relu",
        dropout=0.1,
    )


def main():
    df_raw = load_original_dataset()

    rng = np.random.RandomState(1415)
    noise_scale_arr = get_noise_scale_arr()
    noise_scale = noise_scale_arr[noise_scale_i]
    df_raw = inject_noise(df_raw, rng, noise_scale)

    R = 100
    df_C, df_R = degrade_data(df_raw)

    phase_range = (-20, 50)
    ptp_range = (0.1, 100)
    wvl_range = (4500, 7000)
    df_CP, df_RP = clean_data(df_C, df_R)

    train_frac = 0.50

    df_CP_trn, df_CP_tst, df_RP_trn, df_RP_tst = split_train_test(
        df_CP, df_RP, train_frac, rng
    )

    spike_scale = 3
    max_spikes = 5
    df_CPA_trn, df_RPA_trn = augment_training_set(
        df_CP_trn, df_RP_trn, rng, wvl_range, spike_scale, max_spikes
    )

    Xtrn, Ytrn, num_trn, num_wvl, num_classes = extract(df_RPA_trn)
    Xtst, Ytst, num_tst, num_wvl, num_classes = extract(df_RP_tst)

    input_shape = Xtrn.shape[1:]
    model = get_model(input_shape, num_classes)
    model.summary()

    lr0 = 1e-5
    loss = CategoricalCrossentropy()
    acc = CategoricalAccuracy(name="ca")
    f1 = F1Score(num_classes=num_classes, average="macro", name="f1")
    opt = Nadam(learning_rate=lr0)
    model.compile(loss=loss, optimizer=opt, metrics=[acc, f1])

    early = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=2,
        mode="min",
        restore_best_weights=True,
    )
    file_log = f"../data/snr_test/{noise_scale_i}_history.log"
    logger = callbacks.CSVLogger(file_log, append=False)
    cbs = [early, logger]

    history = model.fit(
        Xtrn,
        Ytrn,
        validation_data=(Xtst, Ytst),
        epochs=10_000,
        batch_size=32,
        verbose=2,
        callbacks=cbs,
    )

    loss_trn, ca_trn, f1_trn = model.evaluate(x=Xtrn, y=Ytrn, verbose=0)
    loss_tst, ca_tst, f1_tst = model.evaluate(x=Xtst, y=Ytst, verbose=0)

    results = f"{loss_trn},{ca_trn},{f1_trn},{loss_tst},{ca_tst},{f1_tst}\n"
    with open("../data/snr_test/results.csv", "a") as f:
        f.write(results)


def gen_noise(spectrum, noise_scale, rng):
    filt = savgol_filter(
        spectrum,
        11,
        1,
        mode="mirror",
    )
    res = spectrum - filt
    noise = res * noise_scale
    return noise


gen_noise = np.vectorize(gen_noise, signature="(n),(),()->(n)")


long_str = (
    "Hello take a look at this long ass string omg take a look at this "
    "string its so long. Hello take a look at this long ass string omg take a"
    " look at this string its so long."
)
print(long_str)

longer_string = """test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test test """
