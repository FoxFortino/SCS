import sys
import json
import argparse
from copy import deepcopy
from os import mkdir
from os.path import join
from os.path import isfile
from os.path import isdir
from os.path import abspath
from os.path import basename
from icecream import ic
from glob import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid

from tensorflow import keras
from keras import layers
from keras import callbacks
from keras.regularizers import L1L2
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Nadam
from keras.utils import to_categorical
from tensorflow_addons.metrics import F1Score

import scs_config
import data_degrading as dd
import data_preparation as dp
import data_augmentation as da
import data_plotting as dplt


MAX_JOBS = 300
RESULTS_FILES = [
    "loss_curve.pdf",
    "loss_curve.png",
    "CMtrn.pdf",
    "CMtrn.png",
    "CMtst.pdf",
    "CMtst.png",
    "metrics.json",
    "history.json",
    "hyperparameters.json",
]


def load_original_dataset():
    ic()
    file_df_raw = "/home/2649/repos/SCS/data/raw/sn_data.parquet"
    df_raw = pd.read_parquet(file_df_raw)
    return df_raw


def load_R100_data():
    ic()
    file_df_R = "/home/2649/repos/SCS/data/R100/df_R.parquet"
    file_df_C = "/home/2649/repos/SCS/data/R100/df_C.parquet"
    df_R = pd.read_parquet(file_df_R)
    df_C = pd.read_parquet(file_df_C)
    return df_C, df_R


def degrade_data(df_raw, R):
    ic()
    df_C, df_R = dd.degrade_dataframe(R, df_raw)
    return df_C, df_R


def clean_data(df_C, df_R, phase_range, ptp_range, wvl_range):
    ic()
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
    ic()
    df_CP_trn, df_CP_tst = dp.split_data(df_CP, train_frac, rng)
    df_RP_trn, df_RP_tst = dp.split_data(df_RP, train_frac, rng)
    return df_CP_trn, df_CP_tst, df_RP_trn, df_RP_tst


def augment_training_set(df_CP_trn, df_RP_trn, rng, wvl_range, noise_scale, spike_scale, max_spikes):
    ic()
    df_CPA_trn = da.augment(
        df_CP_trn,
        rng,
        wvl_range=wvl_range,
        noise_scale=noise_scale,
        spike_scale=spike_scale,
        max_spikes=max_spikes,
    )
    df_RPA_trn = da.augment(
        df_RP_trn,
        rng,
        wvl_range=wvl_range,
        noise_scale=noise_scale,
        spike_scale=spike_scale,
        max_spikes=max_spikes,
    )
    return df_CPA_trn, df_RPA_trn


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


def extract(df):
    data = extract_dataframe(df)
    X = data[6]
    Y = data[5]["SN Subtype ID"].to_numpy(dtype=int)
    
    N = X.shape[0]
    num_wvl = X.shape[1]
    num_classes = np.unique(Y).size
    wvl = data[1]
    
    Y_OH = to_categorical(Y, num_classes=num_classes)

    return X, Y_OH, N, num_wvl, num_classes, wvl


def get_model(
    input_shape,
    num_classes,
    encoder_blocks,
    encoder_heads,
    encoder_key_dim,
    encoder_proj_dim,
    encoder_dropout_attention,
    encoder_dropout_projection,
    feed_forward_dropout,
    feed_forward_units,
    feed_forward_activation,
    encoder_kreg_l1_att,
    encoder_kreg_l2_att,
    encoder_breg_l1_att,
    encoder_breg_l2_att,
    encoder_areg_l1_att,
    encoder_areg_l2_att,
    encoder_kreg_l1_proj1,
    encoder_kreg_l2_proj1,
    encoder_breg_l1_proj1,
    encoder_breg_l2_proj1,
    encoder_areg_l1_proj1,
    encoder_areg_l2_proj1,
    encoder_kreg_l1_proj2,
    encoder_kreg_l2_proj2,
    encoder_breg_l1_proj2,
    encoder_breg_l2_proj2,
    encoder_areg_l1_proj2,
    encoder_areg_l2_proj2,
    feed_forward_kreg_l1,
    feed_forward_kreg_l2,
    feed_forward_breg_l1,
    feed_forward_breg_l2,
    feed_forward_areg_l1,
    feed_forward_areg_l2,
):
    ic()
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for _ in range(encoder_blocks):
        x = transformer_block(
            x,
            input_shape,
            encoder_heads,
            encoder_key_dim,
            encoder_proj_dim,
            encoder_dropout_attention,
            encoder_dropout_projection,
            encoder_kreg_l1_att,
            encoder_kreg_l2_att,
            encoder_breg_l1_att,
            encoder_breg_l2_att,
            encoder_areg_l1_att,
            encoder_areg_l2_att,
            encoder_kreg_l1_proj1,
            encoder_kreg_l2_proj1,
            encoder_breg_l1_proj1,
            encoder_breg_l2_proj1,
            encoder_areg_l1_proj1,
            encoder_areg_l2_proj1,
            encoder_kreg_l1_proj2,
            encoder_kreg_l2_proj2,
            encoder_breg_l1_proj2,
            encoder_breg_l2_proj2,
            encoder_areg_l1_proj2,
            encoder_areg_l2_proj2,
        )
    # Perhaps all this time to GlobalMaxPooling layer has been completely destroying all of the relevant information learned in the preceding layers... Trying now just a reshape.
    x = layers.Reshape((np.prod(x.shape[1:]),))(x)
    # x = layers.GlobalMaxPooling1D(data_format="channels_first")(x)

    for n in feed_forward_units:
        x = layers.Dense(
            n,
            activation=feed_forward_activation,
            kernel_regularizer=L1L2(
                l1=feed_forward_kreg_l1, 
                l2=feed_forward_kreg_l2,
            ),
            bias_regularizer=L1L2(
                l1=feed_forward_breg_l1, 
                l2=feed_forward_breg_l2,
            ),
            activity_regularizer=L1L2(
                l1=feed_forward_areg_l1, 
                l2=feed_forward_areg_l2,
            ),
        )(x)
        x = layers.Dropout(feed_forward_dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    return model


def transformer_block(
    x,
    input_shape,
    heads,
    key_dim,
    encoder_proj_dim,
    dropout_attention,
    dropout_projection,
    encoder_kreg_l1_att,
    encoder_kreg_l2_att,
    encoder_breg_l1_att,
    encoder_breg_l2_att,
    encoder_areg_l1_att,
    encoder_areg_l2_att,
    encoder_kreg_l1_proj1,
    encoder_kreg_l2_proj1,
    encoder_breg_l1_proj1,
    encoder_breg_l2_proj1,
    encoder_areg_l1_proj1,
    encoder_areg_l2_proj1,
    encoder_kreg_l1_proj2,
    encoder_kreg_l2_proj2,
    encoder_breg_l1_proj2,
    encoder_breg_l2_proj2,
    encoder_areg_l1_proj2,
    encoder_areg_l2_proj2,
):
    x0 = layers.MultiHeadAttention(
        num_heads=heads,
        key_dim=key_dim,
        kernel_regularizer=L1L2(
            l1=encoder_kreg_l1_att,
            l2=encoder_kreg_l2_att,
        ),
        bias_regularizer=L1L2(
            l1=encoder_breg_l1_att,
            l2=encoder_breg_l2_att,
        ),
        activity_regularizer=L1L2(
            l1=encoder_areg_l1_att,
            l2=encoder_areg_l2_att,
        ),
    )(x, x)
    x0 = layers.Dropout(dropout_attention)(x0)
    x0 = layers.Add()([x, x0])
    x0 = layers.LayerNormalization()(x0)

    x1 = layers.Conv1D(
        filters=encoder_proj_dim,
        kernel_size=1,
        activation="relu",
        kernel_regularizer=L1L2(
            l1=encoder_kreg_l1_proj1,
            l2=encoder_kreg_l2_proj1,
        ),
        bias_regularizer=L1L2(
            l1=encoder_breg_l1_proj1,
            l2=encoder_breg_l2_proj1,
        ),
        activity_regularizer=L1L2(
            l1=encoder_areg_l1_proj1,
            l2=encoder_areg_l2_proj1,
        ),
    )(x0)

    x1 = layers.Conv1D(
        filters=input_shape[1],
        kernel_size=1,
        activation="relu",
        kernel_regularizer=L1L2(
            l1=encoder_kreg_l1_proj2,
            l2=encoder_kreg_l2_proj2,
        ),
        bias_regularizer=L1L2(
            l1=encoder_breg_l1_proj2,
            l2=encoder_breg_l2_proj2,
        ),
        activity_regularizer=L1L2(
            l1=encoder_areg_l1_proj2,
            l2=encoder_areg_l2_proj2,
        ),
    )(x1)
    x1 = layers.Dropout(dropout_projection)(x1)
    x1 = layers.Add()([x0 + x1])
    x1 = layers.LayerNormalization()(x1)

    return x1


def get_callbacks(model_dir):
    ic()
    cb_es = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=50,
        verbose=2,
        mode="min",
        restore_best_weights=True,
    )
    
    cb_mc = callbacks.ModelCheckpoint(
        join(model_dir, "model.hdf5"),
        monitor="val_loss",
        verbose=2,
        mode="min",
        save_best_only=True,
    )
    
    cb_rlrp = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.75,
        patience=25,
        verbose=2,
        mode="min",
        min_delta=0,
        cooldown=50,
        min_lr=1e-7,
    )
    return [cb_es, cb_mc, cb_rlrp]


def get_quantile_probability(num_quantiles, low_outlier, high_outlier):
    ic()
    quantiles = np.linspace(low_outlier, high_outlier, num_quantiles)
    return quantiles


def get_wvl_mask(wvl_range, wvl):
    ic()
    wvl_mask = (wvl_range[0] < wvl) & (wvl < wvl_range[1])
    return wvl_mask


def get_quantiles(data, wvl_mask, quantile_probabilities):
    ic()
    data = deepcopy(data)
    nopad_data = data[:, wvl_mask]
    nopad_bins = np.quantile(nopad_data, quantile_probabilities, axis=0)
    return nopad_bins


def digitize_spectra(data, wvl_mask, nopad_bins):
    ic()
    data = deepcopy(data)
    nopad_data = data[:, wvl_mask]
    
    nopad_data_digitized = []
    for i in range(wvl_mask.sum()):
        wvl_digitized = np.digitize(nopad_data[:, i], nopad_bins[:, i])
        nopad_data_digitized.append(wvl_digitized)
    nopad_data_digitized = np.vstack(nopad_data_digitized).T
    
    data_digitized = np.zeros_like(data)    
    data_digitized[:, wvl_mask] = nopad_data_digitized
    return data_digitized


def onehot_digitized_spectra(data, data_digitized, num_quantiles):
    ic()
    data_embedded = []
    
    # Loop through each spectrum
    for i in range(data_digitized.shape[0]):
        
        # Convert the spectrum from a regular 
        digitized_spectrum = data_digitized[i]
        digitized_spectrum = pd.Series(digitized_spectrum, dtype=pd.CategoricalDtype(categories=np.arange(num_quantiles)))        
        digitized_spectrum_onehot = pd.get_dummies(digitized_spectrum).values
        digitized_spectrum_onehot = np.expand_dims(digitized_spectrum_onehot, 0)

        spectrum = data[i]
        spectrum = np.expand_dims(spectrum, (0, 2))

        spectrum_complete = np.concatenate((spectrum, digitized_spectrum_onehot), axis=2)
        
        data_embedded.append(spectrum_complete)

    data_embedded = np.concatenate(data_embedded, axis=0)
    return data_embedded


def get_positional_encoding(num_wvl, num_embed, period):
    ic()
    PE = np.zeros((num_wvl, num_embed))
    for pos in range(num_wvl):
        for i in range(num_embed):
            if i % 2 == 0:
                value = np.sin(pos / np.power(period, 2 * i / num_embed))
            else:
                value = np.cos(pos / np.power(period, 2 * i / num_embed))
            PE[pos, i] = value
    return PE


def add_positional_encoding(data, PE):
    ic()
    assert data.ndim == 3 or data.ndim == 2
    if data.ndim == 3:
        assert data.shape[1] == PE.shape[0]
        assert data.shape[2] == PE.shape[1]
    elif data.ndim == 2:
        assert data.shape[0] == PE.shape[0]
        assert data.shape[1] == PE.shape[1]
    return data + PE


def write_json(stuff, file):
    ic()
    dumps = json.dumps(stuff, indent=4, sort_keys=True)
    with open(file, mode="w") as f:
        f.write(dumps)
    return


def make_model_results(model_history, R, Xtrn, Xtst, Ytrn, Ytst, model_dir):
    ic()
    
    # Loss Curve
    fig_loss = dplt.plot_loss(model_history.history, scale=6)
    
    # Confusion Matrix
    Ytrn_flat = np.argmax(Ytrn, axis=1)
    SNtypes_int = np.unique(Ytrn_flat)
    SNtypes_str = [scs_config.SN_Stypes_int_to_str[sn] for sn in SNtypes_int]

    Ptst = model_history.model.predict(Xtst, verbose=0)
    Ptst_flat = np.argmax(Ptst, axis=1)
    Ytst_flat = np.argmax(Ytst, axis=1)
    CMtst = confusion_matrix(Ytst_flat, Ptst_flat)
    fig_CMtst = dplt.plot_cm(CMtst, SNtypes_str, R)
    
    Ptrn = model_history.model.predict(Xtrn)
    Ptrn_flat = np.argmax(Ptrn, axis=1)
    Ytrn_flat = np.argmax(Ytrn, axis=1)
    CMtrn = confusion_matrix(Ytrn_flat, Ptrn_flat)
    fig_CMtrn = dplt.plot_cm(CMtrn, SNtypes_str, R)
        
    # Final model evaluation
    loss_tst, ca_tst, f1_tst = model_history.model.evaluate(Xtst, Ytst, verbose=0)
    loss_trn, ca_trn, f1_trn = model_history.model.evaluate(Xtrn, Ytrn, verbose=0)
    
    metrics = {
        "trn_set_metrics": {"loss": loss_trn, "ca": ca_trn, "f1": f1_trn},
        "tst_set_metrics": {"loss": loss_tst, "ca": ca_tst, "f1": f1_tst},
    }
    return fig_loss, fig_CMtst, fig_CMtrn, metrics


def save_model_metrics(model_dir, fig_loss, fig_CMtst, fig_CMtrn, metrics, history, hyperparameters):
    ic()

    fig_loss.savefig(join(model_dir, "loss_curve.pdf"))
    fig_loss.savefig(join(model_dir, "loss_curve.png"))
    
    fig_CMtst.savefig(join(model_dir, "CMtst.pdf"))
    fig_CMtst.savefig(join(model_dir, "CMtst.png"))
    
    fig_CMtrn.savefig(join(model_dir, "CMtrn.pdf"))
    fig_CMtrn.savefig(join(model_dir, "CMtrn.png"))
    
    write_json(metrics, join(model_dir, "metrics.json"))
    write_json(history, join(model_dir, "history.json"))
    write_json(hyperparameters, join(model_dir, "hyperparameters.json"))
    return


def main(
    R,
    model_dir,
    phase_range_start,
    phase_range_end,
    ptp_range_start,
    ptp_range_end,
    wvl_range_start,
    wvl_range_end,
    train_frac,
    noise_scale,
    spike_scale,
    max_spikes,
    num_quantiles,
    low_outlier,
    high_outlier,
    PE_period,
    lr0,
    encoder_blocks,
    encoder_heads,
    encoder_key_dim,
    encoder_proj_dim,
    encoder_dropout_attention,
    encoder_dropout_projection,
    feed_forward_dropout,
    feed_forward_units,
    feed_forward_activation,
    encoder_kreg_l1_att,
    encoder_kreg_l2_att,
    encoder_breg_l1_att,
    encoder_breg_l2_att,
    encoder_areg_l1_att,
    encoder_areg_l2_att,
    encoder_kreg_l1_proj1,
    encoder_kreg_l2_proj1,
    encoder_breg_l1_proj1,
    encoder_breg_l2_proj1,
    encoder_areg_l1_proj1,
    encoder_areg_l2_proj1,
    encoder_kreg_l1_proj2,
    encoder_kreg_l2_proj2,
    encoder_breg_l1_proj2,
    encoder_breg_l2_proj2,
    encoder_areg_l1_proj2,
    encoder_areg_l2_proj2,
    feed_forward_kreg_l1,
    feed_forward_kreg_l2,
    feed_forward_breg_l1,
    feed_forward_breg_l2,
    feed_forward_areg_l1,
    feed_forward_areg_l2,
    epochs,
    batch_size,
    hp={},
    show_figs=False,
):  
    rng = np.random.RandomState(1415)

    df_C, df_R = load_R100_data()

    phase_range = (phase_range_start, phase_range_end)
    ptp_range = (ptp_range_start, ptp_range_end)
    wvl_range = (wvl_range_start, wvl_range_end)
    df_CP, df_RP = clean_data(df_C, df_R, phase_range, ptp_range, wvl_range)

    df_CP_trn, df_CP_tst, df_RP_trn, df_RP_tst = split_train_test(
        df_CP, df_RP, train_frac, rng
    )

    df_CPA_trn, df_RPA_trn = augment_training_set(
        df_CP_trn, df_RP_trn, rng, wvl_range, noise_scale, spike_scale, max_spikes
    )

    Xtrn, Ytrn, num_trn, num_wvl, num_classes, wvl = extract(df_RPA_trn)
    Xtst, Ytst, num_tst, num_wvl, num_classes, wvl = extract(df_RP_tst)

    quantile_probabilities = get_quantile_probability(num_quantiles, low_outlier, high_outlier)
    wvl_mask = get_wvl_mask(wvl_range, wvl)
    nopad_bins = get_quantiles(Xtrn, wvl_mask, quantile_probabilities)

    Xtrn_digitized = digitize_spectra(Xtrn, wvl_mask, nopad_bins)
    Xtst_digitized = digitize_spectra(Xtst, wvl_mask, nopad_bins)

    Xtrn_embedded = onehot_digitized_spectra(Xtrn, Xtrn_digitized, num_quantiles)
    Xtst_embedded = onehot_digitized_spectra(Xtst, Xtst_digitized, num_quantiles)

    num_embed = num_quantiles + 1
    PE = get_positional_encoding(num_wvl, num_embed, PE_period)

    Xtrnf = add_positional_encoding(Xtrn_embedded, PE)
    Xtstf = add_positional_encoding(Xtst_embedded, PE)

    input_shape = Xtrnf.shape[1:]
    model = get_model(
        input_shape,
        num_classes,
        encoder_blocks,
        encoder_heads,
        encoder_key_dim,
        encoder_proj_dim,
        encoder_dropout_attention,
        encoder_dropout_projection,
        feed_forward_dropout,
        feed_forward_units,
        feed_forward_activation,
        encoder_kreg_l1_att,
        encoder_kreg_l2_att,
        encoder_breg_l1_att,
        encoder_breg_l2_att,
        encoder_areg_l1_att,
        encoder_areg_l2_att,
        encoder_kreg_l1_proj1,
        encoder_kreg_l2_proj1,
        encoder_breg_l1_proj1,
        encoder_breg_l2_proj1,
        encoder_areg_l1_proj1,
        encoder_areg_l2_proj1,
        encoder_kreg_l1_proj2,
        encoder_kreg_l2_proj2,
        encoder_breg_l1_proj2,
        encoder_breg_l2_proj2,
        encoder_areg_l1_proj2,
        encoder_areg_l2_proj2,
        feed_forward_kreg_l1,
        feed_forward_kreg_l2,
        feed_forward_breg_l1,
        feed_forward_breg_l2,
        feed_forward_areg_l1,
        feed_forward_areg_l2,
    )
    model.summary()

    loss = CategoricalCrossentropy()
    acc = CategoricalAccuracy(name="ca")
    f1 = F1Score(num_classes=num_classes, average="macro", name="f1")
    opt = Nadam(learning_rate=lr0)
    model.compile(loss=loss, optimizer=opt, metrics=[acc, f1])

    cbs = get_callbacks(model_dir)

    model_history = model.fit(
        Xtrnf,
        Ytrn,
        validation_data=(Xtstf, Ytst),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=cbs,
    )
    num_epochs = len(model_history.history["loss"])
    model_history.history["epoch"] = np.arange(num_epochs, dtype=int) + 1
    model_history.history["epoch"] = model_history.history["epoch"].tolist()
    model_history.history["lr"] = np.array(model_history.history["lr"]).tolist()
    
    fig_loss, fig_CMtst, fig_CMtrn, metrics = make_model_results(
        model_history, R, Xtrnf, Xtstf, Ytrn, Ytst, model_dir,
    )
    save_model_metrics(
        model_dir, fig_loss, fig_CMtst, fig_CMtrn, metrics, model_history.history, hp,
    )
    
    if show_figs:
        fig_loss.show()
        fig_CMtst.show()
        fig_CMtrn.show()
    
    return


def generate_param_grid():
    hp = {
        "phase_range_start": -20,
        "phase_range_end": 50,
        "ptp_range_start": 0.1,
        "ptp_range_end": 100,
        "wvl_range_start": 4500,
        "wvl_range_end": 7000,

        "train_frac": 0.50,
        "noise_scale": 0.1,
        "spike_scale": 1.0,
        "max_spikes": 3,

        "num_quantiles": 9,
        "low_outlier": 0.01,
        "high_outlier": 0.99,
        "PE_period": 10_000,

        "lr0": 1e-6,
        
        "encoder_blocks": 1,
        "encoder_heads": 8,
        "encoder_key_dim": 8,
        "encoder_proj_dim": 2048,
        "encoder_dropout_attention": 0,
        "encoder_dropout_projection": 0,
        "feed_forward_dropout": 0,
        "feed_forward_units": [128],
        "feed_forward_activation": "relu",
        "encoder_kreg_l1_att": 0,
        "encoder_kreg_l2_att": 0,
        "encoder_breg_l1_att": 0,
        "encoder_breg_l2_att": 0,
        "encoder_areg_l1_att": 0,
        "encoder_areg_l2_att": 0,
        "encoder_kreg_l1_proj1": 0,
        "encoder_kreg_l2_proj1": 0,
        "encoder_breg_l1_proj1": 0,
        "encoder_breg_l2_proj1": 0,
        "encoder_areg_l1_proj1": 0,
        "encoder_areg_l2_proj1": 0,
        "encoder_kreg_l1_proj2": 0,
        "encoder_kreg_l2_proj2": 0,
        "encoder_breg_l1_proj2": 0,
        "encoder_breg_l2_proj2": 0,
        "encoder_areg_l1_proj2": 0,
        "encoder_areg_l2_proj2": 0,
        "feed_forward_kreg_l1": 0,
        "feed_forward_kreg_l2": 0,
        "feed_forward_breg_l1": 0,
        "feed_forward_breg_l2": 0,
        "feed_forward_areg_l1": 0,
        "feed_forward_areg_l2": 0,

        "epochs": 10_000,
        "batch_size": 8,
    }
    
    hp = {key: [val] for key, val in hp.items()}
    
    hp["encoder_blocks"] = [1, 2, 3, 4, 5, 6, 7, 8]
    hp["feed_forward_units"] = [[128], [1024, 1024, 1024], [1024, 256]]
    hp["feed_forward_activation"] = ["relu", "sigmoid", "tanh", "elu", "leaky_relu", "relu6", "linear", "silu"]
    
    
    return ParameterGrid(hp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, required=True)
    parser.add_argument("--dir_batch_model", required=True)
    parser.add_argument("--SLURM_RESTART_COUNT", required=True)
    parser.add_argument("--SLURM_ARRAY_TASK_ID", type=int, required=True)
    args = parser.parse_args()
    
    ic(args.SLURM_RESTART_COUNT)
    
    ic(args.dir_batch_model)
    assert isdir(args.dir_batch_model)

    PG = generate_param_grid()
    ic(len(PG))
    i = ic(args.SLURM_ARRAY_TASK_ID)
    while True:
        try:
            hp = ic(PG[i])
        except IndexError:
            sys.exit(0)

        # Check if the set of hyperparameters, hp, has been trained on yet by checking
        # if the RESULTS_FILES exist. If they do exist, then we continue the for-loop
        # and move onto the next set of hyperparameters to train. If they don't exist
        # then main() get's called.
        dir_model = ic(join(args.dir_batch_model, f"model_{i:0>5}"))
        if ic(isdir(dir_model)):
            dir_files = glob(join(dir_model, "*"))
            dir_files_relative = [basename(f) for f in dir_files]
            if ic(set(RESULTS_FILES).issubset(dir_files_relative)):
                i += MAX_JOBS
                continue
        else:
            ic(mkdir(dir_model))

        main(
            args.R,
            dir_model,
            hp["phase_range_start"],
            hp["phase_range_end"],
            hp["ptp_range_start"],
            hp["ptp_range_end"],
            hp["wvl_range_start"],
            hp["wvl_range_end"],
            hp["train_frac"],
            hp["noise_scale"],
            hp["spike_scale"],
            hp["max_spikes"],
            hp["num_quantiles"],
            hp["low_outlier"],
            hp["high_outlier"],
            hp["PE_period"],
            hp["lr0"],
            hp["encoder_blocks"],
            hp["encoder_heads"],
            hp["encoder_key_dim"],
            hp["encoder_proj_dim"],
            hp["encoder_dropout_attention"],
            hp["encoder_dropout_projection"],
            hp["feed_forward_dropout"],
            hp["feed_forward_units"],
            hp["feed_forward_activation"],
            hp["encoder_kreg_l1_att"],
            hp["encoder_kreg_l2_att"],
            hp["encoder_breg_l1_att"],
            hp["encoder_breg_l2_att"],
            hp["encoder_areg_l1_att"],
            hp["encoder_areg_l2_att"],
            hp["encoder_kreg_l1_proj1"],
            hp["encoder_kreg_l2_proj1"],
            hp["encoder_breg_l1_proj1"],
            hp["encoder_breg_l2_proj1"],
            hp["encoder_areg_l1_proj1"],
            hp["encoder_areg_l2_proj1"],
            hp["encoder_kreg_l1_proj2"],
            hp["encoder_kreg_l2_proj2"],
            hp["encoder_breg_l1_proj2"],
            hp["encoder_breg_l2_proj2"],
            hp["encoder_areg_l1_proj2"],
            hp["encoder_areg_l2_proj2"],
            hp["feed_forward_kreg_l1"],
            hp["feed_forward_kreg_l2"],
            hp["feed_forward_breg_l1"],
            hp["feed_forward_breg_l2"],
            hp["feed_forward_areg_l1"],
            hp["feed_forward_areg_l2"],
            hp["epochs"],
            hp["batch_size"],
            hp=hp,
            show_figs=False,
        )
        i += MAX_JOBS
