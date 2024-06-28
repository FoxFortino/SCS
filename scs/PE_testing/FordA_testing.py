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

sys.path.insert(0, "../../scs")

import scs_config
import data_degrading as dd
import data_preparation as dp
import data_augmentation as da
import data_plotting as dplt


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
    # wvl_mask = (wvl_range[0] < wvl) & (wvl < wvl_range[1])
    # return wvl_mask
    return np.ones(wvl.size).astype(bool)


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
    ic(data.shape)
    ic(nopad_data.shape)
    ic(wvl_mask.shape)
    
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


def make_model_results(model_history, R, Xtrn, Xtst, Ytrn, Ytst):
    ic()
    
#     # Loss Curve
#     fig_loss = dplt.plot_loss(model_history.history, scale=6)
    
#     # Confusion Matrix
#     Ytrn_flat = np.argmax(Ytrn, axis=1)
#     SNtypes_int = np.unique(Ytrn_flat)
#     SNtypes_str = [scs_config.SN_Stypes_int_to_str[sn] for sn in SNtypes_int]

#     Ptst = model_history.model.predict(Xtst, verbose=0)
#     Ptst_flat = np.argmax(Ptst, axis=1)
#     Ytst_flat = np.argmax(Ytst, axis=1)
#     CMtst = confusion_matrix(Ytst_flat, Ptst_flat)
#     fig_CMtst = dplt.plot_cm(CMtst, SNtypes_str, R)
    
#     Ptrn = model_history.model.predict(Xtrn)
#     Ptrn_flat = np.argmax(Ptrn, axis=1)
#     Ytrn_flat = np.argmax(Ytrn, axis=1)
#     CMtrn = confusion_matrix(Ytrn_flat, Ptrn_flat)
#     fig_CMtrn = dplt.plot_cm(CMtrn, SNtypes_str, R)
        
    # Final model evaluation
    loss_tst, ca_tst, f1_tst = model_history.model.evaluate(Xtst, Ytst, verbose=0)
    loss_trn, ca_trn, f1_trn = model_history.model.evaluate(Xtrn, Ytrn, verbose=0)
    
    metrics = {
        "trn_set_metrics": {"loss": loss_trn, "ca": ca_trn, "f1": f1_trn},
        "tst_set_metrics": {"loss": loss_tst, "ca": ca_tst, "f1": f1_tst},
    }
    # return fig_loss, fig_CMtst, fig_CMtrn, metrics
    return metrics


def save_model_metrics(model_dir, ext, metrics, history):
    ic()

#     fig_loss.savefig(join(model_dir, "loss_curve.pdf"))
#     fig_loss.savefig(join(model_dir, "loss_curve.png"))
    
#     fig_CMtst.savefig(join(model_dir, "CMtst.pdf"))
#     fig_CMtst.savefig(join(model_dir, "CMtst.png"))
    
#     fig_CMtrn.savefig(join(model_dir, "CMtrn.pdf"))
#     fig_CMtrn.savefig(join(model_dir, "CMtrn.png"))
    
    write_json(metrics, join(model_dir, f"metrics{ext}.json"))
    write_json(history, join(model_dir, f"history{ext}.json"))
    # write_json(hyperparameters, join(model_dir, f"hyperparameters{ext}.json"))
    return


def load_ford_data():
    def readucr(filename):
        data = np.loadtxt(filename, delimiter="\t")
        y = data[:, 0]
        x = data[:, 1:]
        return x, y.astype(int)


    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

    x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
    x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    n_classes = len(np.unique(y_train))

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    
    return x_train, x_test, y_train, y_test, n_classes


def feed_forward():
    Xtrn, Xtst, Ytrn, Ytst, num_classes = load_ford_data()
    ic(Xtrn.shape)
    ic(Ytrn.shape)
    ic(Xtst.shape)
    ic(Ytst.shape)
    ic(num_classes)
    
    Ytrn_OH = keras.utils.to_categorical(Ytrn, num_classes=2)
    Ytst_OH = keras.utils.to_categorical(Ytst, num_classes=2)
    ic(Ytrn_OH.shape)
    ic(Ytst_OH.shape)
    
    input_shape = ic(Xtrn.shape[1:])
    model = get_model(
        input_shape,
        num_classes=num_classes,
        encoder_blocks=0,
        encoder_heads=0,
        encoder_key_dim=0,
        encoder_proj_dim=0,
        encoder_dropout_attention=0,
        encoder_dropout_projection=0,
        feed_forward_dropout=0,
        feed_forward_units=[1024, 1024, 1024],
        feed_forward_activation="relu",
        encoder_kreg_l1_att=0,
        encoder_kreg_l2_att=0,
        encoder_breg_l1_att=0,
        encoder_breg_l2_att=0,
        encoder_areg_l1_att=0,
        encoder_areg_l2_att=0,
        encoder_kreg_l1_proj1=0,
        encoder_kreg_l2_proj1=0,
        encoder_breg_l1_proj1=0,
        encoder_breg_l2_proj1=0,
        encoder_areg_l1_proj1=0,
        encoder_areg_l2_proj1=0,
        encoder_kreg_l1_proj2=0,
        encoder_kreg_l2_proj2=0,
        encoder_breg_l1_proj2=0,
        encoder_breg_l2_proj2=0,
        encoder_areg_l1_proj2=0,
        encoder_areg_l2_proj2=0,
        feed_forward_kreg_l1=0,
        feed_forward_kreg_l2=0,
        feed_forward_breg_l1=0,
        feed_forward_breg_l2=0,
        feed_forward_areg_l1=0,
        feed_forward_areg_l2=0,
    )
    model.summary()
    
    loss = CategoricalCrossentropy()
    acc = CategoricalAccuracy(name="ca")
    f1 = F1Score(num_classes=num_classes, average="macro", name="f1")
    opt = Nadam(learning_rate=1e-5)
    model.compile(loss=loss, optimizer=opt, metrics=[acc, f1])


    cb_es = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=20,
        verbose=2,
        mode="min",
        restore_best_weights=True,
    )

    model_history = model.fit(
        Xtrn,
        Ytrn_OH,
        validation_data=(Xtst, Ytst_OH),
        epochs=10_000,
        batch_size=32,
        verbose=2,
        callbacks=[cb_es],
    )
    return


def transformer_noPE(encoder_blocks):
    Xtrn, Xtst, Ytrn, Ytst, num_classes = load_ford_data()
    ic(Xtrn.shape)
    ic(Ytrn.shape)
    ic(Xtst.shape)
    ic(Ytst.shape)
    ic(num_classes)
    
    Ytrn = keras.utils.to_categorical(Ytrn, num_classes=2)
    Ytst = keras.utils.to_categorical(Ytst, num_classes=2)
    ic(Ytrn.shape)
    ic(Ytst.shape)
    
    input_shape = ic(Xtrn.shape[1:])
    model = get_model(
        input_shape,
        num_classes=num_classes,
        encoder_blocks=encoder_blocks,
        encoder_heads=8,
        encoder_key_dim=64,
        encoder_proj_dim=1024,
        encoder_dropout_attention=0.1,
        encoder_dropout_projection=0.1,
        feed_forward_dropout=0.1,
        feed_forward_units=[1024, 1024, 1024],
        feed_forward_activation="relu",
        encoder_kreg_l1_att=0,
        encoder_kreg_l2_att=0,
        encoder_breg_l1_att=0,
        encoder_breg_l2_att=0,
        encoder_areg_l1_att=0,
        encoder_areg_l2_att=0,
        encoder_kreg_l1_proj1=0,
        encoder_kreg_l2_proj1=0,
        encoder_breg_l1_proj1=0,
        encoder_breg_l2_proj1=0,
        encoder_areg_l1_proj1=0,
        encoder_areg_l2_proj1=0,
        encoder_kreg_l1_proj2=0,
        encoder_kreg_l2_proj2=0,
        encoder_breg_l1_proj2=0,
        encoder_breg_l2_proj2=0,
        encoder_areg_l1_proj2=0,
        encoder_areg_l2_proj2=0,
        feed_forward_kreg_l1=0,
        feed_forward_kreg_l2=0,
        feed_forward_breg_l1=0,
        feed_forward_breg_l2=0,
        feed_forward_areg_l1=0,
        feed_forward_areg_l2=0,
    )
    model.summary()
    
    loss = CategoricalCrossentropy()
    acc = CategoricalAccuracy(name="ca")
    f1 = F1Score(num_classes=num_classes, average="macro", name="f1")
    opt = Nadam(learning_rate=1e-5)
    model.compile(loss=loss, optimizer=opt, metrics=[acc, f1])


    cb_es = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=20,
        verbose=2,
        mode="min",
        restore_best_weights=True,
    )

    model_history = model.fit(
        Xtrn,
        Ytrn,
        validation_data=(Xtst, Ytst),
        epochs=10_000,
        batch_size=8,
        verbose=2,
        callbacks=[cb_es],
    )
    
    num_epochs = len(model_history.history["loss"])
    model_history.history["epoch"] = np.arange(num_epochs, dtype=int) + 1
    model_history.history["epoch"] = model_history.history["epoch"].tolist()
    
    metrics = make_model_results(model_history, 100, Xtrn, Xtst, Ytrn, Ytst)
    save_model_metrics(
        ".",
        f"_transformer_noPE_{encoder_blocks}",
        metrics,
        model_history.history,
    )

    
    return


def transformer_with_embedding(encoder_blocks, num_quantiles):
    Xtrn, Xtst, Ytrn, Ytst, num_classes = load_ford_data()
    Xtrn = Xtrn[..., 0]
    Xtst = Xtst[..., 0]
    ic(Xtrn.shape)
    ic(Ytrn.shape)
    ic(Xtst.shape)
    ic(Ytst.shape)
    ic(num_classes)
    
    Ytrn = keras.utils.to_categorical(Ytrn, num_classes=2)
    Ytst = keras.utils.to_categorical(Ytst, num_classes=2)
    ic(Ytrn.shape)
    ic(Ytst.shape)
    
    quantile_probabilities = get_quantile_probability(num_quantiles, 0.01, 0.99)
    wvl_mask = get_wvl_mask(None, Xtrn[0, ...])
    nopad_bins = get_quantiles(Xtrn, wvl_mask, quantile_probabilities)
    ic(quantile_probabilities)
    ic(wvl_mask.shape)
    ic(nopad_bins.shape)

    Xtrn_digitized = digitize_spectra(Xtrn, wvl_mask, nopad_bins)
    Xtst_digitized = digitize_spectra(Xtst, wvl_mask, nopad_bins)
    ic(Xtrn_digitized.shape)
    ic(Xtst_digitized.shape)

    Xtrn_embedded = onehot_digitized_spectra(Xtrn, Xtrn_digitized, num_quantiles)
    Xtst_embedded = onehot_digitized_spectra(Xtst, Xtst_digitized, num_quantiles)
    ic(Xtrn_embedded.shape)
    ic(Xtst_embedded.shape)

    Xtrn = Xtrn_embedded
    Xtst = Xtst_embedded

    input_shape = ic(Xtrn.shape[1:])
    model = get_model(
        input_shape,
        num_classes=num_classes,
        encoder_blocks=encoder_blocks,
        encoder_heads=4,
        encoder_key_dim=64,
        encoder_proj_dim=1024,
        encoder_dropout_attention=0,
        encoder_dropout_projection=0,
        feed_forward_dropout=0,
        feed_forward_units=[128, 128, 128],
        feed_forward_activation="relu",
        encoder_kreg_l1_att=0,
        encoder_kreg_l2_att=0,
        encoder_breg_l1_att=0,
        encoder_breg_l2_att=0,
        encoder_areg_l1_att=0,
        encoder_areg_l2_att=0,
        encoder_kreg_l1_proj1=0,
        encoder_kreg_l2_proj1=0,
        encoder_breg_l1_proj1=0,
        encoder_breg_l2_proj1=0,
        encoder_areg_l1_proj1=0,
        encoder_areg_l2_proj1=0,
        encoder_kreg_l1_proj2=0,
        encoder_kreg_l2_proj2=0,
        encoder_breg_l1_proj2=0,
        encoder_breg_l2_proj2=0,
        encoder_areg_l1_proj2=0,
        encoder_areg_l2_proj2=0,
        feed_forward_kreg_l1=0,
        feed_forward_kreg_l2=0,
        feed_forward_breg_l1=0,
        feed_forward_breg_l2=0,
        feed_forward_areg_l1=0,
        feed_forward_areg_l2=0,
    )
    model.summary()
    
    loss = CategoricalCrossentropy()
    acc = CategoricalAccuracy(name="ca")
    f1 = F1Score(num_classes=num_classes, average="macro", name="f1")
    opt = Nadam(learning_rate=1e-5)
    model.compile(loss=loss, optimizer=opt, metrics=[acc, f1])

    cb_es = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=20,
        verbose=2,
        mode="min",
        restore_best_weights=True,
    )

    model_history = model.fit(
        Xtrn,
        Ytrn,
        validation_data=(Xtst, Ytst),
        epochs=10_000,
        batch_size=8,
        verbose=2,
        callbacks=[cb_es],
    )
    
    num_epochs = len(model_history.history["loss"])
    model_history.history["epoch"] = np.arange(num_epochs, dtype=int) + 1
    model_history.history["epoch"] = model_history.history["epoch"].tolist()
    
    metrics = make_model_results(model_history, 100, Xtrn, Xtst, Ytrn, Ytst)
    save_model_metrics(
        ".",
        f"_transformer_with_embedding_{encoder_blocks}",
        metrics,
        model_history.history,
    )
    return


if __name__ == "__main__":
    # feed_forward()
    
    # for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    #     transformer_noPE(i)
    
    # Now try transformer with quantile embedding
    transformer_with_embedding(2, 19)
    
    # Now try transformer with quantile embedding and positional encoding
    # transformer_with_embedding_and_PE(4)
    
    # Now that I am not using a GLobalMaxPoolingLayer to go from the eoncder blocks to the feedforward classification head, maybe I can reshape the data and use a Convolutional layers? For no good reason but could try.
