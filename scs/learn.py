from os.path import join

from absl import app
from absl import flags

import numpy as np
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
import tensorflow_addons as tfa

import data_preparation as dp
import data_loading as dl
import prepare_dataset

import models
import lr_schedules


default_hyper_parameters = {
    "phase_range": (-20, 50),
    "ptp_range": (0.1, 100),
    "wvl_range": (4500, 7000),
    "train_frac": 0.50,
    "noise_scale": 0.25,
    "spike_scale": 3,
    "max_spikes": 5,
    "random_state": 1415,

    "lr0": 0.001,
    "lr_schedule": "constant_lr",

    "num_transformer_blocks": 6,
    "num_heads": 8,
    "key_dim": 64,
    "kr_l2": 0,
    "br_l2": 0,
    "ar_l2": 0,
    "dropout_attention": 0.1,
    "dropout_projection": 0.1,
    "filters": 4,
    "num_feed_forward_layers": 1,
    "feed_forward_layer_size": 1024,
    "dropout_feed_forward": 0.1,
}


def train(
    R,
    model_dir,
    data_dir_original,
    data_dir_degraded,
    data_dir_preprocessed,
    data_dir_train_test,
    data_dir_augmented,
    file_trn,
    file_tst,
    hp,
    restart_fit=True,
    num_epochs=100000,
    batch_size=32,
    verbose=0,
):
    # Prepare the dataset from the original dataset dataframe `sn_data_file`.
    sn_data_file = join(data_dir_original, "sn_data.parquet")
    prepare_dataset.prepare_dataset(
        R,
        sn_data_file,
        data_dir_degraded,
        data_dir_preprocessed,
        data_dir_train_test,
        data_dir_augmented,
        hp["phase_range"],
        hp["ptp_range"],
        hp["wvl_range"],
        hp["train_frac"],
        hp["noise_scale"],
        hp["spike_scale"],
        hp["max_spikes"],
        random_state=hp["random_state"],
        redo_preprocess=restart_fit,
        redo_split=restart_fit,
        redo_augment=restart_fit,
    )

    # Load the dataset.
    df_trn = dl.load_sn_data(file_trn)
    df_tst = dl.load_sn_data(file_tst)
    dataset, num_wvl, num_classes = prepare_datasets_for_training(
        df_trn, df_tst)
    Xtrn, Ytrn, Xtst, Ytst = dataset

    # Generate the model to be trained.
    model = models.model_transformer(
        input_shape=Xtrn.shape[1:],
        num_classes=num_classes,
        num_transformer_blocks=hp["num_transformer_blocks"],
        num_heads=hp["num_heads"],
        key_dim=hp["key_dim"],
        kr_l2=hp["kr_l2"],
        br_l2=hp["br_l2"],
        ar_l2=hp["ar_l2"],
        dropout_attention=hp["dropout_attention"],
        dropout_projection=hp["dropout_projection"],
        filters=hp["filters"],
        num_feed_forward_layers=hp["num_feed_forward_layers"],
        feed_forward_layer_size=hp["feed_forward_layer_size"],
        dropout_feed_forward=hp["dropout_feed_forward"])


    lr_schedule = lr_schedules.get_lr_schedule(hp)
    callbacks = gen_callbacks(model_dir, lr_schedule)

    # Compile model with losses, metrics, and optimizer.
    loss = losses.CategoricalCrossentropy()
    acc = metrics.CategoricalAccuracy(name="ca")
    f1 = tfa.metrics.F1Score(num_classes=num_classes, average="macro", name="f1")
    opt = optimizers.Nadam(learning_rate=hp["lr0"])
    model.compile(loss=loss, optimizer=opt, metrics=[acc, f1])
    fit = model.fit(
        Xtrn,
        Ytrn,
        validation_data=(Xtst, Ytst),
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks)

    return fit


def prepare_datasets_for_training(df_trn, df_tst):
    data_trn = dp.extract_dataframe(df_trn)
    Xtrn = data_trn[6][... , None]
    Ytrn = data_trn[5]["SN Subtype ID"].to_numpy(dtype=int)

    num_wvl = Xtrn.shape[1]
    num_classes = np.unique(Ytrn).size

    data_tst = dp.extract_dataframe(df_tst)
    Xtst = data_tst[6][..., None]
    Ytst = data_tst[5]["SN Subtype ID"].to_numpy(dtype=int)

    Ytrn = keras.utils.to_categorical(Ytrn, num_classes=num_classes)
    Ytst = keras.utils.to_categorical(Ytst, num_classes=num_classes)

    return (Xtrn, Ytrn, Xtst, Ytst), num_wvl, num_classes


def gen_callbacks(model_dir, lr_schedule):
    checkpoint = callbacks.ModelCheckpoint(
        join(model_dir, "model.hdf5"),
        monitor="val_loss",
        verbose=0,
        mode="min",
        save_best_only=True)

    early = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=25,
        verbose=0,
        mode="min",
        restore_best_weights=True)

    logger = keras.callbacks.CSVLogger(
        join(model_dir, "history.log"),
        append=True)

    backup_dir = join(model_dir, "backup")
    backup = callbacks.BackupAndRestore(
        backup_dir=backup_dir)

    schedule = callbacks.LearningRateScheduler(lr_schedule)

    return [checkpoint, early, logger, backup, schedule]


if __name__ == "__main__":
    app.run(main)