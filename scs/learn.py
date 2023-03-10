from absl import app
from absl import flags

import sys
from os import mkdir
from os.path import join
from os.path import isdir
from shutil import rmtree

import numpy as np
from tensorflow import keras
from keras import callbacks
from keras import losses
from keras import metrics
from keras import optimizers
from keras import layers
from keras import regularizers
import tensorflow_addons as tfa

import data_loading as dl
import data_preparation as dp
import prepare_dataset
import lr_schedules
import hp_sets
import module_flags

FLAGS = flags.FLAGS


def main(argv):
    del argv

    R = FLAGS.R
    hp_set = FLAGS.hp_set

    # Check whether this function is being called as a batch job requeue. If it
    # isn't, then set `restart_fit` to `True` so that the directories and data
    # are made fresh.
    restart_fit = True if (FLAGS.num_requeue == "") else False
    print(f"`FLAGS.num_requeue`: {FLAGS.num_requeue}")
    print(f"`restart_fit`: {restart_fit}")

    # If the array_index is equal to or greater than the length of the
    # parameter grid, then the array_index is out of bounds so we can terminate
    # the program.
    array_index = int(FLAGS.array_index)
    PG = eval(f"hp_sets.{hp_set}()")
    if array_index >= len(PG):
        sys.exit(
            f"Array index was {array_index} but Parameter Grid is size"
            f" {len(PG)}."
        )
    hp = PG[int(array_index)]

    # Construct the directories if they don't exist or delete them and recreate
    # them if they do and `restart_fit` is `True`.
    dir_model = join(FLAGS.dir_models, f"{R}_{hp_set}_{array_index}")
    dir_backup = join(dir_model, "backup")
    dir_model_data = join(dir_model, "data")
    if isdir(dir_model) and restart_fit:
        rmtree(dir_model)
    mkdir(dir_model)
    mkdir(dir_backup)
    mkdir(dir_model_data)

    file_trn = join(dir_model_data, f"sn_data_trn.RPA.parquet")
    file_tst = join(dir_model_data, f"sn_data_tst.RP.parquet")

    train(
        R,
        dir_model,
        FLAGS.dir_raw,
        dir_model_data,
        file_trn,
        file_tst,
        hp,
        restart_fit=restart_fit,
        num_epochs=100000,
        batch_size=32,
        verbose=2,
    )


def train(
    R,
    dir_model,
    data_dir_original,
    dir_model_data,
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
        dir_model_data,
        dir_model_data,
        dir_model_data,
        dir_model_data,
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
        df_trn, df_tst
    )
    Xtrn, Ytrn, Xtst, Ytst = dataset

    # Generate the model to be trained.
    model = model_transformer(
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
        dropout_feed_forward=hp["dropout_feed_forward"],
    )

    lr_schedule = lr_schedules.get_lr_schedule(hp)
    callbacks = gen_callbacks(dir_model, lr_schedule)

    # Compile model with losses, metrics, and optimizer.
    loss = losses.CategoricalCrossentropy()
    acc = metrics.CategoricalAccuracy(name="ca")
    f1 = tfa.metrics.F1Score(
        num_classes=num_classes,
        average="macro",
        name="f1",
    )
    opt = optimizers.Nadam(learning_rate=hp["lr0"])
    model.compile(loss=loss, optimizer=opt, metrics=[acc, f1])
    fit = model.fit(
        Xtrn,
        Ytrn,
        validation_data=(Xtst, Ytst),
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )

    return fit


def prepare_datasets_for_training(df_trn, df_tst):
    data_trn = dp.extract_dataframe(df_trn)
    Xtrn = data_trn[6][..., None]
    Ytrn = data_trn[5]["SN Subtype ID"].to_numpy(dtype=int)

    num_wvl = Xtrn.shape[1]
    num_classes = np.unique(Ytrn).size

    data_tst = dp.extract_dataframe(df_tst)
    Xtst = data_tst[6][..., None]
    Ytst = data_tst[5]["SN Subtype ID"].to_numpy(dtype=int)

    Ytrn = keras.utils.to_categorical(Ytrn, num_classes=num_classes)
    Ytst = keras.utils.to_categorical(Ytst, num_classes=num_classes)

    return (Xtrn, Ytrn, Xtst, Ytst), num_wvl, num_classes


def gen_callbacks(dir_model, lr_schedule):
    checkpoint = callbacks.ModelCheckpoint(
        join(dir_model, "model.hdf5"),
        monitor="val_loss",
        verbose=0,
        mode="min",
        save_best_only=True,
    )

    early = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=25,
        verbose=0,
        mode="min",
        restore_best_weights=True,
    )

    logger = keras.callbacks.CSVLogger(
        join(dir_model, "history.log"),
        append=True,
    )

    backup_dir = join(dir_model, "backup")
    backup = callbacks.BackupAndRestore(backup_dir=backup_dir)

    schedule = callbacks.LearningRateScheduler(lr_schedule)

    return [checkpoint, early, logger, backup, schedule]


def model_transformer(
    input_shape,
    num_classes,
    num_transformer_blocks,
    num_heads,
    key_dim,
    kr_l2,
    br_l2,
    ar_l2,
    dropout_attention,
    dropout_projection,
    filters,
    num_feed_forward_layers,
    feed_forward_layer_size,
    dropout_feed_forward,
):
    inputs = keras.Input(shape=input_shape)

    x = inputs
    for _ in range(num_transformer_blocks):
        sublayer_start = x
        x = layers.LayerNormalization(epsilon=1e-6)(sublayer_start)
        x = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            kernel_regularizer=regularizers.L2(kr_l2),
            bias_regularizer=regularizers.L2(br_l2),
            activity_regularizer=regularizers.L2(ar_l2),
        )(x, x)
        x = layers.Dropout(dropout_attention)(x)
        sublayer_end = x + sublayer_start

        x = layers.LayerNormalization(epsilon=1e-6)(sublayer_end)
        x = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            activation="relu",
            kernel_regularizer=regularizers.L2(kr_l2),
            bias_regularizer=regularizers.L2(br_l2),
            activity_regularizer=regularizers.L2(ar_l2),
        )(x)
        x = layers.Dropout(dropout_projection)(x)
        x = layers.Conv1D(filters=input_shape[-1], kernel_size=1)(x)
        x = sublayer_end + x

    x = layers.GlobalMaxPooling1D(data_format="channels_first")(x)
    for _ in range(num_feed_forward_layers):
        x = layers.Dense(
            feed_forward_layer_size,
            activation="relu",
            kernel_regularizer=regularizers.L2(kr_l2),
            bias_regularizer=regularizers.L2(br_l2),
            activity_regularizer=regularizers.L2(ar_l2),
        )(x)
        x = layers.Dropout(dropout_feed_forward)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    return model


if __name__ == "__main__":
    app.run(main)
