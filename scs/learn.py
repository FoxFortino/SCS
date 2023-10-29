from absl import app
from absl import flags

# Python standard modules
import sys
import json
from os import mkdir
from os import listdir
from os.path import join
from os.path import isdir
from os.path import isfile
from os.path import abspath

# Community Packages
import numpy as np
import pandas as pd
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
from tensorflow_addons.metrics import F1Score

# My packages
sys.path.insert(0, "../scs/")
from prepare_R_data import prepare_R_data
from data_loading import load_sn_data
from prepare_datasets_for_training import extract
from prepare_datasets_for_training import add_dim
from lr_schedules import get_lr_schedule
import data_plotting as dplt

sys.path.insert(0, "../scs/models")
import feed_forward
import transformer_encoder


def run_SCS(dir_model, R, hp, file_raw_data=None, resume=False):
    dir_model_backup = join(dir_model, "backup")
    dir_model_data = join(dir_model, "data")
    dir_model_results = join(dir_model, "results")
    assert isdir(dir_model), f"`{dir_model}` does not exist."
    assert isdir(dir_model_backup), f"`{dir_model_backup}` does not exist."
    assert isdir(dir_model_data), f"`{dir_model_data}` does not exist."
    assert isdir(dir_model_results), f"`{dir_model_results}` does not exist."

    file_model = join(dir_model, "model.hdf5")
    file_model_history = join(dir_model, "history.log")
    file_model_results = join(dir_model_results, "results.json")
    file_model_hp = join(dir_model, "hp.json")
    file_model_curves = join(dir_model_results, "curves.pdf")
    
    if isfile(file_model_results):
        os.remove(file_model_results)
        
    if isfile(file_model_hp):
        os.remove(file_model_hp)
    
    file_df_trn = join(dir_model_data, "df_trn.parquet")
    file_df_tst = join(dir_model_data, "df_tst.parquet")
    if not resume:
        df_trn, df_tst = prepare_R_data(
            R,
            file_raw_data,
            phase_range=hp["phase_range"],
            ptp_range=hp["ptp_range"],
            wvl_range=hp["wvl_range"],
            train_frac=hp["train_frac"],
            noise_scale=hp["noise_scale"],
            spike_scale=hp["spike_scale"],
            max_spikes=hp["max_spikes"],
            random_state=hp["random_state"],
        )
        df_trn.to_parquet(file_df_trn)
        df_tst.to_parquet(file_df_tst)

    df_trn = load_sn_data(file_df_trn)
    df_tst = load_sn_data(file_df_tst)

    # TODO: Add a function call here to generate some summary statistics
    # and/or plots based on df_trn and df_tst.

    Xtrn, Ytrn, num_trn, num_wvl, num_classes = extract(df_trn)
    Xtst, Ytst, num_tst, num_wvl, num_classes = extract(df_tst)
    if hp["add_dim"]:
        Xtrn = add_dim(Xtrn, swap=hp["swap"])
        Xtst = add_dim(Xtst, swap=hp["swap"])

    write_json(hp, file_model_hp)

    input_shape = Xtrn.shape[1:]
    model = get_model(input_shape, num_classes, hp)
    model.summary()

    compile_model(model, num_classes, hp["lr0"])
    lr_schedule = get_lr_schedule(hp)
    callbacks = get_callbacks(dir_model, lr_schedule)

    history = train(
        model,
        Xtrn,
        Ytrn,
        Xtst,
        Ytst,
        hp["epochs"],
        hp["batch_size"],
        callbacks,
    )

    results = evaluate(model, Xtrn, Ytrn, Xtst, Ytst, verbose=0)
    write_json(results, file_model_results)
    
    log = pd.read_csv(file_model_history)
    fig = dplt.plot_loss(log)
    fig.savefig(file_model_curves)
    fig.clf()


def load_SCS_model(file_model, file_model_hp):
    hp = load_json(file_model_hp)
    num_classes = hp["num_classes"]
    f1 = F1Score(num_classes=num_classes, average="macro", name="f1")
    model = load_model(file_model, custom_objects={"F1Score": f1})
    return model


def write_json(stuff, file):
    assert not isfile(file), f"'{file}' already exists."
    dumps = json.dumps(stuff, indent=4, sort_keys=True)
    with open(file, mode="x") as f:
        f.write(dumps)
    return abspath(file)


def load_json(file):
    assert isfile(file), f"'{file}' does not exist."
    with open(file, mode="r") as f:
        dumps = "".join(f.readlines())
        stuff = json.loads(dumps)
    return stuff


def evaluate(model, Xtrn, Ytrn, Xtst, Ytst, verbose=0):
    loss_trn, ca_trn, f1_trn = model.evaluate(x=Xtrn, y=Ytrn, verbose=0)
    loss_tst, ca_tst, f1_tst = model.evaluate(x=Xtst, y=Ytst, verbose=0)
    results = {
        "trn": {"loss": loss_trn, "ca": ca_trn, "f1": f1_trn},
        "tst": {"loss": loss_tst, "ca": ca_tst, "f1": f1_tst},
    }
    return results


def train(model, Xtrn, Ytrn, Xtst, Ytst, epochs, batch_size, callbacks):
    history = model.fit(
        Xtrn,
        Ytrn,
        validation_data=(Xtst, Ytst),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks,
    )
    return history


def get_model(input_shape, num_classes, hp):
    if hp["model_type"] == "feed_forward":
        model = feed_forward.model(
            input_shape,
            num_classes,
            hp["units"],
            hp["activation"],
            hp["dropout"],
        )

    if hp["model_type"] == "transformer_encoder":
        model = transformer_encoder.model(
            input_shape,
            num_classes,
            hp["encoder_blocks"],
            hp["encoder_heads"],
            hp["encoder_key_dim"],
            hp["encoder_proj_dim"],
            hp["encoder_dropout_attention"],
            hp["encoder_dropout_projection"],
            hp["feed_forward_units"],
            hp["feed_forward_activation"],
            hp["feed_forward_dropout"],
        )

    return model


def compile_model(model, num_classes, lr0):
    loss = CategoricalCrossentropy()
    acc = CategoricalAccuracy(name="ca")
    f1 = F1Score(num_classes=num_classes, average="macro", name="f1")
    opt = Nadam(learning_rate=lr0)
    model.compile(loss=loss, optimizer=opt, metrics=[acc, f1])


def get_callbacks(model_dir, lr_schedule):
    checkpoint = callbacks.ModelCheckpoint(
        join(model_dir, "model.hdf5"),
        monitor="val_loss",
        verbose=2,
        mode="min",
        save_best_only=True,
    )

    early = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=2,
        mode="min",
        restore_best_weights=True,
    )
    
    rlrp = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=0,
        mode="min",
        min_delta=0,
        cooldown=0,
        min_lr=0,
    )


    logger = callbacks.CSVLogger(join(model_dir, "history.log"), append=True)

    backup_dir = join(model_dir, "backup")
    backup = callbacks.BackupAndRestore(backup_dir=backup_dir)

    schedule = callbacks.LearningRateScheduler(lr_schedule)

    return [checkpoint, early, rlrp, logger, backup, schedule]


if __name__ == "__main__":
    app.run(main)
