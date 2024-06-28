import sys
from os.path import join
from os.path import isfile

from icecream import ic
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.signal import savgol_filter
from scipy import stats

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


def load_original_dataset():
    ic()
    file_df_raw = "../data/raw/sn_data.parquet"
    df_raw = pd.read_parquet(file_df_raw)
    return df_raw

def load_R100_data():
    ic()
    file_df_R = "../data/R100/df_R.parquet"
    file_df_C = "../data/R100/df_C.parquet"
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


def add_dim(X, swap=False):
    X = X[..., None]
    if swap:
        X = np.swapaxes(X, 1, 2)
    return X


def get_model(
    input_shape, 
    num_classes,
    encoder_blocks,
    encoder_heads,
    encoder_key_dim,
    encoder_proj_dim,
    encoder_dropout_attention,
    encoder_dropout_projection,
    feed_forward_units,
    feed_forward_activation,
    feed_forward_dropout,
):
    ic()
    model = transformer_encoder.model(
        input_shape,
        num_classes,
        encoder_blocks,
        encoder_heads,
        encoder_key_dim,
        encoder_proj_dim,
        encoder_dropout_attention,
        encoder_dropout_projection,
        feed_forward_units,
        feed_forward_activation,
        feed_forward_dropout,
    )
    return model


def get_callbacks(model_dir):
    ic()
    cb_es = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
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
        factor=0.5,
        patience=5,
        verbose=2,
        mode="min",
        min_delta=0,
        cooldown=0,
        min_lr=0,
    )

    cb_log = callbacks.CSVLogger(join(model_dir, "history.log"), append=False)
    cbs = [cb_es, cb_mc, cb_rlrp, cb_log]
    return cbs


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
    lr0,
    encoder_blocks,
    encoder_heads,
    encoder_key_dim,
    encoder_proj_dim,
    encoder_dropout_attention,
    encoder_dropout_projection,
    feed_forward_units,
    feed_forward_activation,
    feed_forward_dropout,
    epochs=10_000,
    batch_size=32,
    ):
    rng = np.random.RandomState(1415)
    
    if R == 100:
        df_C, df_R = load_R100_data()
    else:
        df_raw = load_original_dataset()
        df_C, df_R = degrade_data(df_raw, R)
        
    # df_R is rebinned degraded spectra, df_C is continuous (not rebinned) spectra.
    # We will use df_R.

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

    Xtrn, Ytrn, num_trn, num_wvl, num_classes = extract(df_RPA_trn)
    Xtst, Ytst, num_tst, num_wvl, num_classes = extract(df_RP_tst)
    
    Xtrn = add_dim(Xtrn, swap=True)
    Xtst = add_dim(Xtst, swap=True)

    input_shape = Xtrn.shape[1:]
    model = get_model(
        input_shape,
        num_classes,
        encoder_blocks,
        encoder_heads,
        encoder_key_dim,
        encoder_proj_dim,
        encoder_dropout_attention,
        encoder_dropout_projection,
        feed_forward_units,
        feed_forward_activation,
        feed_forward_dropout,
    )
    model.summary()

    loss = CategoricalCrossentropy()
    acc = CategoricalAccuracy(name="ca")
    f1 = F1Score(num_classes=num_classes, average="macro", name="f1")
    opt = Nadam(learning_rate=lr0)
    model.compile(loss=loss, optimizer=opt, metrics=[acc, f1])

    cbs = get_callbacks(model_dir)

    history = model.fit(
        Xtrn,
        Ytrn,
        validation_data=(Xtst, Ytst),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=cbs,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--SLURM_ARRAY_TASK_ID", type=int, required=True)
    parser.add_argument("--SLURM_RESTART_COUNT", required=True)
    parser.add_argument("--R", type=int, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--phase_range_start", type=float, required=True)
    parser.add_argument("--phase_range_end", type=float, required=True)
    parser.add_argument("--ptp_range_start", type=float, required=True)
    parser.add_argument("--ptp_range_end", type=float, required=True)
    parser.add_argument("--wvl_range_start", type=float, required=True)
    parser.add_argument("--wvl_range_end", type=float, required=True)
    parser.add_argument("--train_frac", type=float, required=True)
    parser.add_argument("--noise_scale", type=float, required=True)
    parser.add_argument("--spike_scale", type=float, required=True)
    parser.add_argument("--max_spikes", type=float, required=True)
    parser.add_argument("--lr0", type=float, required=True)
    parser.add_argument("--encoder_blocks", type=int, required=True)
    parser.add_argument("--encoder_heads", type=int, required=True)
    parser.add_argument("--encoder_key_dim", type=int, required=True)
    parser.add_argument("--encoder_proj_dim", type=int, required=True)
    parser.add_argument("--encoder_dropout_attention", type=float, required=True)
    parser.add_argument("--encoder_dropout_projection", type=float, required=True)
    parser.add_argument("--feed_forward_units", type=int, required=True)
    parser.add_argument("--feed_forward_activation", type=str, required=True)
    parser.add_argument("--feed_forward_dropout", type=float, required=True)
    args = parser.parse_args()
                        
    main(
        args.R,
        args.model_dir,
        args.phase_range_start,
        args.phase_range_end,
        args.ptp_range_start,
        args.ptp_range_end,
        args.wvl_range_start,
        args.wvl_range_end,
        args.train_frac,
        args.noise_scale,
        args.spike_scale,
        args.max_spikes,
        args.lr0,
        args.encoder_blocks,
        args.encoder_heads,
        args.encoder_key_dim,
        args.encoder_proj_dim,
        args.encoder_dropout_attention,
        args.encoder_dropout_projection,
        args.feed_forward_units,
        args.feed_forward_activation,
        args.feed_forward_dropout,
        epochs=10_000,
        batch_size=32,
    )
