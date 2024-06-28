import sys
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


from scipy import stats as st
import scipy
from scipy import optimize as opt


def get_noise_scale_arr():
    ic()
    noise_scale_arr = np.linspace(0, 10, num=100+1)
    return noise_scale_arr


def smooth(wvl, flux, cut_vel):
    c_kms = 299792.47 # speed of light in km/s
    vel_toolarge = 100_000 # km/s

    wvl_ln = np.log(wvl)
    num = wvl_ln.shape[0]
    binsize = wvl_ln[-1] - wvl_ln[-2]
    f_bin, wln_bin = binspec(wvl_ln, flux, min(wvl_ln), max(wvl_ln), binsize)
    
    fbin_ft = np.fft.fft(f_bin)#*len(f_bin)
    freq = np.fft.fftfreq(num)
    
    num_upper = np.max(np.where(1.0/freq[1:] * c_kms * binsize > cut_vel))
    num_lower = np.max(np.where(1.0/freq[1:] * c_kms * binsize > vel_toolarge))
    mag_avg = np.mean(np.abs(fbin_ft[num_lower:num_upper+1]))

    powerlaw = lambda x, amp, exp: amp*x**exp
    num_bin = len(f_bin)
    
    xdat = freq[num_lower:num_upper]
    ydat = np.abs(fbin_ft[num_lower:num_upper])
    
    nonzero_mask = xdat!=0
    slope, intercept, _, _, _ = st.linregress(
        np.log(xdat[nonzero_mask]),
        np.log(ydat[nonzero_mask]),
    )
    exp_guess = slope
    amp_guess = np.exp(intercept)

    #do powerlaw fit
    xdat = freq[num_lower:int(num_bin/2)]
    ydat = np.abs(fbin_ft[num_lower:int(num_bin/2)])
    
    #exclude data where x=0 because this can cause 1/0 errors if exp < 0
    finite_mask = np.logical_not(xdat==0)
    finite_mask = np.logical_and(finite_mask, np.isfinite(ydat))
    
    ampfit, expfit = opt.curve_fit(
        powerlaw,
        xdat[finite_mask],
        ydat[finite_mask],
        p0=[amp_guess, exp_guess],
    )[0]

    # find intersection of average fbin_ft magnitude and powerlaw fit to calculate
    # separation velocity between signal and noise.
    intersect_x = np.power((mag_avg/ampfit), 1.0/expfit)
    sep_vel = 1.0/intersect_x * c_kms * binsize

    # filter out frequencies with velocities higher than sep_vel
    smooth_fbin_ft = np.array(
        [fbin_ft[ind] if (np.abs(freq[ind]) < np.abs(intersect_x)) else 0 for ind in range(len(freq))]
    )#/len(f_bin)

    #inverse fft on smoothed fluxes
    smooth_fbin_ft_inv = np.real(np.fft.ifft(smooth_fbin_ft))

    # interpolate smoothed fluxes back onto original wavelengths
    w_smoothed = np.exp(wln_bin)
    f_smoothed = np.interp(wvl, w_smoothed, smooth_fbin_ft_inv)

    return w_smoothed, f_smoothed, sep_vel


def binspec(wvl, flux, wstart, wend, wbin):
    nlam = (wend - wstart) / wbin + 1
    nlam = int(np.ceil(nlam))
    outlam = np.arange(nlam) * wbin + wstart
    answer = np.zeros(nlam)
    interplam = np.unique(np.concatenate((wvl, outlam)))
    interpflux = np.interp(interplam, wvl, flux)

    for i in np.arange(0, nlam - 1):
        cond = np.logical_and(interplam >= outlam[i], interplam <= outlam[i+1])
        w = np.where(cond)
        if len(w) == 2:
            answer[i] = 0.5*(np.sum(interpflux[cond])*wbin)
        else:
            answer[i] = scipy.integrate.simps(interpflux[cond], interplam[cond])

    answer[nlam - 1] = answer[nlam - 2]
    cond = np.logical_or(outlam >= max(wvl), outlam < min(wvl))
    answer[cond] = 0
    return answer/wbin, outlam


def load_original_dataset():
    ic()
    file_df_raw = "../data/raw/sn_data.parquet"
    df_raw = pd.read_parquet(file_df_raw)
    return df_raw


# def inject_noise(df_raw, rng, noise_scale):
#     ic()
#     data = dp.extract_dataframe(df_raw)
#     index, wvl, flux_columns, metadata_columns, df_fluxes, df_metadata, fluxes = data
    
#     fluxes_noise = np.zeros_like(fluxes)
#     for i in range(fluxes.shape[0]):
#         wvl_S, flux_S, sep_vel = smooth(wvl, fluxes[i], 1000)
#         noise = fluxes[i] - flux_S
#         fluxes_noise[i] = signal + noise * noise_scale
#     df_raw[flux_columns] = fluxes_noise
    
#     return df_raw


def inject_noise(df_raw, rng, noise_scale):
    ic()
    data = dp.extract_dataframe(df_raw)
    index, wvl, flux_columns, metadata_columns, df_fluxes, df_metadata, fluxes = data
    
    noise = stats.norm.rvs(
        loc=0, scale=noise_scale, size=fluxes.shape, random_state=rng
    )

    df_raw[flux_columns] += noise
    
    return df_raw


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


def augment_training_set(df_CP_trn, df_RP_trn, rng, wvl_range, spike_scale, max_spikes):
    ic()
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
    ic()
    model = feed_forward.model(
        input_shape,
        num_classes,
        [1024, 1024, 1024],
        activation="relu",
        dropout=0.5,
    )
    return model


def main(noise_scale_i):
    df_raw = load_original_dataset()

    rng = np.random.RandomState(1415)
    noise_scale_arr = get_noise_scale_arr()
    noise_scale = ic(noise_scale_arr[noise_scale_i])
    df_raw = inject_noise(df_raw, rng, noise_scale)
    
    ic(np.nanmean(np.nanstd(df_raw.iloc[:, 5:], axis=1)))

    # R = 100
    # df_C, df_R = degrade_data(df_raw, R)
    df_C, df_R = df_raw.copy(deep=True), df_raw.copy(deep=True)

    phase_range = (-20, 50)
    ptp_range = (0.1, 100)
    wvl_range = (4500, 7000)
    df_CP, df_RP = clean_data(df_C, df_R, phase_range, ptp_range, wvl_range)

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

    results = f"{noise_scale},{loss_trn},{ca_trn},{f1_trn},{loss_tst},{ca_tst},{f1_tst}\n"
    ic(results)
    with open("../data/snr_test/results.csv", "a") as f:
        f.write(results)


if __name__ == "__main__":
    print(sys.argv)
    noise_scale_i = int(sys.argv[1])
    main(noise_scale_i)
    