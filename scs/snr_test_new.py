import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.signal import savgol_filter
from scipy import optimize as opt
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Nadam

from tensorflow_addons.metrics import F1Score

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# My packages
sys.path.insert(0, "../scs/")
import data_degrading as dd
import data_preparation as dp
import data_augmentation as da
from prepare_datasets_for_training import extract
import scs_config

sys.path.insert(0, "../scs/models/")
import feed_forward
# import transformer_encoder

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PLOT = False
# PLOT = True
TESTING = False
# TESTING = True


def get_noise_scale_arr():
    noise_scale_arr = np.linspace(0, 10, num=101)
    return noise_scale_arr


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
    return model


def invertrfftfreq(x, bs):
    """ this utility function helps making the x axis in the plots interpretable; invert rfft"""
    freq = np.fft.rfftfreq(len(x))
    N = len(freq) * 2
    if len(x) % 2:
        return np.arange(0, max(freq * N * bs) * 2, bs)
    else:
        return np.arange(0, max(freq * N * bs) * 2 - 2 * bs, bs)


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


def preppowerlaw(wvl, flux, cut_vel, c_kms, vel_toosmall, vel_toolarge, plot=False):
    """this function contains functionality for noise extraction in common for the SNID and non SNID case"""
    wvl_ln = np.log(wvl)  # log base e
    binsize = wvl_ln[-1] - wvl_ln[-2]  # equal bin size in log space

    f_bin, wln_bin = binspec(wvl_ln, flux, min(wvl_ln), max(wvl_ln), binsize)  # binned spectrum

    fbin_ft = np.fft.fft(f_bin)  # *len(f_bin) # real fft of the binned spectrum
    freq = np.fft.fftfreq(wln_bin.shape[0], binsize)  # 1 / ln(wavelength)
    indx = np.arange(1, freq.shape[0] // 2)
    ps = np.abs(fbin_ft[indx])  # magnitude of the power spectrum = sqrt(P)

    if plot:
        plt.figure(1)
        plt.plot(wvl_ln, flux)
        plt.plot(wln_bin, f_bin, '--')
        plt.title("spectrum")
        plt.show()

        plt.figure(2)
        plt.plot(freq, fbin_ft)
        plt.xlabel("frequency (1/ln(wavelength))")
        plt.yscale('log')
        plt.title("FFT")
        plt.show()

    # Dlambda/lambda = v/c
    freq_natural_units = freq / c_kms  # * binsize

    num_upper = np.arange(len(freq))[1.0 / freq_natural_units >= vel_toosmall][-1]
    num_lower = np.arange(len(freq))[1.0 / freq_natural_units <= vel_toolarge][0]
    mag_avg = np.mean(ps[num_lower:num_upper])

    # power spectrum in the signal region
    xps = freq[num_lower:num_upper]
    yps = ps[num_lower:num_upper]

    finite_mask = np.logical_not(ps == 0)
    finite_mask = np.logical_and(finite_mask, np.isfinite(ps))
    if finite_mask.sum() == 0:
        print("no good data points here")
        return None, None, None, None, None, None, None, None, None

    if plot:

        plt.figure(3)
        plt.plot(freq[indx], ps)
        plt.axvline(1.0 / (vel_toosmall / c_kms))
        plt.axvline(1.0 / (vel_toolarge / c_kms))
        plt.plot(freq[indx][finite_mask],
                 ps[finite_mask],  'c--')
        plt.plot(xps, yps, color='r')
        plt.xlabel("frequency (1/ln(wavelength))")
        plt.title("power spectrum")
        plt.yscale('log')
        plt.ylabel("power")
        plt.yscale('log')
        plt.show()

    powerlaw = lambda x, amp, exp: amp * x ** exp

    # TODO FBB: these should not be hard coded
    exp_guess = 2  # *slope -hard coded atm
    amp_guess = 800  # np.exp(intercept) hard coded atm

    ampfit, expfit = opt.curve_fit(
        powerlaw,
        freq[indx][finite_mask],
        ps[finite_mask],  # sigma=np.sqrt(freq[indx][finite_mask]),
        p0=[amp_guess, exp_guess],
    )[0]

    if plot:
        plt.figure(5)
        plt.axvline(1.0 / (vel_toosmall / c_kms))
        plt.axvline(1.0 / (vel_toolarge / c_kms))
        plt.plot(freq[indx][finite_mask], ps[finite_mask])
        plt.plot(freq[indx][finite_mask], powerlaw(freq[indx][finite_mask], ampfit, expfit), 'k--')
        plt.xticks(plt.xticks()[0], labels=["%d" % (t * binsize * c_kms)
                                            for t in plt.xticks()[0]],
                   rotation=45)
    # TODO FBB: this should be cleaner - ATM returning everything _and_ the kitchen sink

    return mag_avg, ampfit, expfit, fbin_ft, wln_bin, xps, yps, freq, f_bin


def smooth(wvl, flux, cut_vel, sv=None, plot=False, snidified=False):
    c_kms = 299792.47  # speed of light in km/s
    vel_toosmall = 3_000
    vel_toolarge = 100_000

    # common preprocessing for SNID and non SNID spectra
    mag_avg, ampfit, expfit, fbin_ft, wln_bin, xps, yps, freq, f_bin = preppowerlaw(wvl, flux, cut_vel, c_kms,
                                                                                    vel_toosmall, vel_toolarge,
                                                                                    plot=plot)
    if mag_avg is None:
        return wvl, flux, 0

    # find intersection of average fbin_ft magnitude and powerlaw fit to calculate
    # separation velocity between signal and noise.
    intersect_x = np.power((mag_avg / ampfit), 1.0 / expfit)
    sep_vel = 1.0 / intersect_x * c_kms

    if sv:
        sep_vel = sv  # allow sv to be passed as a user selected parameter - do that for SNID
    if plot:
        plt.figure(5)
        plt.axvline(intersect_x, color='purple')
        plt.xlabel("velocity")

        plt.plot([xps[0], xps[-1]], [mag_avg, mag_avg])
        plt.ylabel("power")
        plt.yscale('log')
        plt.title("power law fit")
        plt.show()

    # filter out frequencies with velocities higher than sep_vel
    smooth_fbin_ft = fbin_ft.copy()
    noise_fbin_ft = fbin_ft.copy()
    ind = np.arange(len(freq))[1.0 / freq * c_kms >= sep_vel][-1]

    noise_fbin_ft[:ind] = 0
    smooth_fbin_ft[ind:] = 0

    smooth_fbin_ft_inv = np.real(np.fft.ifft(smooth_fbin_ft))
    noise_fbin_ft_inv = np.real(np.fft.ifft(noise_fbin_ft))

    # here is the split between SNIDified and non SNIDified spectra
    if snidified:
        amplitude = lambda y, amp: amp * y
        mask = f_bin != 0

        ampfit = opt.curve_fit(amplitude,
                               smooth_fbin_ft_inv[mask],
                               f_bin[mask],  # sigma=np.sqrt(freq[indx][finite_mask]),
                               p0=[2],
                               )[0]
        smooth_fbin_ft_inv *= ampfit
        smooth_fbin_ft_inv[~mask] = 0
    else:
        from scipy.interpolate import splrep, BSpline
        tck = splrep(wln_bin[:smooth_fbin_ft_inv.shape[0]],
                     f_bin[:smooth_fbin_ft_inv.shape[0]] - smooth_fbin_ft_inv, s=9)
        smooth_fbin_ft_inv += BSpline(*tck)(wln_bin[:smooth_fbin_ft_inv.shape[0]])

    if plot:
        plt.figure(6)
        plt.plot(wln_bin, f_bin, label="orig")
        plt.plot(wln_bin, smooth_fbin_ft_inv, label="inverse")
        if not snidified:
            plt.plot(wln_bin[:smooth_fbin_ft_inv.shape[0]],
                     BSpline(*tck)(wln_bin[:smooth_fbin_ft_inv.shape[0]]),
                     label="correction")

        plt.plot(wln_bin[:smooth_fbin_ft_inv.shape[0]], smooth_fbin_ft_inv, 'k', label="corrected")
        plt.plot(wln_bin[:smooth_fbin_ft_inv.shape[0]], f_bin - noise_fbin_ft_inv, 'r--',
                 label="f-noise")
        plt.legend()
        plt.show()

    w_smoothed = np.interp(wvl, np.exp(wln_bin),
                           np.exp(wln_bin))

    f_smoothed = np.interp(wvl, w_smoothed, smooth_fbin_ft_inv)

    return w_smoothed, f_smoothed, sep_vel


def get_noise(wvl, flux, snidified=False, sv=None, plot=False):

    cut_vel = 1_000 # km/s - min line velocoty for SN
    # cut_vel_indx = np.argmax(flux)

    w_smoothed, signal, sepv = smooth(wvl, flux, cut_vel,
                                      snidified=snidified, sv=sv, plot=plot)
    if sepv == 0:
        print("WARNING: failed on this SN")
    if not (wvl == w_smoothed).all():
        print("this will fail")
    assert (wvl == w_smoothed).all(), "error in resampling spectra"
    return signal


def gen_noise(wvl, spectrum):
    if not scs_config.FILT:
        smooth = get_noise(wvl, spectrum, snidified=scs_config.SNIDIFIED,
                           plot=PLOT)

    else:
        smooth = savgol_filter(
            spectrum,
            11,
            1,
            mode="mirror",
        )

    noise = spectrum - smooth

    if PLOT:
        plt.plot(spectrum, label="spectrum")
        plt.plot(smooth, label="spectrum")
        plt.legend()
        plt.show()
        plt.plot(noise, label="noise")
        plt.legend()
        plt.show()
    return noise


def inject_noise(noise_scale, plot=False, recalculate=False):
    # do not change the noise
    if float(noise_scale) == 0:
        print("no noise manipulation")
        df = dp.load_dataset("../data/raw/sn_data.parquet")
        if TESTING:
            df = df.iloc[:10]
        return df
    # recalculate and save noise vector and clean flux
    if recalculate:
        df = dp.load_dataset("../data/raw/sn_data.parquet")
        # FBB cut for time
        if TESTING:
            df = df.iloc[:10]
        data = dp.extract_dataframe(df)
        index, wvl, flux_columns, metadata_columns, df_fluxes, df_metadata, fluxes = data

        # vectorised operation to extract noise
        vecnoise = np.vectorize(gen_noise, signature="(n),(n)->(n)")
        noise = vecnoise([wvl] * fluxes.shape[0], fluxes)
        if plot:
            for i in range(1):
                plt.plot(noise[i], label="noise")
                plt.legend()

            plt.show()
        dp.save_noise_dataset(noise, testing=TESTING)

        # generate clean spectra
        fluxes = fluxes - noise
        df.loc[:, flux_columns] = fluxes - noise

        dp.save_clean_dataset(df, testing=TESTING)

    else:
        df = dp.load_dataset("../data/raw/sn_clean.parquet")
        if TESTING:
            df = df.iloc[:10]
        data = dp.extract_dataframe(df)
        index, wvl, flux_columns, metadata_columns, df_fluxes, df_metadata, fluxes = data

        noise = dp.read_noise_dataset()
        if TESTING:
            noise = noise[:10]
    # add noise to clean flux
    fluxes_noise = fluxes + noise * float(noise_scale)  # (fluxes, noise_scale, rng)

    if plot:
        for i in range(10):
            plt.plot(fluxes[i])
            plt.plot(fluxes_noise[i])
    # reset df with modified flux
    df.iloc[:noise.shape[0]].loc[:, flux_columns] = fluxes_noise
    return df


def main(noise_scale):
    # noise scale should be 0-100
    df_raw = inject_noise(noise_scale, recalculate=True)
    if PLOT:
        plt.plot(df_raw.iloc[0, 5:], label="degraded?")
        plt.legend()
        plt.show()
    rng = np.random.RandomState(1415)

    R = 100
    df_C, df_R = dd.degrade_dataframe(R, df_raw)
    if PLOT:
        plt.plot(df_raw.iloc[0, 5:], label="degraded?")
        plt.legend()
        plt.show()

    phase_range = (-20, 50)
    ptp_range = (0.1, 100)
    wvl_range = (4500, 7000)
    df_CP, df_RP = clean_data(df_C, df_R, phase_range, ptp_range, wvl_range)
    if TESTING:
        return
    train_frac = 0.50

    df_CP_trn, df_CP_tst, df_RP_trn, df_RP_tst = split_train_test(
        df_CP, df_RP, train_frac, rng
    )

    spike_scale = 3
    max_spikes = 5
    print("augmentation")

    df_CPA_trn, df_RPA_trn = augment_training_set(
        df_CP_trn, df_RP_trn, rng, wvl_range, spike_scale, max_spikes
    )
    print("train test split")

    Xtrn, Ytrn, num_trn, num_wvl, num_classes = extract(df_RPA_trn)
    Xtst, Ytst, num_tst, num_wvl, num_classes = extract(df_RP_tst)

    input_shape = Xtrn.shape[1:]
    model = get_model(input_shape, num_classes)
    model.summary()

    lr0 = 1e-5
    loss = CategoricalCrossentropy()
    acc = CategoricalAccuracy(name="ca")
    f1 = F1Score(num_classes=num_classes, average="macro", name="f1")

    optimizer = Nadam(learning_rate=lr0)
    model.compile(loss=loss, optimizer=optimizer, metrics=[acc, f1])

    early = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=2,
        mode="min",
        restore_best_weights=True,
    )
    file_log = f"../data/snr_test/{noise_scale}_history.log"
    logger = callbacks.CSVLogger(file_log, append=False)
    cbs = [early, logger]

    print("start training")
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

    if not os.path.isfile(f"../data/snr_test/results_ns{noise_scale}.csv"):
        with open(f"../data/snr_test/results_ns{noise_scale}.csv", "w") as f:
            f.write(f"loss_trn\t ca_trn\t f1_trn\t loss_tst\t ca_tst\t f1_tst\t epochs\n")

    results = f"{loss_trn},{ca_trn},{f1_trn},{loss_tst},{ca_tst},{f1_tst},{len(history.history['loss'])}\n"
    with open(f"../data/snr_test/results_ns{noise_scale}.csv", "a") as f:
        f.write(results)


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'model loss ({noise_scale})')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    return


if __name__ == "__main__":
    print("noise scale now:", sys.argv[1])
    noise_scale = sys.argv[1]
    main(noise_scale)
