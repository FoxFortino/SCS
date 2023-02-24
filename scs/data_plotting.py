import numpy as np
from matplotlib import pyplot as plt

import data_preparation as dp

def plot_specs(
    sn_data, ind,
    ncols=4, scale=4,
):
    # The function below neatly and reproducibly extracts all of the relevant 
    # subsets of the dataframe.
    data = dp.extract_dataframe(sn_data)
    index = data[0]  # SN Name for each spectrum
    wvl0 = data[1]  # Wavelength array
    flux0_columns = data[2]  # Columns that index the fluxes in the dataframe
    metadata_columns = data[3]  # Columns that index the metadata
    df_fluxes0 = data[4]  # Sub-dataframe containing only the fluxes
    df_metadata = data[5]  # Sub-dataframe containing only the metadata
    fluxes0 = data[6]  # Only the flux values in a numpy array
    
    nrows = int(np.ceil(ind.size / ncols))

    fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(ncols*scale, nrows*scale))
    axes = ax.flatten()

    for i, row in enumerate(sn_data.iloc[ind].iterrows()):
        sn_name = row[0]
        sn_subtype = row[1]["SN Subtype"]
        sn_phase = row[1]["Spectral Phase"]
        title = f"{sn_name} | {sn_subtype} | {sn_phase}"
        axes[i].set_title(title)

        spectrum = row[1][flux0_columns].to_numpy(float)
        axes[i].plot(wvl0, spectrum)
        
        axes[i].axhline(y=0, c="k", ls=":")

    fig.show()


def plot_random_spectra(df, N, ncols=4, scale=4):
    inds = np.random.randint(low=0, high=df.shape[0], size=N)
    
    nrows = int(np.ceil(inds.size / ncols))
    
    fig, ax  = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(ncols*scale, nrows*scale))
    axes = ax.flatten()
    
    for i, row in enumerate(df.iloc[inds].iterrows()):
        sn_name = row[0]
        sn_subtype = row[1]["SN Subtype"]
        sn_phase = row[1]["Spectral Phase"]
        title = f"{sn_name} | {sn_subtype} | {sn_phase}"
        axes[i].set_title(title)

        spectrum = row[1][flux_columns].to_numpy(float)
        axes[i].plot(wvl0, spectrum)
        
        axes[i].axhline(y=0, c="k", ls=":")
    fig.show()

    
def plot_history(history):
    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=True, sharey=False, figsize=(10, 8)
    )
    plt.subplots_adjust(hspace=0)

    axes[0].plot(history["epoch"], history["loss"], label="Training")
    axes[0].plot(history["epoch"], history["val_loss"], label="Testing")
    axes[0].axhline(y=history["loss"].min(), c="k", ls=":")
    axes[0].axhline(y=history["val_loss"].min(), c="k", ls=":")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Categorical Cross Entropy")

    axes[0].grid()
    axes[0].legend()

    axes[1].plot(
        history["epoch"], history["ca"], c="tab:blue", ls="-", label="Accuracy (Trn)"
    )
    axes[1].plot(
        history["epoch"],
        history["val_ca"],
        c="tab:orange",
        ls="-",
        label="Accuracy (Tst)",
    )

    axes[1].plot(
        history["epoch"], history["f1"], c="tab:blue", ls=":", label="F1-Score (Trn)"
    )
    axes[1].plot(
        history["epoch"],
        history["val_f1"],
        c="tab:orange",
        ls=":",
        label="F1-Score (Tst)",
    )
    axes[1].axhline(y=0.547, c="k", label="~Random Guess Accuracy")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")

    axes[1].legend()
    axes[1].grid()

    # axes[0].set_ylim((0, 5))
    axes[1].set_ylim((0, 1))

    # axes[0].set_xlim((0, 500))

    axes[0].set_yscale("log")
    # axes[1].set_yscale("log")

    plt.show()
