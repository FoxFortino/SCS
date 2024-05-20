### SNR Testing

The algorithm for injecting gradually more and more noise into a spectrum in order to test the SNR dependence of the model is as follows:

1. Before training or preproccessing, each spectrum is de-noised:
    1. A savgol filter is applied to each spectrum.
    2. The filtered spectrum is substracted from the original spectrum, giving us the noise spectrum.
    3. The noise spectrum is multiplied by some factor (this factor is how much noise we want to add back into the spectrum.
2. The new dataset where the noise for each spectrum has been changed is now preprocessed, and trained on in no other special way.

This algorithm takes place in `SCS/scs/snr_test.py`.
