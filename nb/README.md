# TO DO

1. Measure the "signal" for each spectra that we have.
    - Need to create a catalog where each row is a spectrum and each column is spectral feature. Each entry would then be the measured strength of that feature using our algorithm.
    - There should be flag for whether the signal was measured using the cleaned or the raw spectrum
    - There should be a flag for whether the algorithm found more than one minimum.
    
2. Data augmentation
    - Redshift augmentation: simply shift the spectrum left or right by up to 5 pixels.
    - Spikes augmentation: review this code to ensure that it is efficient and correct.
    - Noise augmentation: for each spectrum we need to look up the measured "signal" for that spectrum and then inject Gaussian noise. The stddev of the injected Gaussian noise constitutes the "noise". Inject noise to reach a user-desired signal-to-noise ratio.