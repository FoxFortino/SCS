To Do
=============
- Refactor FFT smoothing code
    - Submit a pull request to change it

- Extract noise profile once and save it

- Look through Blondin and Tonry to figure out how they defined SNR

- Function to plot spectrum given only df and index. If index is None or something, let it be random

- Visually inspect every single raw spectra and just mark the bad spectra by hand so I don't have to use the ptp cropping thingns

- Need to do data augmentation before preprocessing I think... generally should consider the order of operations. Consider what order I need to do things in so that I have to redo the least amount of steps.

- Normalization/Standardization... Should I min-max normalize? I think that is what makes sense when binning the data for the positional encoding...
