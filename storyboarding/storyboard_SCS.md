# Paper 2: Novel Classification Method For Supernova Subtype Classification

<img src="SNe_over_time.jpg" alt="Figure 1" width="300"/>

### Figure 1: Each bar represents the total number of SNe discovered each year since SN1937D, a SN Ia in NGC 1003 (Baade & Zwicky 1938). Note the y-axis is in log-scale. The blue section denotes the SNe without spectral confirmation. The orange section denotes the SNe that were spectroscopically confirmed with one or two spectra. The red section denotes SNe that have been spectroscopically studied over time with more than two spectra. As the discovery rate of SNe grows, the rate of spectral classification of SNe does not keep up. The LSST will cause an unprecedented influx of new SNe that will only add to the stress on existing spectrograph facilities. The lack of spectrally studied SNe in recent years is likely due to publication delay. Data from the Open SN Catalog API gathered on August 4th, 2022 (we note this service is no longer maintained and recent data may be incomplete).

***

<img src="treemap.jpg" alt="Figure 2" width="300"/>

### Figure 2: A 'treemap' plot showing the number of spectra in our dataset for each SN type. The area of each rectangle is proportional to the number of spectra. Blue, orange, green and magenta rectangles denote SN spectra with broadtype Ia, Ib, Ic, and II, respectively.

***

<img src="Fig2_from_Villar_et_al_2023.png" alt="Figure 2" width="300"/>

### Table 1 (or maybe a figure??): Right now this is Figure 2 from Villar et al 2023, the paper that discusses the hierarchical cross entropy loss. I am not sure how we should truly be re-categorizing the SNe so I am putting this here for now.

***

<img src="sn1998dt_spectra.jpg" alt="Figure 3" width="300"/>

### Figure 3: A spectrum from SN1998dt, Type Ib, observed 1.8 days after peak brightness. The blackbody continuum has been removed. Type Ib SNe do not show hydrogen spectral lines nor the Siii line at 6355A that characterizes SNe Ia. (Top) The spectrum is plotted at the original high-resolution of R = 738. (Middle) The spectrum is plotted at the low-resolution of R = 100, the same resolution that the SEDM operates at. (Bottom) The spectrum is plotted at the ultra low-resolution of R = 24.

***

<img src="placeholder_transformer architecture.jpg" alt="Figure 3" width="300"/>

### Figure 4: PLACEHOLDER Transformer architecture

***

<img src="placeholder_model_training_prep_params.jpg" alt="Figure 3" width="300"/>

### Table 2: Hyperparameters for the model, the training, and the preprocessing steps

***

<img src="placeholder_loss_curve.jpg" alt="Figure 3" width="300"/>

### Figure 5: Transformer and DASH training/validation loss curves at R = 100.

***

### Figure 6+: Confusion Matrices for the various models
