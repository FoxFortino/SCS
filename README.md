Dependencies
============
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/pandas-docs/version/1.1/index.html)
    - [pyarrow](https://arrow.apache.org/docs/python/index.html) (See [Pandas Docs](https://pandas.pydata.org/pandas-docs/version/1.1/user_guide/io.html#io-parquet) about the .parquet file format])
- [Abseil](https://abseil.io/docs/python/quickstart)
- [Tensorflow](https://www.tensorflow.org/)

TODO
====

BUGS
----
1. See train CA results of dev8. Why is there a model that got over 100% CA?????????

Model Development
-----------------

- Test only at R = 100 and lower. We know classification can be done at higher resolutions but let's stay focused at low R.



### Artificial Neural Networks
1. Simple Feed Forward
    - Nail down an architecture for a simple feed-forward network

2. Transformers
    - Seeing that my changes to the network didn't seem to make any changes, try adding a convolutional layer right at the start to increase the dimensionality of the data from maybe (100, 1) to (100, N).
    - Research other implementations of transformers for time series classification.
    - Investigate whether the full encoder-decoder structure of the transformer can be implemented for this task of time-series classification.
    - Try to run astronet code while varying the number of transformer blocks.
    - Try to visualize weights for the attention layers


### Alternative Model Types
1. XGBoost
    - As a popular and successful classifier, XGBoost might be worth looking into
    - This didn't work lol

2. Creative Models
    - If off-diagonal elements of the confusion matrix suggest it would be helpful, try a hierarchical model that classifies maintype and then subtype. Would need to have a different model each for classifiying subtypes of Ia, Ib, Ic, and II. This would mean 5 models in total.
        - Each sub-model for classifying maintype Ix should ahve an extra class that represents non-Ix subtypes. This way all of these sub-models can be trained on the entire dataset.

    - Add another channel alongside the spectrum which could contain anything. Maybe something like the hfft of the data?


Plotting and Misc.
------------------
1. Have all of the platting and analysis done in model_review.ipynb, do it whenever the batch job finishes running.

1. When doing hyper-parameter testing, always have one varying parameters with a number of retries OR 2 vary parameters with a number of retries. The former will produce a 1D plot and the latter will produce a 2D plot.

2. Summary plots and stats on raw data
    - A function that ingests a dataframe and outputs a complete summary of the dataset
        - Treemap of the SN subtypes
        - Treemap of the SN maintypes
        - Histogram of the spectral phases
            - Color coded by SN subtype?
            - Color coded by SN maintype?
        - N Example spectra
        - What else...

3. Plotting function that can easily plot random or specific spectra, given a dataframe.


Documentation and Support
-------------------------
- Added verbose kwarg for `load_sn_data()` and other functions that print things. Consider using the logging module or the absl logger module
- `sns_config.py` dictionaries should maybe be wrapped into functions.
- Create flow chart of functions for all of this
- Comments for the code
- Complete docstrings
- Type annotations
- Consider converting to poetry

Refactoring
-----------
1. Look into how things are handled when a job needs to be requeued. It doesn't quite seem to be working.
    - I think it has to do with how I check for whether directories exist in learn.py. I should be checking whether the files exist, not the directory exists.
    - See below: I should not be remaking the data every time... perhaps I need to generate some sort of config file that is a json or somethin with the model parameters and such. Maybe I can also write to whether or not the model was finished running in that json file. That way I can check if that bool is true or not.

2. Data should not be remade every time. It can be made once for each R and then copied.
    - Have all of the R values in the same pandas dataframe
    - The train-test split will be done every time though since we might want to change the train_frac
    - Augmentation paremeters havent really changed since I empirically found them so why not just do that

3. Figure out the best way to organize all the functions again. Clearly separate data handling operations, machine learning operations, plotting poerations into their own folders.
    - I think object oriented is not worth it in this case. I think I can have learn.

