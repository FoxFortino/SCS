import numpy as np
from tensorflow.keras.utils import to_categorical

from data_preparation import extract_dataframe


def extract(df):
    data = extract_dataframe(df)
    X = data[6]
    Y = data[5]["SN Subtype ID"].to_numpy(dtype=int)
    
    N = X.shape[0]
    num_wvl = X.shape[1]
    num_classes = np.unique(Y).size
    
    Y_OH = to_categorical(Y, num_classes=num_classes)

    return X, Y_OH, N, num_wvl, num_classes


def add_dim(X, swap=False):
    X = X[..., None]
    if swap:
        X = np.swapaxes(X, 1, 2)
    return X