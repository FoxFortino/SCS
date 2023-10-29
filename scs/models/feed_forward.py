from tensorflow import keras
from keras import layers
from keras import regularizers


def model(
    input_shape,
    num_classes,
    units,
    activation="relu",
    dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for n in units:
        x = layers.Dense(
            n,
            activation=activation,
        )(x)
        x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    return model
