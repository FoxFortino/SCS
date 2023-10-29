from tensorflow import keras
from keras import layers
from keras import regularizers

def model(
    input_shape,
    numm_classes,
    SimpleRNN_units,
    feed_forward_units,
    feed_forward_activation,
    feed_forward_dropout,
):
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    x = layers.SimpleRNN(SimpleRNN_units)(x)
    
    x = layers.Dense(
        feed_forward_units,
        activation=feed_forward_activation
    )(x)
    x = layers.Dropout(feed_forward_dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)