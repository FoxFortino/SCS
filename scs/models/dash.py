from tensorflow import keras
from keras import layers
from keras import regularizers


def model(num_classes, dropout=0.50):
    inputs = keras.Input(shape=(1024,))
    x = layers.Reshape((32, 32, 1))(inputs)
    
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Reshape((4096,))(x)
    
    x = layers.Dropout(dropout)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    
    return model
