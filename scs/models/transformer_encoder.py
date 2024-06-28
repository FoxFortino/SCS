from tensorflow import keras
from keras import layers
from keras import regularizers


def model(
    input_shape,
    num_classes,
    encoder_blocks,
    encoder_heads,
    encoder_key_dim,
    encoder_proj_dim,
    encoder_dropout_attention,
    encoder_dropout_projection,
    feed_forward_units,
    feed_forward_activation,
    feed_forward_dropout,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for _ in range(encoder_blocks):
        x = transformer_block(
            x,
            input_shape,
            encoder_heads,
            encoder_key_dim,
            encoder_proj_dim,
            encoder_dropout_attention,
            encoder_dropout_projection,
        )

    x = layers.GlobalMaxPooling1D(data_format="channels_first")(x)
    for n in feed_forward_units:
        x = layers.Dense(n, activation=feed_forward_activation)(x)
        x = layers.Dropout(feed_forward_dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    return model


def transformer_block(
    x,
    input_shape,
    heads,
    key_dim,
    encoder_proj_dim,
    dropout_attention,
    dropout_projection,
):
    x0 = layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim)(x, x)
    x0 = layers.Dropout(dropout_attention)(x0)
    x0 = layers.Add()([x, x0])
    x0 = layers.LayerNormalization()(x0)


    x1 = layers.Conv1D(
        filters=encoder_proj_dim,
        kernel_size=1,
        activation="relu",
    )(x0)

    x1 = layers.Conv1D(
        filters=input_shape[1],
        kernel_size=1,
        activation="relu",
    )(x1)
    x1 = layers.Dropout(dropout_projection)(x1)
    x1 = layers.Add()([x0 + x1])
    x1 = layers.LayerNormalization()(x1)

    return x1
