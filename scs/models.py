from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def model_transformer(
    input_shape,
    num_classes,
    num_transformer_blocks,
    num_heads,
    key_dim,
    kr_l2,
    br_l2,
    ar_l2,
    dropout_attention,
    dropout_projection,
    filters,
    num_feed_forward_layers,
    feed_forward_layer_size,
    dropout_feed_forward,
):
    inputs = keras.Input(shape=input_shape)

    x = inputs
    for i in range(num_transformer_blocks):
        sublayer_start = x
        x = layers.LayerNormalization(epsilon=1e-6)(sublayer_start)
        x = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            kernel_regularizer=regularizers.L2(kr_l2),
            bias_regularizer=regularizers.L2(br_l2),
            activity_regularizer=regularizers.L2(ar_l2),
        )(x, x)
        x = layers.Dropout(dropout_attention)(x)
        sublayer_end = x + sublayer_start

        x = layers.LayerNormalization(epsilon=1e-6)(sublayer_end)
        x = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            activation="relu",
            kernel_regularizer=regularizers.L2(kr_l2),
            bias_regularizer=regularizers.L2(br_l2),
            activity_regularizer=regularizers.L2(ar_l2),
        )(x)
        x = layers.Dropout(dropout_projection)(x)
        x = layers.Conv1D(filters=input_shape[-1], kernel_size=1)(x)
        x = sublayer_end + x

    x = layers.GlobalMaxPooling1D(data_format="channels_first")(x)
    for i in range(num_feed_forward_layers):
        x = layers.Dense(
            feed_forward_layer_size,
            activation="relu",
            kernel_regularizer=regularizers.L2(kr_l2),
            bias_regularizer=regularizers.L2(br_l2),
            activity_regularizer=regularizers.L2(ar_l2),
        )(x)
        x = layers.Dropout(dropout_feed_forward)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    return model