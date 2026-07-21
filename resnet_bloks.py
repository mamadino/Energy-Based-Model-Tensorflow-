import keras, tensorflow as tf
from keras import layers

def residual_block(x, filters, stride=1):

    SN = layers.SpectralNormalization

    shortcut = x

    x = SN(layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False
    ))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = SN(layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False
    ))(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = SN(layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding="same",
            use_bias=False
        ))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


def build_model():

    inputs = keras.Input((28,28,1))

    x = layers.Conv2D(8, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = residual_block(x, 8)
    x = residual_block(x, 16, stride=2)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    return keras.Model(inputs, outputs)