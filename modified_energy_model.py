import keras

def modified_model(pretrained_model_path):
    SN = keras.layers.SpectralNormalization
    old_model = keras.models.load_model(pretrained_model_path)
    hidden_layers = keras.Model(
        inputs = old_model.input,
        outputs = old_model.layers[-2].output
    )
    x = hidden_layers.output
    energy_head = SN(keras.layers.Dense(1), name='energy_head')(x)
    return keras.Model(inputs = hidden_layers.input, outputs=energy_head)
