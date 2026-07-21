import tensorflow as tf
from keras import layers
import keras
import numpy as np
import pandas as pd
import resnet_bloks as mdl
import Live_plot

class energy_model():

    def __init__(self):
        train_path = "./dataset/MNist/mnist_train.csv"
        test_path  = "./dataset/MNist/mnist_test.csv"

        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)

        y_train = train_df.iloc[:, 0].values
        x_train = train_df.iloc[:, 1:].values

        self.y_test = test_df.iloc[:, 0].values
        x_test = test_df.iloc[:, 1:].values

        x_train = x_train.astype("float32") / 255.0
        x_test  = x_test.astype("float32") / 255.0

        x_train = x_train.reshape(-1, 28, 28, 1)
        self.x_test  = x_test.reshape(-1, 28, 28, 1)
        
        self.x_val = x_train[-10000:]
        self.y_val = y_train[-10000:]

        self.x_train = x_train[:-10000]
        self.y_train = y_train[:-10000]

        self.model = mdl.build_model()
        self.model.summary()
        self.model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
        )
        self.callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2),
            
        Live_plot.LivePlotCallback()    ]
        

    def train(self, epochs, batch_size):
        self.model.fit(
        self.x_train, self.y_train,
        validation_data=(self.x_val, self.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=self.callbacks
        )
        self.model.save("./pretrained_model/tiny_resnet_mnist.keras")
        self.model.save_weights("./pretrained_model/tiny_resnet_mnist.weights.h5")

        self.model.evaluate(self.x_test, self.y_test)
    
