import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf

class LivePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs["loss"])
        self.val_loss.append(logs["val_loss"])
        self.train_acc.append(logs["accuracy"])
        self.val_acc.append(logs["val_accuracy"])

        clear_output(wait=True)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy plot
        axs[0].plot(self.train_acc, label='Train Accuracy')
        axs[0].plot(self.val_acc, label='Val Accuracy')
        axs[0].set_title("Accuracy")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend()
        axs[0].grid(True)

        # Loss plot
        axs[1].plot(self.train_loss, label='Train Loss')
        axs[1].plot(self.val_loss, label='Val Loss')
        axs[1].set_title("Loss")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Loss")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

    def report(self):
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
        }
