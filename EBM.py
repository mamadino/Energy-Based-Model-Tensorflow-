import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras import layers
import keras
import numpy as np
import pandas as pd
import importlib
import modified_energy_model
importlib.reload(modified_energy_model)

class EBM(keras.Model):

    def __init__(self, K, dt, num_samples=6000, replay_buffer_size=10_000, batch_size=32, load_EBM=False):
        super().__init__()
        self.num_samples = num_samples
        self.K = K
        self.dt = dt
        self.batch_size = batch_size
        self.buffer_size = replay_buffer_size
        self.sample_buffer = tf.random.uniform(shape=(replay_buffer_size, 28, 28, 1), minval=0.0, maxval=1.0)
        self.x_train, self.y_train = self.dataloader()
        print('x_train shape: ', self.x_train.shape)
        pretrained_model_path = "./pretrained_model/tiny_resnet_mnist.keras"
        self.model = modified_energy_model.modified_model(pretrained_model_path)
        if load_EBM:
            self.load('./pretrained_model/')
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    def call(self, x):
        return self.model(x)

    def langevin(self, K, dt, only_inference=False, init_x=None):
        if init_x is not None:
            x = init_x
        else:
            if only_inference:
                x = tf.random.uniform(shape=(self.batch_size, 28 ,28, 1), minval=0, maxval=1)
            else:
                selector = np.random.rand()
                if selector < 0.95:
                    idx = np.random.choice(self.buffer_size, self.batch_size,replace=False)
                    x = tf.gather(self.sample_buffer, idx)
                else:
                    x = tf.random.uniform(shape=(self.batch_size, 28 ,28, 1), minval=0, maxval=1)
                    idx = np.random.choice(self.buffer_size, self.batch_size,replace=False)
        noise_scale = tf.sqrt(tf.cast(dt, dtype=tf.float32))
        for step in range(K):
            print('langevin step: ', step+1)
            with tf.GradientTape() as tape:
                tape.watch(x)
                energy = self.model(x)
            score = tape.gradient(energy, x)
            x = x - dt/2 * score + noise_scale*tf.random.normal(shape=x.shape,mean=0,stddev=1) 
            x = tf.clip_by_value(x, 0.0, 1.0)
        x = tf.stop_gradient(x)
        if not only_inference:
            if init_x is None:
                self.sample_buffer = tf.tensor_scatter_nd_update(self.sample_buffer, np.expand_dims(idx,1), x)
        print('langevin done!')
        return x
    
    def dataloader(self):
        train_path = "./dataset/MNist/mnist_train.csv"
        train_df = pd.read_csv(train_path)
        y_train = train_df.iloc[:, 0].values
        x_train = train_df.iloc[:, 1:].values
        x_train = x_train.astype("float32") / 255.0
        x_train = x_train.reshape(-1, 28, 28, 1)
        selected_indices = np.random.choice(np.arange(len(x_train)), self.num_samples, replace=False)
        x_train = x_train[selected_indices]
        y_train = y_train[selected_indices]
        return x_train, y_train

    def train_step(self, x_pos):
        alpha=0.0001
        x_neg = self.langevin(self.K, self.dt, only_inference=False)
        with tf.GradientTape() as tape:
            pos_energy = tf.reduce_mean(self.model(x_pos))
            neg_energy = tf.reduce_mean(self.model(x_neg))
            loss = pos_energy - neg_energy + alpha * (tf.square(pos_energy) + tf.square(neg_energy))
        print(f'train step\n  pos_energy: {pos_energy.numpy():.4f}\n  neg_energy: {neg_energy.numpy():.4f}')
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return {
            "loss": loss,
            "pos_energy": pos_energy,
            "neg_energy": neg_energy
        }
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_weights(os.path.join(path, "corrected_trained_EBM.weights.h5"))
        np.save(os.path.join(path, "corrected_replay_buffer.npy"), self.sample_buffer.numpy())

    def load(self, path):
        self.model.load_weights(os.path.join(path, "corrected_trained_EBM.weights.h5"))
        buffer = np.load(os.path.join(path, "corrected_replay_buffer.npy"))
        self.sample_buffer = tf.convert_to_tensor(buffer, dtype=tf.float32)
    
    def inference(self, K, dt, only_inference=False, inpainting=False):
        if inpainting:
            train_path = "./dataset/MNist/mnist_train.csv"
            train_df = pd.read_csv(train_path)
            x_train = train_df.iloc[:, 1:].values
            x_train = x_train.astype("float32") / 255.0
            x_train = x_train.reshape(-1, 28, 28, 1)
            idx = np.random.choice(len(x_train), self.batch_size, replace=False)
            orig_batch = x_train[idx].copy()
            batch = x_train[idx].copy()
            H = batch.shape[1]
            batch[:, H//2:, :, :] = 0
            batch = tf.convert_to_tensor(batch)
            img = self.langevin(K, dt, only_inference=only_inference, init_x=batch).numpy()    
        else:
            img = self.langevin(K, dt, only_inference=only_inference, init_x=None).numpy()
        
        if inpainting:
            fig, axes = plt.subplots(3, 4, figsize=(8, 6))

            for i in range(4):
                # original
                axes[0, i].imshow(orig_batch[i, :, :, 0], cmap="gray")
                axes[0, i].axis("off")
                if i == 0:
                    axes[0, i].set_title("Original")

                # masked
                axes[1, i].imshow(batch.numpy()[i, :, :, 0], cmap="gray")
                axes[1, i].axis("off")
                if i == 0:
                    axes[1, i].set_title("Masked")

                # reconstruction
                axes[2, i].imshow(img[i, :, :, 0], cmap="gray")
                axes[2, i].axis("off")
                if i == 0:
                    axes[2, i].set_title("Reconstructed")

            plt.tight_layout()
            plt.show()
        else:
            fig, axes = plt.subplots(4, 4, figsize=(6, 6))
            idx = 0
            for i in range(4):
                for j in range(4):
                    axes[i, j].imshow(img[idx, :, :, 0], cmap="gray")
                    axes[i, j].axis("off")
                    idx += 1
            plt.tight_layout()
            plt.show()
            
