import tensorflow as tf, keras, numpy as np, matplotlib.pyplot as plt, importlib

class Train:
    def __init__(self, ebm):

        self.ebm = ebm
        self.batch_size = ebm.batch_size

        self.dataset = tf.data.Dataset.from_tensor_slices(
            self.ebm.x_train
        ).shuffle(10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def fit(self, epochs=10):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            avg_loss = 0
            avg_pos_energy = 0
            avg_neg_energy = 0
            steps = 0
            for x_batch in self.dataset:
                print('train step: ', steps+1)
                logs = self.ebm.train_step(x_batch)

                avg_loss += logs["loss"]
                avg_pos_energy += logs['pos_energy']
                avg_neg_energy += logs['neg_energy']
                steps += 1

            avg_loss /= steps
            avg_neg_energy /= steps
            avg_pos_energy /= steps
            self.ebm.save('./EBM_repo/pretrained_model/')
            print('Generating a sample till now...')
            self.ebm.inference(K=200, dt=0.01)

            print(f"Avg Loss: {avg_loss.numpy():.4f}")
            print(f'Avg pos energy: {avg_pos_energy.numpy():.4f}')
            print(f'Avg neg energy: {avg_neg_energy.numpy():.4f}')
