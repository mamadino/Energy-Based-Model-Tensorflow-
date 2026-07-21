import tensorflow as tf
from keras import layers
import keras
import numpy as np

class sampler():

    def __init__(self, K, batch_size=32):
        self.K = K
        self.dt = 1/K
        self.sample_buffer = np.random.uniform(0, 1, size=(batch_size, 28, 28))

    def langevin(self, init_x):
        



