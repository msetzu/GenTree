import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
import tensorflow as tf
from time import time
from math import ceil, log2

from joblib import Parallel, delayed

# Vedere https://www.tensorflow.org/tutorials/generative/cvae

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2]) 
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)

    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, input_shape=(32, 2)): 
        super(CVAE, self).__init__()
        self._input_shape = input_shape
        self.latent_dim = ceil(log2(input_shape[0])) 
        self._target_shape = (int(input_shape[0]/4), 32) # 4=2*2*1 strides of Conv1DTranspose layers
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self._input_shape), 
                tf.keras.layers.Conv1D(
                    filters=32, kernel_size=3, activation='relu'), 
                tf.keras.layers.Conv1D(
                    filters=64, kernel_size=3, activation='relu'),  
                # No activation
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=self._target_shape[0]*self._target_shape[1], activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=self._target_shape),
                tf.keras.layers.Conv1DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same', 
                    activation='relu'),
                tf.keras.layers.Conv1DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same', 
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv1DTranspose(
                    filters=2, kernel_size=3, strides=1, padding='same'),
            ]
        )

    @tf.function(experimental_compile=True)
    def sample(self, n_samples=100, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(n_samples, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def fit(
        self, 
        epochs, 
        train_dataset, 
        test_dataset, 
        min_delta=0.05, 
        delta_patient=3, 
        max_patient=2,
        optimizer=tf.keras.optimizers.Adam(1e-4) 
    ):

        self.plot_elbo = [] 
        delta_count = 0
        max_count = 0
        previous_elbo = None

        for epoch in range(1, epochs + 1):
            start_time = time()
            optimizer._create_all_weights(self.trainable_variables)
            for train_x in train_dataset:
                #optimizer._create_all_weights(self.trainable_variables)
                train_step(self, train_x, optimizer)
            end_time = time()

            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(compute_loss(self, test_x))

            elbo = -loss.result()
            self.plot_elbo.append(elbo.numpy()) 
                    
            display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, elbo, end_time - start_time))

            # Early stopping
            if epoch == 1:
                previous_elbo = elbo
            elif epoch >= 2:
                elbo_diff = abs(previous_elbo - elbo)

                if elbo_diff <= min_delta: # Se il miglioramento è minimo
                    delta_count += 1
                if delta_count > delta_patient:
                    break

                if elbo < 0 and elbo < previous_elbo: # Se la elbo è "peggiore"
                    max_count += 1
                elif elbo > 0 and elbo > previous_elbo:
                    max_count += 1
                else: 
                    max_count = 0

                if max_count > max_patient:
                    break

                previous_elbo = elbo

    def plot_loss(self):
        sns.set_theme()
        plt.gca().invert_yaxis()
        plt.xlabel('epoch')
        plt.ylabel('ELBO')
        plt.plot(self.plot_elbo)

    def latent_space(self, n_samples=1, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(n_samples, self.latent_dim))
        return eps

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
