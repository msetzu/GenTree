"""PyTorch port of the convolutional VAE used by GenTree.

Port of ``vae.py`` (TensorFlow, based on https://www.tensorflow.org/tutorials/generative/cvae).
Data layout is kept identical to the TF version: inputs are ``(batch, length, 2)``
(channels last); the model permutes to PyTorch's ``(batch, channels, length)`` internally.
"""
from math import ceil, log2, pi, log
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from IPython import display
except ImportError:  # IPython is optional
    display = None

LOG2PI = log(2. * pi)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    mean = torch.as_tensor(mean, dtype=sample.dtype, device=sample.device)
    logvar = torch.as_tensor(logvar, dtype=sample.dtype, device=sample.device)
    return torch.sum(-.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + LOG2PI), dim=raxis)


def compute_loss(model, x):
    """Negative single-sample Monte Carlo ELBO estimate (same as the TF version)."""
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = F.binary_cross_entropy_with_logits(x_logit, x, reduction='none')
    logpx_z = -torch.sum(cross_ent, dim=(1, 2))
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -torch.mean(logpx_z + logpz - logqz_x)


def train_step(model, x, optimizer):
    """Executes one training step and returns the loss."""
    optimizer.zero_grad()
    loss = compute_loss(model, x)
    loss.backward()
    optimizer.step()
    return loss.detach()


def make_loader(data, batch_size, shuffle=True):
    """Builds a DataLoader from a numpy array of shape (n, length, 2)."""
    tensor = torch.as_tensor(np.asarray(data, dtype=np.float32))
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)


def _keras_init(module):
    # Keras defaults: glorot_uniform kernels, zero biases
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class CVAE(nn.Module):
    """Convolutional variational autoencoder."""

    def __init__(self, input_shape=(32, 2), device=None):
        super().__init__()
        self._input_shape = tuple(input_shape)
        length, channels = self._input_shape
        if length % 4 != 0:
            raise ValueError("input length must be divisible by 4 (got {})".format(length))
        self.latent_dim = ceil(log2(length))
        self._target_shape = (length // 4, 32)  # 4 = 2*2*1 strides of the transposed convolutions

        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=3), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3), nn.ReLU(),
            nn.Flatten(),
            # No activation
            nn.Linear(64 * (length - 4), self.latent_dim + self.latent_dim),
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(self.latent_dim, self._target_shape[0] * self._target_shape[1]), nn.ReLU(),
        )
        # padding='same' equivalents: k=3, s=2 -> padding=1, output_padding=1 (doubles length)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            # No activation
            nn.ConvTranspose1d(32, channels, kernel_size=3, stride=1, padding=1),
        )

        self.apply(_keras_init)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _to_tensor(self, x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32) if not torch.is_tensor(x) else x,
                               dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ #
    def encode(self, x):
        x = self._to_tensor(x)                    # (n, length, 2)
        h = self.encoder(x.permute(0, 2, 1))      # -> (n, 2, length)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean)
        return eps * torch.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        z = self._to_tensor(z)
        h = self.decoder_fc(z).view(-1, self._target_shape[1], self._target_shape[0])
        logits = self.decoder_conv(h).permute(0, 2, 1)  # -> (n, length, 2)
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits

    def forward(self, x):
        mean, logvar = self.encode(x)
        return self.decode(self.reparameterize(mean, logvar)), mean, logvar

    @torch.no_grad()
    def sample(self, n_samples=100, eps=None):
        if eps is None:
            eps = torch.randn(n_samples, self.latent_dim, device=self.device)
        return self.decode(eps, apply_sigmoid=True)

    def latent_space(self, n_samples=1, eps=None):
        if eps is None:
            eps = torch.randn(n_samples, self.latent_dim, device=self.device)
        return eps

    # ------------------------------------------------------------------ #
    def fit(
        self,
        epochs,
        train_dataset,
        test_dataset,
        min_delta=0.05,
        delta_patient=3,
        max_patient=2,
        optimizer=None,
        lr=1e-4,
        verbose=True,
    ):
        """Trains the VAE. ``train_dataset``/``test_dataset`` are iterables of batches
        (e.g. ``make_loader(...)``); a batch may be a tensor or a 1-tuple ``(tensor,)``.
        """
        if optimizer is None:
            # Fresh optimizer per call (Keras Adam defaults: eps=1e-7)
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-7)

        self.plot_elbo = []
        delta_count = 0
        max_count = 0
        previous_elbo = None

        for epoch in range(1, epochs + 1):
            start_time = time()
            self.train()
            for train_x in train_dataset:
                train_step(self, self._batch(train_x), optimizer)
            end_time = time()

            self.eval()
            total, n_batches = 0.0, 0
            with torch.no_grad():
                for test_x in test_dataset:
                    total += compute_loss(self, self._batch(test_x)).item()
                    n_batches += 1

            elbo = -total / max(n_batches, 1)
            self.plot_elbo.append(elbo)

            if verbose:
                if display is not None:
                    display.clear_output(wait=False)
                print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(
                    epoch, elbo, end_time - start_time))

            # Early stopping (same rules as the TF version)
            if epoch == 1:
                previous_elbo = elbo
            elif epoch >= 2:
                elbo_diff = abs(previous_elbo - elbo)

                if elbo_diff <= min_delta:  # minimal improvement
                    delta_count += 1
                if delta_count > delta_patient:
                    break

                if elbo < 0 and elbo < previous_elbo:  # "worse" ELBO
                    max_count += 1
                elif elbo > 0 and elbo > previous_elbo:
                    max_count += 1
                else:
                    max_count = 0

                if max_count > max_patient:
                    break

                previous_elbo = elbo

        self.eval()
        return self

    def _batch(self, batch):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return self._to_tensor(batch)

    def plot_loss(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme()
        plt.gca().invert_yaxis()
        plt.xlabel('epoch')
        plt.ylabel('ELBO')
        plt.plot(self.plot_elbo)
