from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


class ResidualBlock(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        residual = x
        x = nn.GroupNorm(num_groups=8, dtype=jnp.float32)(x)
        x = nn.swish(x)
        x = nn.Conv(self.filters, kernel_size=(3, 3))(x)
        x = nn.GroupNorm(num_groups=8, dtype=jnp.float32)(x)
        x = nn.swish(x)

        x = nn.Conv(self.filters, kernel_size=(3, 3), use_bias=False)(x)

        if input_dim != self.filters:
            residual = nn.Conv(self.filters, kernel_size=(1, 1), use_bias=False)(x)

        return x + residual


class Encoder(nn.Module):
    latent_dim: int
    channel_multipliers: Sequence[int]
    filters: int = 128
    num_res_block: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.filters, kernel_size=(3, 3), use_bias=False)(x)
        num_blocks = len(self.channel_multipliers)
        for i in range(num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            for _ in range(self.num_res_block):
                x = ResidualBlock(filters)(x)
            if i < num_blocks - 1:
                x = nn.Conv(features=filters, kernel_size=(4, 4), strides=(2, 2))(x)

        for _ in range(self.num_res_block):
            x = ResidualBlock(filters)(x)

        x = nn.GroupNorm(num_groups=8, dtype=jnp.float32)(x)
        x = nn.swish(x)
        x = nn.Conv(features=self.latent_dim, kernel_size=(1, 1))(x)

        return x


class Decoder(nn.Module):
    output_channels: int
    channel_multipliers: Sequence[int]
    filters: int = 128
    num_res_block: int = 2

    @nn.compact
    def __call__(self, x):
        num_blocks = len(self.channel_multipliers)
        filters = self.filters * self.channel_multipliers[-1]
        x = nn.Conv(features=filters, kernel_size=(3, 3))(x)
        for _ in range(self.num_res_block):
            x = ResidualBlock(filters)(x)
        for i in reversed(range(num_blocks)):
            filters = self.filters * self.channel_multipliers[i]
            for _ in range(self.num_res_block):
                x = ResidualBlock(filters)(x)
            if i > 0:
                n, h, w, c = x.shape
                x = jax.image.resize(x, (n, h * 2, w * 2, c), method='nearest')
                x = nn.Conv(features=filters, kernel_size=(3, 3))(x)

        x = nn.GroupNorm(num_groups=8, dtype=jnp.float32)(x)
        x = nn.swish(x)
        x = nn.Conv(features=self.output_channels, kernel_size=(3, 3))(x)

        return x


class VectorQuantizer(nn.Module):
    embedding_dim: int
    num_embeddings: int
    commitment_cost: float

    @nn.compact
    def __call__(self, z_e):  # z_e -> [N, H, W, D]
        # Codebook
        codebook = self.param(
            'codebook',
            jax.nn.initializers.variance_scaling(
                scale=1.0, mode='fan_in', distribution='uniform'),
            (self.num_embeddings, self.embedding_dim))
        codebook = jnp.asarray(codebook, dtype=jnp.float32)
        z_e_flat = jnp.reshape(z_e, (-1, self.embedding_dim))  # [NxHxW, D]
        distances = (
            jnp.sum(z_e_flat**2, axis=1, keepdims=True)
            - 2 * jnp.dot(z_e_flat, codebook.T)
            + jnp.sum(codebook**2, axis=1)
        )  # [NxHxW, K]

        encoding_indices = jnp.argmin(distances, axis=1)  # [NxHxW]
        quantized = codebook[encoding_indices]  # [NxHxW, D]

        quantized = jnp.reshape(quantized, z_e.shape)  # [N, H, W, D]
        commitment_loss = self.commitment_cost * jnp.mean((jax.lax.stop_gradient(quantized) - z_e) ** 2)
        embedding_loss = jnp.mean((quantized - jax.lax.stop_gradient(z_e)) ** 2)

        quantized = z_e + jax.lax.stop_gradient(quantized - z_e)
        return quantized, commitment_loss, embedding_loss, encoding_indices


class VQGAN(nn.Module):
    channel_multipliers: Sequence[int]
    embedding_dim: int = 256
    num_embeddings: int = 1024
    commitment_cost: float = 0.25
    output_channels: int = 3

    def setup(self):
        self.encoder = Encoder(
            latent_dim=self.embedding_dim,
            channel_multipliers=self.channel_multipliers,
        )
        self.quantizer = VectorQuantizer(
            embedding_dim=self.embedding_dim,
            num_embeddings=self.num_embeddings,
            commitment_cost=self.commitment_cost,
        )
        self.decoder = Decoder(
            output_channels=self.output_channels,
            channel_multipliers=self.channel_multipliers,
        )

    def __call__(self, x):
        z_e = self.encoder(x)
        z_q, commitment_loss, embedding_loss, enc_indices = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, z_q, commitment_loss, embedding_loss, enc_indices

    def encode(self, x):
        latents = self.encoder(x)  # [N, H, W, D]
        flat_latents = latents.reshape(-1, self.embedding_dim)  # [NxHxW, D]

        codebook = self.quantizer.variables['params']['codebook']  # [K, D]
        distances = (
                jnp.sum(flat_latents ** 2, axis=1, keepdims=True) +
                jnp.sum(codebook ** 2, axis=1) -
                2 * flat_latents @ codebook.T
        )  # [NxHxW, K]
        indices = jnp.argmin(distances, axis=1)  # [NxHxW]
        return indices.reshape(latents.shape[:3])  # [N, H, W]

    def decode(self, x):
        # x: [N, H, W]
        features = jnp.take(self.quantizer.variables['params']['codebook'], x, axis=0)
        return self.decoder(features)
