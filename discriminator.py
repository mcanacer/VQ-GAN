import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Sequence


class DiscriminatorBlock(nn.Module):
    in_channels: int
    out_channels: int
    downsample: bool = True

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.swish(x)
        x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding='SAME')(x)

        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.swish(x)
        x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding='SAME')(x)

        if self.in_channels != self.out_channels:
            residual = nn.Conv(features=self.out_channels, kernel_size=(1, 1), padding='SAME')(residual)

        x = x + residual

        if self.downsample:
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        return x


class Discriminator(nn.Module):
    channel_multipliers: Sequence[int]
    base_channels: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.base_channels, kernel_size=(3, 3), padding='SAME')(x)

        channels = self.base_channels
        for mult in self.channel_multipliers:
            out_channels = self.base_channels * mult
            x = DiscriminatorBlock(channels, out_channels)(x)
            channels = out_channels

        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.swish(x)
        x = nn.Conv(1, kernel_size=(3, 3), padding='SAME')(x)  # [N, H', W', 1] → patch-level logits

        return x


if __name__ == '__main__':
    disc = Discriminator([1, 2, 4, 8])
    disc_vars = disc.init(jax.random.PRNGKey(0), jnp.ones((1, 128, 128, 3)))
    logits = disc.apply(disc_vars, jnp.ones((1, 128, 128, 3), dtype=jnp.float32))
