import jax
import jax.numpy as jnp

import flax.linen as nn
import flaxmodels as fm

from typing import Sequence

# https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py


class ScalingLayer(nn.Module):

    @nn.compact
    def __call__(self, x):
        shift = jnp.array([-.030, -.088, -.188])[None, None, None, :]
        scale = jnp.array([.458, .448, .450])[None, None, None, :]
        return (x - shift) / scale


class LPIPS(nn.Module):
    channels: Sequence[int]

    def setup(self):
        self.scaling = ScalingLayer()
        self.lin_layers = [
            nn.Conv(features=1, kernel_size=(1, 1), use_bias=False, name=f'lin{i}')
            for i in range(len(self.channels))
        ]
        self.vgg = fm.vgg.VGG16(pretrained="imagenet", include_head=False)
        vgg_vars = self.vgg.init(jax.random.PRNGKey(0), jnp.ones((2, 128, 128, 3)))
        self.vgg_params = {'params': vgg_vars['params']}

    def __call__(self, real_x, fake_x):
        real_x = (real_x + 1) / 2
        fake_x = (fake_x + 1) / 2

        real_x = self.scaling(real_x)
        fake_x = self.scaling(fake_x)

        feats_real = self.extract_features(real_x)
        feats_fake = self.extract_features(fake_x)

        diffs = []
        for f_real, f_fake, lin in zip(feats_real, feats_fake, self.lin_layers):
            f_real = get_call_output(f_real)
            f_fake = get_call_output(f_fake)

            diff = (norm_tensor(f_real) - norm_tensor(f_fake)) ** 2

            diff = lin(diff)
            diffs.append(spatial_average(diff))

        return jnp.sum(jnp.array(diffs))  # [N, 1, 1, 1]

    def extract_features(self, x):
        outputs, intermediates = self.vgg.apply(
            self.vgg_params,
            x,
            capture_intermediates=True,
            mutable=['intermediates'],
        )
        intermediates = intermediates["intermediates"]
        return [
            intermediates['conv1_2'],  # conv1_2
            intermediates['conv2_2'],  # conv2_2
            intermediates['conv3_3'],  # conv3_3
            intermediates['conv4_3'],  # conv4_3
            intermediates['conv5_3'],  # conv5_3
        ]


def norm_tensor(x, eps=1e-10):
    norm_factor = jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True))
    return x / (norm_factor + eps)


def spatial_average(x):
    return jnp.mean(x, axis=(1, 2), keepdims=True)


def get_call_output(x):
    return x['__call__'][0]


if __name__ == '__main__':
    lpips = LPIPS([64, 128, 256, 512, 512])
    params = lpips.init(jax.random.PRNGKey(0), jnp.ones((2, 128, 128, 3)), jnp.ones((2, 128, 128, 3)))
