import sys
import yaml
import os

import torch
import numpy as np

import jax
import jax.numpy as jnp
from flax import serialization
from torchvision.utils import save_image
from transformer import GPTConfig, GPT
from vqgan import VQGAN


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def generate_samples(rng, vqgan, vqgan_params, gpt, gpt_params, seq_len, sos_token, tempature=1.0, num_samples=8):
    def make_predict_fn(*, apply_fn):
        def predict_fn(params, sequences):
            logits = apply_fn(params, sequences, train=False)
            return logits

        return jax.pmap(predict_fn, axis_name='batch', donate_argnums=())

    def shard(x):
        n, *s = x.shape
        return x.reshape((num_devices, n // num_devices, *s))

    def unshard(x):
        d, b, *s = x.shape
        return x.reshape((d * b, *s))

    devices = jax.local_devices()
    num_devices = len(devices)
    replicate = lambda tree: jax.device_put_replicated(tree, devices)
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    predict_fn = make_predict_fn(apply_fn=gpt.apply)

    params_repl = replicate(gpt_params)
    rng, sample_rng = jax.random.split(rng, 2)

    sequences = jnp.ones((num_samples, 1), dtype=jnp.int32) * sos_token

    for i in range(seq_len):
        sequences = jax.tree_util.tree_map(lambda x: shard(np.array(x)), sequences)
        logits = predict_fn(params_repl, sequences)
        sequences = jax.tree_util.tree_map(lambda x: unshard(x), sequences)
        logits = jax.tree_util.tree_map(lambda x: unshard(x), logits)
        logits = logits[:, -1, :] / tempature  # [N, C]

        rng, sample_rng = jax.random.split(rng, 2)
        next_token = jax.random.categorical(sample_rng, logits)
        sequences = jnp.concatenate([sequences, next_token[:, None]], axis=1)

    sequences = sequences[:, 1:]
    sequences = sequences.reshape(8, 8, 8)
    decoded_images = vqgan.apply(vqgan_params, sequences, method=vqgan.decode)
    decoded_images = (decoded_images + 1.0) / 2.0

    return jnp.clip(decoded_images, 0.0, 1.0)


def main(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    model_config = config['model']
    vqgan_config = config['first_stage_config']

    seed = model_config['params']['seed']
    key = jax.random.PRNGKey(seed)

    model_params = config["model"]["params"]

    gpt_config = GPTConfig(
        vocab_size=model_params["vocab_size"],
        block_size=model_params["block_size"],
        n_layer=model_params["n_layer"],
        n_head=model_params["n_head"],
        n_embed=model_params["n_embed"],
        dropout=model_params["dropout"],
    )

    vqgan = VQGAN(**vqgan_config['params'])
    gpt = GPT(config=gpt_config)

    checkpoint_path = vqgan_config['checkpoint_path']
    transformer_c_path = model_config['checkpoint_path']
    vqgan_params = load_checkpoint(checkpoint_path, None)['ema_params']
    gpt_params = load_checkpoint(transformer_c_path, None)['params']

    seq_len = model_config['seq_len']
    sos_token = model_config['sos_token']

    x_gen = generate_samples(key, vqgan, vqgan_params, gpt, gpt_params, seq_len, sos_token)

    for i in range(x_gen.shape[0]):
        img = np.array(x_gen[i])

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        save_image(img, f'/content/drive/MyDrive/VQ-GAN/gen_images/generated_image{i}.png')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1])
