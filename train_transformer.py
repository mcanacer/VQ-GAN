import sys
import os
import yaml

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import serialization
from dataset import HuggingFace
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from vqgan import VQGAN
from transformer import GPT, GPTConfig

import numpy as np


def save_checkpoint(path, state):
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def make_update_fn(*, vqgan_apply_fn, vqgan_method, gpt_apply_fn, optimizer, sos_token, pkeep, vocab_size):
    def update_fn(vqgan_params, gpt_params, opt_state, images, drop_key, bernoulli_key, randint_key):
        def loss_fn(params):
            indices = vqgan_apply_fn(
                vqgan_params,
                images,
                method=vqgan_method,
            )

            indices = indices.reshape(images.shape[0], -1)

            sos_tokens = jnp.ones((images.shape[0], 1), dtype=jnp.int32) * sos_token

            mask = jax.random.bernoulli(bernoulli_key, p=pkeep, shape=indices.shape)
            mask = mask.astype('int64')

            r_indices = jax.random.randint(randint_key, shape=indices.shape, minval=0, maxval=vocab_size)
            a_indices = mask * indices + (1 - mask) * r_indices

            a_indices = jnp.concatenate([sos_tokens, a_indices], axis=1)

            targets = indices

            logits = gpt_apply_fn(
                params,
                a_indices[:, :-1],
                train=True,
                rngs={'dropout': drop_key}
            )  # [N, T, C]

            loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(targets, vocab_size)).mean()

            return loss

        loss, grad = jax.value_and_grad(loss_fn)(gpt_params)

        loss, grad = jax.tree_util.tree_map(
            lambda x: jax.lax.pmean(x, axis_name='batch'),
            (loss, grad),
        )

        updates, opt_state = optimizer.update(grad, opt_state, gpt_params)
        params = optax.apply_updates(gpt_params, updates)
        return params, opt_state, loss

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=())


def main(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    model_config = config['model']
    vqgan_config = config['first_stage_config']
    dataset_params = config['dataset_params']
    wandb_config = config['wandb']

    seed = model_config['params']['seed']

    transform = transforms.Compose([
        transforms.Resize((dataset_params['img_size'], dataset_params['img_size'])),
        transforms.ToTensor(),  # Normalize [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale [-1, 1]
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Convert [C, H, W] to [H, W, C]
    ])

    train_dataset = HuggingFace(
        dataset=load_dataset("flwrlabs/celeba", split='train'),
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_params['batch_size'],
        shuffle=True,
        num_workers=dataset_params['num_workers'],
        drop_last=True,
    )

    model_params = config["model"]["params"]

    gpt_config = GPTConfig(
        vocab_size=model_params["vocab_size"],
        block_size=model_params["block_size"],
        n_layer=model_params["n_layer"],
        n_head=model_params["n_head"],
        n_embed=model_params["n_embed"],
        dropout=model_params["dropout"],
    )

    model = VQGAN(**vqgan_config['params'])
    gpt = GPT(config=gpt_config)
    optimizer = optax.chain(optax.adam(learning_rate=1e-5))
    epochs = model_config['epochs']

    run = wandb.init(
        project=wandb_config['project'],
        name=wandb_config['name'],
        reinit=True,
        config=config
    )

    checkpoint_path = vqgan_config['checkpoint_path']
    transformer_c_path = model_config['checkpoint_path']
    vqgan_params = load_checkpoint(checkpoint_path, None)['ema_params']

    key = jax.random.PRNGKey(seed)
    key, params_key, drop_key = jax.random.split(key, 3)

    gpt_params = gpt.init({'params': params_key, 'dropout': drop_key}, jnp.ones((2, 64), dtype=jnp.int32),
                          train=False)

    opt_state = optimizer.init(gpt_params)

    devices = jax.local_devices()
    replicate = lambda tree: jax.device_put_replicated(tree, devices)
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    update_fn = make_update_fn(
        vqgan_apply_fn=model.apply,
        vqgan_method=model.encode,
        gpt_apply_fn=gpt.apply,
        optimizer=optimizer,
        sos_token=model_config['sos_token'],
        pkeep=model_config['pkeep'],
        vocab_size=model_config['vocab_size'],
    )

    vqgan_params_repl = replicate(vqgan_params)
    gpt_params_repl = replicate(gpt_params)
    opt_state_repl = replicate(opt_state)

    del vqgan_params
    del gpt_params
    del opt_state

    num_devices = jax.local_device_count()

    state_template = {
        "params": unreplicate(gpt_params_repl),
        "opt_state": unreplicate(opt_state_repl),
        "epoch": 0,
    }

    loaded_state = load_checkpoint(transformer_c_path, state_template)
    start_epoch = 0
    if loaded_state:
        gpt_params_repl = replicate(loaded_state['params'])
        opt_state_repl = replicate(loaded_state['opt_state'])
        start_epoch = loaded_state['epoch'] + 1

    def shard(x):
        n, *s = x.shape
        return np.reshape(x, (num_devices, n // num_devices, *s))

    def unshard(inputs):
        num_devices, batch_size, *shape = inputs.shape
        return jnp.reshape(inputs, (num_devices * batch_size, *shape))

    for epoch in range(start_epoch, epochs):
        for images in train_loader:
            key, drop_key, bernoulli_key, randint_key = jax.random.split(key, 4)
            images = jax.tree_util.tree_map(lambda x: shard(np.array(x)), images)

            drop_keys = jax.random.split(drop_key, num_devices)
            bernoulli_keys = jax.random.split(bernoulli_key, num_devices)
            randint_keys = jax.random.split(randint_key, num_devices)

            (
                gpt_params_repl,
                opt_state_repl,
                loss,
            ) = update_fn(
                vqgan_params_repl,
                gpt_params_repl,
                opt_state_repl,
                images,
                drop_keys,
                bernoulli_keys,
                randint_keys,
            )

            loss = unreplicate(loss)

            run.log({
                "total_loss": loss,
                "epoch": epoch})

        save_checkpoint(transformer_c_path, {
            "params": unreplicate(gpt_params_repl),
            "opt_state": unreplicate(opt_state_repl),
            "epoch": epoch,
        })


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1])
