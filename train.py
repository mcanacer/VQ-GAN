import sys
import importlib
import os
import wandb

import jax
import jax.numpy as jnp
import optax
import operator
from flax import serialization

import numpy as np


def get_everything(config_path, args):
    module_path = config_path.replace('/', '.').replace('.py', '')
    module = importlib.import_module(module_path, package=None)
    return module.everything(args)


def save_checkpoint(path, state):
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def adopt_weight(step, threshold, value=0.0):
    return jnp.where(step < threshold, value, 1.0)


def ema_update(ema_params, new_params, decay):
    return jax.tree_util.tree_map(
        lambda e, p: decay * e + (1.0 - decay) * p,
        ema_params,
        new_params
    )


def sigmoid_cross_entropy_with_logits(logits, labels):
    zeros = jnp.zeros_like(logits, dtype=logits.dtype)
    condition = (logits >= zeros)
    relu_logits = jnp.where(condition, logits, zeros)
    neg_abs_logits = jnp.where(condition, -logits, logits)
    return relu_logits - logits * labels + jnp.log1p(jnp.exp(neg_abs_logits))


def make_generator_update_fn(*, vqgan_apply_fn, vqgan_optimizer, disc_apply_fn, lpips_apply_fn, ema_decay, disc_start):
    def update_fn(vqgan_params, vqgan_opt_state, disc_params, lpips_params, images, ema_params, global_step):
        def loss_fn(params):
            images_recon, quantized_latents, commitment_loss, embedding_loss, enc_indices = vqgan_apply_fn(params, images)

            disc_factor = adopt_weight(global_step, disc_start)
            disc_fake = disc_apply_fn(disc_params, images_recon)

            recon_loss = jnp.mean(jnp.abs(images_recon - images))
            perceptual_loss = 0.1 * lpips_apply_fn(lpips_params, images, images_recon)

            g_loss = disc_factor * 0.1 * sigmoid_cross_entropy_with_logits(disc_fake, jnp.ones_like(disc_fake)).mean()

            losses = recon_loss, commitment_loss, embedding_loss, perceptual_loss, g_loss
            loss = jax.tree_util.tree_reduce(operator.add, losses)

            return loss, (losses, images_recon, enc_indices)

        ((loss, (losses, fake_images, enc_indices)), grad) = jax.value_and_grad(loss_fn, has_aux=True)(vqgan_params)

        loss, losses, grad = jax.tree_util.tree_map(
            lambda x: jax.lax.pmean(x, axis_name='batch'),
            (loss, losses, grad),
        )

        updates, opt_state = vqgan_optimizer.update(grad, vqgan_opt_state, vqgan_params)
        new_params = optax.apply_updates(vqgan_params, updates)
        new_ema_params = ema_update(ema_params, new_params, decay=ema_decay)

        return new_params, opt_state, new_ema_params, loss, losses, fake_images, enc_indices

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=())


def make_disc_update_fn(*, apply_fn, optimizer, disc_start):
    def update_fn(params, opt_state, real_images, fake_images, global_step):
        def loss_fn(params):
            disc_real = apply_fn(params, real_images)
            disc_fake = apply_fn(params, fake_images)

            loss_real = sigmoid_cross_entropy_with_logits(disc_real, jnp.ones_like(disc_real)).mean()
            loss_fake = sigmoid_cross_entropy_with_logits(disc_fake, jnp.zeros_like(disc_fake)).mean()

            disc_factor = adopt_weight(global_step, disc_start)
            return disc_factor * jnp.mean(loss_real + loss_fake)

        loss, grad = jax.value_and_grad(loss_fn)(params)
        loss, grad = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), (loss, grad))
        updates, opt_state = optimizer.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=())


def main(config_path, args):
    evy = get_everything(config_path, args)

    seed, train_loader = evy['seed'], evy['train_loader']
    inputs = next(iter(train_loader))
    model, disc, lpips = evy['vqgan'], evy['disc'], evy['lpips']
    vqgan_optimizer, disc_optimizer, epochs = evy['vqgan_optimizer'], evy['disc_optimizer'], evy['epochs']
    run, checkpoint_path, disc_start = evy['run'], evy['checkpoint_path'], evy['disc_start']

    key = jax.random.PRNGKey(seed)
    vqgan_params = model.init(key, inputs)
    disc_params = disc.init(key, inputs)
    lpips_params = lpips.init(key, jnp.ones((2, 128, 128, 3)), jnp.ones((2, 128, 128, 3)))

    vqgan_opt_state = vqgan_optimizer.init(vqgan_params)
    disc_opt_state = disc_optimizer.init(disc_params)

    replicate = lambda tree: jax.device_put_replicated(tree, jax.local_devices())
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    ema_params = vqgan_params
    ema_params_repl = replicate(ema_params)

    generator_update_fn = make_generator_update_fn(
        vqgan_apply_fn=model.apply,
        vqgan_optimizer=vqgan_optimizer,
        disc_apply_fn=disc.apply,
        lpips_apply_fn=lpips.apply,
        ema_decay=0.9999,
        disc_start=disc_start,
    )

    disc_update_fn = make_disc_update_fn(apply_fn=disc.apply, optimizer=disc_optimizer, disc_start=disc_start)

    vqgan_params_repl = replicate(vqgan_params)
    vqgan_opt_state_repl = replicate(vqgan_opt_state)
    disc_params_repl = replicate(disc_params)
    disc_opt_state_repl = replicate(disc_opt_state)
    lpips_params_repl = replicate(lpips_params)

    state_template = {
        "params": unreplicate(vqgan_params_repl),
        "opt_state": unreplicate(vqgan_opt_state_repl),
        "ema_params": unreplicate(ema_params_repl),
        'lpips_params': unreplicate(lpips_params_repl),
        'disc_params': unreplicate(disc_params_repl),
        'disc_opt_state': unreplicate(disc_opt_state_repl),
        "epoch": 0,
    }

    loaded_state = load_checkpoint(checkpoint_path, state_template)
    start_epoch = 0
    if loaded_state:
        vqgan_params_repl = replicate(loaded_state['params'])
        vqgan_opt_state_repl = replicate(loaded_state['opt_state'])
        ema_params_repl = replicate(loaded_state['ema_params'])
        lpips_params_repl = replicate(loaded_state['lpips_params'])
        disc_params_repl = replicate(loaded_state['disc_params'])
        disc_opt_state_repl = replicate(loaded_state['disc_opt_state'])
        start_epoch = loaded_state['epoch'] + 1

    def shard(x):
        n, *s = x.shape
        return np.reshape(x, (jax.local_device_count(), n // jax.local_device_count(), *s))

    def unshard(x):
        ndev, bs, *s = x.shape
        return jnp.reshape(x, (ndev * bs, *s))

    global_step = 0
    global_step_repl = jnp.array([global_step] * jax.local_device_count())

    for epoch in range(start_epoch, epochs):
        for step, images in enumerate(train_loader):
            images = jax.tree_util.tree_map(lambda x: shard(np.array(x)), images)

            (
                vqgan_params_repl,
                vqgan_opt_state_repl,
                ema_params_repl,
                vq_loss,
                vq_losses,
                fake_images,
                num_codes
            ) = generator_update_fn(
                vqgan_params_repl,
                vqgan_opt_state_repl,
                disc_params_repl,
                lpips_params_repl,
                images,
                ema_params_repl,
                global_step_repl
            )

            (
                disc_params_repl,
                disc_opt_state_repl,
                disc_loss,
            ) = disc_update_fn(
                disc_params_repl,
                disc_opt_state_repl,
                images,
                fake_images,
                global_step_repl
            )

            global_step += 1
            global_step_repl = jnp.array([global_step] * jax.local_device_count())

            loss = unreplicate(vq_loss)
            losses = [jnp.asarray(x) for x in unreplicate(vq_losses)]
            recon_loss, commitment_loss, embedding_loss, perceptual_loss, g_loss = losses

            if global_step % 1000 == 0:
                import matplotlib.pyplot as plt
                import io
                from PIL import Image as PILImage

                def to_numpy_img(img):
                    img = (img + 1) / 2
                    img = np.clip(np.array(img), 0.0, 1.0)
                    return (img * 255).astype(np.uint8)

                real = to_numpy_img(unshard(images)[0])
                recon = to_numpy_img(unshard(fake_images)[0])

                fig, axs = plt.subplots(1, 2, figsize=(4, 2))
                axs[0].imshow(real)
                axs[0].set_title("Original")
                axs[0].axis("off")
                axs[1].imshow(recon)
                axs[1].set_title("Reconstruction")
                axs[1].axis("off")
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plt.close(fig)

                run.log({"reconstruction": wandb.Image(PILImage.open(buf))}, step=global_step)

            run.log({
                "reconstruct_loss": recon_loss,
                "commitment_loss": commitment_loss,
                "embedding_loss": embedding_loss,
                "perceptual_loss": perceptual_loss,
                "g_loss": g_loss,
                'disc_loss': unreplicate(disc_loss),
                "total_loss": loss,
                "epoch": epoch,
                'num_codes': jnp.unique(unshard(num_codes)).size,  # number of unique codebook
            })

        save_checkpoint(checkpoint_path, {
            "params": unreplicate(vqgan_params_repl),
            "opt_state": unreplicate(vqgan_opt_state_repl),
            "ema_params": unreplicate(ema_params_repl),
            'lpips_params': unreplicate(lpips_params_repl),
            'disc_params': unreplicate(disc_params_repl),
            'disc_opt_state': unreplicate(disc_opt_state_repl),
            "epoch": epoch,
        })


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1], sys.argv[2:])
