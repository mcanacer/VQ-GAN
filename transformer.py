"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import jax
import jax.numpy as jnp

import flax.linen as nn

import math


def truncated_normal(stddev, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        return jax.random.truncated_normal(
            key=key, lower=-2, upper=2, shape=shape, dtype=dtype) * stddev
    return init


class GPTConfig:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, mask, layer_past=None, train=True):
        N, T, E = x.shape

        k = nn.Dense(self.config.n_embed, kernel_init=truncated_normal(0.02), name='key')(x)  # [N, T, E]
        q = nn.Dense(self.config.n_embed, kernel_init=truncated_normal(0.02), name='query')(x)  # [N, T, E]
        v = nn.Dense(self.config.n_embed, kernel_init=truncated_normal(0.02), name='value')(x)  # [N, T, E]

        k = k.reshape((N, T, self.config.n_head, E // self.config.n_head)).swapaxes(1, 2)  # [N, NH, T, HD]
        q = q.reshape((N, T, self.config.n_head, E // self.config.n_head)).swapaxes(1, 2)  # [N, NH, T, HD]
        v = v.reshape((N, T, self.config.n_head, E // self.config.n_head)).swapaxes(1, 2)  # [N, NH, T, HD]

        present = jnp.stack([k, v], axis=0)
        if layer_past is not None:
            past_key, past_value = layer_past
            k = jnp.concatenate([past_key, k], axis=-2)
            v = jnp.concatenate([past_value, v], axis=-2)

        att = (q @ k.swapaxes(2, 3)) * (1.0 / math.sqrt(k.shape[-1]))  # [N, NH, T, T]
        if layer_past is None:
            att = jnp.where(mask[:, :, :T, :T] == 0, -jnp.inf, att)

        att = jax.nn.softmax(att, axis=-1)
        att = nn.Dropout(self.config.dropout)(att, deterministic=not train)
        y = att @ v  # [N, NH, T, HD]
        y = y.swapaxes(1, 2).reshape(N, T, E)  # [N, T, E]

        y = nn.Dense(self.config.n_embed, kernel_init=truncated_normal(0.02), name='proj')(y)  # [N, T, E]
        y = nn.Dropout(self.config.dropout)(y, deterministic=not train)
        return y, present


class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Dense(4 * self.config.n_embed, kernel_init=truncated_normal(0.02),)(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(self.config.n_embed, kernel_init=truncated_normal(0.02),)(x)
        x = nn.Dropout(self.config.dropout)(x, deterministic=not train)
        return x


class Block(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, mask, layer_past=None, return_present=False, train=True):
        if return_present:
            assert not train
        residual = x
        x = nn.LayerNorm()(x)
        attn, present = CausalSelfAttention(self.config)(x, mask=mask,
                                                         layer_past=layer_past, train=train)

        x = residual + attn
        residual = x

        x = nn.LayerNorm()(x)
        x = residual + MLP(self.config)(x, train=train)
        if layer_past is not None or return_present:
            return x, present
        return x


class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, idx, embeddings=None, train=True):  # idx: [N, T]
        idx = idx.astype('int32')
        N, T = idx.shape
        token_embeddings = nn.Embed(self.config.vocab_size, self.config.n_embed,
                                    embedding_init=truncated_normal(0.02))(idx)  # [N, T, E]

        if embeddings is not None:
            token_embeddings = jnp.concatenate([embeddings, token_embeddings], axis=1)

        t = token_embeddings.shape[1]
        assert t <= self.config.block_size, "Cannot forward, model block size is exhausted."
        pos_embed = self.param("pos_embedding", truncated_normal(0.02),
                               (1, self.config.block_size, self.config.n_embed))
        x = nn.Dropout(self.config.dropout)(token_embeddings + pos_embed[:, :t], deterministic=not train)

        mask = jnp.tril(jnp.ones((self.config.block_size, self.config.block_size))).astype(jnp.bool_)
        mask = jnp.expand_dims(mask, axis=(0, 1))  # [1, 1, T, T]

        for _ in range(self.config.n_layer):
            x = Block(self.config)(x, mask=mask, train=train)

        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.config.vocab_size)(x)

        return logits


if __name__ == '__main__':
    config = GPTConfig(vocab_size=1024, block_size=512, n_layer=24, n_head=16, n_embed=1024, dropout=0.0)
    gpt = GPT(config)
    key = jax.random.PRNGKey(0)
    params_key, drop_key = jax.random.split(key, 2)
    params = gpt.init({'params': params_key, 'dropout': drop_key}, jnp.ones((2, 512), dtype=jnp.int32), train=False)
