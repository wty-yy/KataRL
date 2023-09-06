from typing import NamedTuple
from katarl.agents.models.base.base_jax import JaxModel

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
from flax.training.train_state import TrainState
import optax

class MLP(nn.Module):
    output_ndim: int

    @nn.compact
    def __call__(self, inputs):  # (batch, 4, 84, 84)
        x = jnp.transpose(inputs, (0, 2, 3, 1)) / 255.0
        x = nn.relu(nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(scale=jnp.sqrt(2)),
            bias_init=constant(0.0),
            name='Conv1'
        )(x))
        x = nn.relu(nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(scale=jnp.sqrt(2)),
            bias_init=constant(0.0),
            name='Conv2'
        )(x))
        x = nn.relu(nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(scale=jnp.sqrt(2)),
            bias_init=constant(0.0),
            name='Conv3'
        )(x))
        feature = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(128, name='Dense1')(feature))
        x = nn.relu(nn.Dense(128, name='Dense2')(x))
        value = nn.Dense(1, name='State-Value')(x)
        x = nn.relu(nn.Dense(128, name='Dense3')(feature))
        x = nn.relu(nn.Dense(128, name='Dense4')(x))
        proba = nn.Dense(self.output_ndim, name='Action-Proba')(x)
        return (value, proba)

class Model(JaxModel):

    def __init__(self, name='ppo-model', input_shape=None, output_ndim=None, args: NamedTuple = None):
        super().__init__(name, input_shape, output_ndim, args)

    def build_model(self):
        model = MLP(output_ndim=self.output_ndim)
        def linear_schedule(count):
            per = self.args.data_size // self.args.batch_size * self.args.epochs
            frac = 1. - count // per / self.args.num_iters
            return self.args.learning_rate * frac
        self.state = TrainState.create(
            apply_fn=model.apply,
            params=model.init(jax.random.PRNGKey(self.args.seed), jnp.empty(self.input_shape)),
            tx=optax.chain(
                optax.clip_by_global_norm(self.args.max_grad_clip_norm),
                optax.inject_hyperparams(optax.adam)(
                    learning_rate=linear_schedule if self.args.flag_anneal_lr else self.args.learning_rate, eps=1e-5
                )
            )
        )
        return model
