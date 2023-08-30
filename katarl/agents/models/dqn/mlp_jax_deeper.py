from typing import NamedTuple
from katarl.agents.models.base.base_jax import JaxModel

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

class MLP(nn.Module):
    output_ndim: int

    @nn.compact
    def __call__(self, inputs:jax.Array):  # BUG3: 网络复杂程度对模型最终稳定性影响非常大，简单的模型可以稳定最终的步数
        x = nn.Dense(128, name='Dense1')(inputs)
        x = nn.relu(x)
        x = nn.Dense(64, name='Dense2')(x)
        x = nn.relu(x)
        x = nn.Dense(64, name='Dense3')(x)
        x = nn.relu(x)
        outputs = nn.Dense(self.output_ndim, name='Output')(x)
        return outputs

class Model(JaxModel):
    
    def __init__(self, name='dqn-model', input_shape=None, output_ndim=None, args: NamedTuple = None):
        super().__init__(name, input_shape, output_ndim, args)

    def set_seed(self):
        self.key = jax.random.PRNGKey(seed=self.args.seed)
        self.key, self.model_key = jax.random.split(self.key)

    def build_model(self):
        model = MLP(output_ndim=self.output_ndim)
        def linear_schedule(count):
            frac = 1. - count * self.args.train_frequency / (self.args.total_timesteps - self.args.start_fit_size)
            return self.args.learning_rate * frac
        self.state = TrainState.create(
            apply_fn=model.apply,
            params=model.init(self.model_key, jnp.empty(self.input_shape)),
            tx=optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if self.args.anneal_lr else self.args.learning_rate, eps=1e-5
            )
        )
        return model
    