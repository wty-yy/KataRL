from katarl.agents.models.base.jax_base import JaxModel

from typing import NamedTuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax

class MLP(nn.Module):
    is_value_model: bool
    output_ndim: int

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(128, name='Dense1')(inputs)
        x = nn.relu(x)
        x = nn.Dense(128, name='Dense2')(x)
        x = nn.relu(x)
        if self.is_value_model:
            outputs = nn.Dense(1, name='State-Value')(x)
        else:
            outputs = nn.softmax(nn.Dense(self.output_ndim, name='Action-Proba')(x))
        return outputs
        
class Model(JaxModel):

    def __init__(self, name='model', input_shape=None, output_ndim=None, args: NamedTuple = None, is_value_model: bool = False):
        self.is_value_model = is_value_model
        super().__init__(name, input_shape, output_ndim, args)
    
    def build_model(self):
        self.learning_rate = self.args.learning_rate_v if self.is_value_model else self.args.learning_rate_p
        model = MLP(output_ndim=self.output_ndim, is_value_model=self.is_value_model)
        def linear_schedule(count):
            frac = 1. - count / self.args.total_timesteps
            return self.learning_rate * frac
        self.state = TrainState.create(
            apply_fn=jax.jit(model.apply),
            params=model.init(jax.random.PRNGKey(self.args.seed), jnp.empty(self.input_shape)),
            tx=optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if self.args.anneal_lr else self.learning_rate, eps=1e-5
            )
        )
        return model
    