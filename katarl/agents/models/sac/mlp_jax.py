from katarl.agents.models.base.base_jax import JaxModel

from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax

class MLP(nn.Module):
    output_ndim: int

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(128, name="Dense1")(inputs)
        x = nn.relu(x)
        x = nn.Dense(128, name="Dense2")(inputs)
        x = nn.relu(x)
        return nn.Dense(self.output_ndim, name="Outputs")(x)
    
class Model(JaxModel):

    def __init__(
            self, name='model', input_shape=None, output_ndim=None,
            args: NamedTuple = None,
            is_policy_model: bool = False, seed_delta: int = 0):
        self.is_policy_model, self.seed_delta = is_policy_model, seed_delta
        super().__init__(name, input_shape, output_ndim, args)
    
    def build_model(self):
        self.learning_rate = self.args.learning_rate_p if self.is_policy_model else self.args.learning_rate_q
        model = MLP(self.output_ndim)
        def linear_schedule(count):
            frac = 1. - count / self.args.total_timesteps
            return self.learning_rate * frac
        self.state = TrainState.create(
            apply_fn=jax.jit(model.apply),
            params=model.init(jax.random.PRNGKey(self.args.seed+self.seed_delta), jnp.empty(self.input_shape)),
            tx=optax.chain(
                # optax.clip_by_global_norm(self.args.max_grad_clip_norm),
                optax.inject_hyperparams(optax.adam)(
                    learning_rate=linear_schedule if self.args.flag_anneal_lr else self.learning_rate, eps=self.args.EPS
                )
            )
        )
        return model