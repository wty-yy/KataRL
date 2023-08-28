from typing import Any
from agents.models.base.jax_base import JaxModel
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

class MLP(nn.Module):
    output_ndim: int

    @nn.compact
    def __call__(self, inputs:jax.Array):  # BUG3: 网络复杂程度对模型最终稳定性影响非常大，简单的模型可以稳定最终的步数
        x = nn.Dense(120, name='Dense1')(inputs)
        x = nn.relu(x)
        x = nn.Dense(84, name='Dense2')(x)
        x = nn.relu(x)
        # x = nn.Dense(64, name='Dense3')(x)
        # x = nn.relu(x)
        outputs = nn.Dense(self.output_ndim, name='Output')(x)
        return outputs

class Model(JaxModel):
    
    def __init__(self, name='dqn-model', seed=1, lr=0.00025, load_name=None, load_id=None, input_shape=None, output_ndim=None, verbose=True, **kwargs):
        super().__init__(name, seed, lr, load_name, load_id, input_shape, output_ndim, verbose, **kwargs)

    def set_seed(self):
        self.key = jax.random.PRNGKey(seed=self.seed)
        self.key, self.model_key = jax.random.split(self.key)

    def build_model(self):
        model = MLP(output_ndim=self.output_ndim)
        self.state = TrainState.create(
            apply_fn=model.apply,
            params=model.init(self.model_key, jnp.empty(self.input_shape)),
            tx=optax.adam(learning_rate=self.lr)
        )
        return model
    