from katarl.agents.constants import PATH
from katarl.agents.models.base import BaseModel

import jax, jax.numpy as jnp
from flax.training.train_state import TrainState
import flax
import flax.linen as nn
from typing import NamedTuple

class JaxModel(BaseModel):

    model: nn.Module
    state: TrainState

    def __init__(self, name='model', input_shape=None, output_ndim=None, args: NamedTuple = None):
        super().__init__(name, input_shape, output_ndim, args)

    def load_weights(self):
        if self.load_path is None: return
        print(f"Load weight from '{self.load_path.absolute()}'")
        with open(self.load_path, 'rb') as file:
            self.state = flax.serialization.from_bytes(self.state, file.read())
    
    def save_weights(self):
        path = PATH.CHECKPOINTS.joinpath(f"{self.name}-{self.save_id:04}")
        with open(path, 'wb') as file:
            file.write(flax.serialization.to_bytes(self.state))
        self.save_id += 1
    
    def plot_model(self, path):
        print(self.model.tabulate(jax.random.PRNGKey(42), jnp.empty(self.input_shape)))
