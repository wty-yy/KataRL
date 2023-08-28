from agents.constants import PATH
from agents.models.base import BaseModel
from flax.training.train_state import TrainState
import flax
import flax.linen as nn

class JaxModel(BaseModel):

    model: nn.Module
    state: TrainState

    def __init__(self, name='model', seed=1, lr=0.00025, load_name=None, load_id=None, input_shape=None, output_ndim=None, verbose=True, **kwargs):
        super().__init__(name, seed, lr, load_name, load_id, input_shape, output_ndim, verbose, **kwargs)
    
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
