from katarl.agents.constants import PATH
from katarl.agents.models.base import BaseModel
from typing import NamedTuple
import torch
from torch import nn

class TorchModel(BaseModel):
    model: nn.Module
    optimizer: torch.optim.Optimizer
    step: int = 0

    def __init__(self, name='model', input_shape=None, output_ndim=None, args: NamedTuple = None):
        super().__init__(name, input_shape, output_ndim, args)

    def __call__(self, X):
        return self.model(X)
    
    def load_weights(self):
        if self.load_path is None: return
        print(f"Load weight from '{self.load_path.absolute()}'")
        self.model.load_state_dict(torch.load(str(self.load_path)))
    
    def save_weights(self):
        path = PATH.CHECKPOINTS.joinpath(f"{self.name}-{self.save_id:04}.pth")
        torch.save(self.model.state_dict(), str(path))
        self.save_id += 1
