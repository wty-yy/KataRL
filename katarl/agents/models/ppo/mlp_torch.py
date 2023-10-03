from typing import NamedTuple
from katarl.agents.models.base.base_torch import TorchModel
import torch
from torch import nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MLP(nn.Module):
    def __init__(self, input_ndim: int, output_ndim: int):
        super().__init__()
        self.linear_value = nn.Sequential(
            layer_init(nn.Linear(input_ndim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )
        self.linear_proba = nn.Sequential(
            layer_init(nn.Linear(input_ndim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128), std=1.0),
            nn.ReLU(),
            layer_init(nn.Linear(128, output_ndim),std=0.01)
        )

    def __call__(self, x):
        value = self.linear_value(x)
        proba = self.linear_proba(x)
        return (value, proba)

class Model(TorchModel):

    def __init__(self, name='ppo-model', input_shape=None, output_ndim=None, args: NamedTuple = None):
        super().__init__(name, input_shape, output_ndim, args)
    
    def set_seed(self):
        torch.manual_seed(self.args.seed)

    def build_model(self):
        model = MLP(input_ndim=self.input_shape[0], output_ndim=self.output_ndim)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate, eps=1e-5)
        return model
