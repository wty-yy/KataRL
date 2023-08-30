import os
from typing import NamedTuple
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from katarl.agents.constants import PATH
from katarl.agents.models.base import BaseModel

import tensorflow as tf
keras = tf.keras

class TFModel(BaseModel):

    model: keras.Model
    optimizer: keras.optimizers.Optimizer

    def __init__(self, name='model', input_shape=None, output_ndim=None, args: NamedTuple = None):
        super().__init__(name, input_shape, output_ndim, args)
        self.optimizer = self.build_optimizer()

    def load_weights(self):
        if self.load_path is None: return
        print(f"Load weight from '{self.load_path.absolute()}'")
        self.model.load_weights(self.load_path)
    
    def save_weights(self):
        path = PATH.CHECKPOINTS.joinpath(f"{self.name}-{self.save_id:04}")
        self.model.save_weights(path)
        self.save_id += 1

    def plot_model(self, path):
        print(f"plot model struct png at '{path.absolute()}'")
        keras.utils.plot_model(self.model, to_file=path, show_shapes=True)
    
    def __call__(self, X):
        return self.model(X)

    def get_trainable_weights(self):
        return self.model.trainable_weights
    
    def apply_gradients(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    
    def build_optimizer(self):
        pass

    def set_seed(self):
        tf.random.set_seed(self.args.seed)
