from agents.constants import PATH
from agents.models.base import BaseModel
import tensorflow as tf
keras = tf.keras

class TFModel(BaseModel):

    model: keras.Model

    def __init__(self, name='model', seed=1, lr=0.00025, load_name=None, load_id=None, input_shape=None, output_ndim=None, verbose=True, **kwargs):
        super().__init__(name, seed, lr, load_name, load_id, input_shape, output_ndim, verbose, **kwargs)
    
    def load_weights(self):
        if self.load_path is None: return
        print(f"Load weight from '{self.load_path.absolute()}'")
        self.model.load_weights(self.load_path)
    
    def save_weights(self):
        path = PATH.CHECKPOINTS.joinpath(f"{self.name}-{self.save_id:04}")
        self.model.save_weights(path)
        self.save_id += 1