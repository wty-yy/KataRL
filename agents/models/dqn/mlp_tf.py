import tensorflow as tf
keras = tf.keras
layers = keras.layers
from agents.models.base import BaseModel

class Model(BaseModel):
    
    def __init__(self, name='dqn-model', seed=1, lr=0.00025, load_name=None, load_id=None, input_shape=None, output_ndim=None, verbose=True, **kwargs):
        super().__init__(name, seed, lr, load_name, load_id, input_shape, output_ndim, verbose, **kwargs)

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape, name='State')
        x = layers.Dense(120, activation='relu', name='Dense1')(inputs)
        x = layers.Dense(84, activation='relu', name='Dense2')(x)
        # x = layers.Dense(16, activation='relu', name='Dense3')(x)
        outputs = layers.Dense(self.output_ndim, name='Q_Value')(x)
        return keras.Model(inputs, outputs)
    
    def build_optimizer(self, lr):
        return keras.optimizers.Adam(learning_rate=lr)
    
    def set_seed(self):
        tf.random.set_seed(self.seed)

    def plot_model(self, path):
        print(f"plot model struct png at '{path.absolute()}'")
        keras.utils.plot_model(self.model, to_file=path, show_shapes=True)
