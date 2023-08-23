import tensorflow as tf
keras = tf.keras
layers = keras.layers
from agents.models import BasicModel

class Model(BasicModel):

    def __init__(self, lr=0.001, load_name=None, load_id=None, verbose=True, name='DQN_MLP', input_shape=None, output_ndim=None, **kwargs):
        super().__init__(lr, load_name, load_id, verbose, name, input_shape, output_ndim, **kwargs)

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape, name='State')
        x = layers.Dense(32, activation='relu', name='Dense1')(inputs)
        x = layers.Dense(32, activation='relu', name='Dense2')(x)
        outputs = layers.Dense(self.output_ndim, name='Q_Value')(x)
        return keras.Model(inputs, outputs)
    
    def build_optimizer(self, lr):
        return keras.optimizers.Adam(learning_rate=lr)
