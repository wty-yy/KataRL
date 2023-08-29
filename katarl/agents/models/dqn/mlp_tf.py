from typing import NamedTuple
from katarl.agents.models.base.tf_base import TFModel

import tensorflow as tf
keras = tf.keras
layers = keras.layers

class Model(TFModel):
    
    def __init__(self, name='dqn-model', input_shape=None, output_ndim=None, args: NamedTuple = None):
        super().__init__(name, input_shape, output_ndim, args)

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

