from katarl.agents.models.base.tf_base import TFModel

from typing import NamedTuple
import tensorflow as tf
keras = tf.keras
layers = keras.layers

class Model(TFModel):

    def __init__(self, name='model', input_shape=None, output_ndim=None, args: NamedTuple = None, is_value_model: bool = False):
        self.is_value_model = is_value_model
        super().__init__(name, input_shape, output_ndim, args)
    
    def build_model(self):
        inputs = layers.Input(shape=self.input_shape, name='State')
        x = layers.Dense(128, activation='relu', name='Dense1')(inputs)
        x = layers.Dense(128, activation='relu', name='Dense2')(x)
        if self.is_value_model:
            outputs = layers.Dense(1, name='State-Value')(x)
        else:  # is policy model
            outputs = layers.Dense(
                self.output_ndim, activation='softmax', name='Action-Proba'
            )(x)
        return keras.Model(inputs, outputs, name=self.name)
    
    def build_optimizer(self):
        return keras.optimizers.Adam(learning_rate=self.args.learning_rate_v if self.is_value_model else self.args.learning_rate_p)
    