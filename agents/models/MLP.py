import tensorflow as tf
from agents.models import Model, keras
keras = tf.keras
layers = keras.layers

class MLP(Model):

    def __init__(self, input_shape, load_id, **kwargs):
        super().__init__(input_shape, load_id, **kwargs)

    def build(self) -> Model:
        inputs = layers.Input(shape=self.input_shape, name='State')
        x = layers.Dense(32, activation='relu', name='Dense1')(inputs)
        x = layers.Dense(32, activation='relu', name='Dense2')(x)
        outputs = layers.Dense(2, name='Q_Value')(x)
        return keras.Model(inputs, outputs)

    