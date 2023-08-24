from agents.models import BaseModel
import tensorflow as tf
keras = tf.keras
layers = keras.layers

class Model(BaseModel):

    def __init__(
            self, lr=0.001,
            load_name=None, load_id=None,
            verbose=True, name='model', is_value_model=False,
            input_shape=None, output_ndim=None,
            **kwargs
        ):
        self.is_value_model = is_value_model
        super().__init__(lr, load_name, load_id, verbose, name, input_shape, output_ndim, **kwargs)
    
    def build_model(self):
        inputs = layers.Input(shape=self.input_shape, name='State')
        x = layers.Dense(128, activation='relu', name='Dense1')(inputs)
        x = layers.Dense(64, activation='relu', name='Dense2')(x)
        x = layers.Dense(16, activation='relu', name='Dense3')(x)
        if self.is_value_model:
            outputs = layers.Dense(1, name='State-Value')(x)
        else:  # is policy model
            outputs = layers.Dense(
                self.output_ndim, activation='softmax', name='Action-Proba'
            )(x)
        return keras.Model(inputs, outputs, name=self.name)
    
    def build_optimizer(self, lr):
        return keras.optimizers.Adam(learning_rate=lr)