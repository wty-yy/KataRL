from typing import NamedTuple
from katarl.agents.models.base.base_tf import TFModel
import tensorflow as tf
keras = tf.keras
layers = keras.layers

class MySchedule(keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, init_lr, per, tot):
        self.init_lr, self.per, self.tot = init_lr, per, tot
        self.lr = tf.Variable(init_lr, dtype='float32', trainable=False)
    
    def __call__(self, count):
        count = tf.cast(count, 'float32')
        self.lr.assign((
            1 -
            (count // self.per) / self.tot
        ) * self.init_lr)
        return self.lr

class Model(TFModel):

    def __init__(self, name='model', input_shape=None, output_ndim=None, args: NamedTuple = None):
        super().__init__(name, input_shape, output_ndim, args)

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape, name='State')
        x = layers.Dense(128, activation='relu', name='Dense1')(inputs)
        x = layers.Dense(64, activation='relu', name='Dense2')(x)
        x = layers.Dense(16, activation='relu', name='Dense3')(x)
        value = layers.Dense(1, name='State-Value')(x)
        x = layers.Dense(128, activation='relu', name='Dense4')(inputs)
        x = layers.Dense(64, activation='relu', name='Dense5')(x)
        x = layers.Dense(16, activation='relu', name='Dense6')(x)
        proba = layers.Dense(self.output_ndim, activation='softmax', name='Action-Proba')(x)
        return keras.Model(inputs, [value, proba], name=self.name)

    def build_optimizer(self):
        if self.args.flag_anneal_lr:
            linear_schedule = MySchedule(
                init_lr=self.args.learning_rate,
                per=((self.args.data_size-1)//self.args.batch_size+1) * self.args.epochs,
                tot=self.args.num_iters
            )
            self.lr = linear_schedule
        else: self.lr = self.args.learning_rate
        return keras.optimizers.Adam(learning_rate=self.lr)
    
