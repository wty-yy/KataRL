from typing import NamedTuple
from agents.models.base import BaseModel
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

class Model(BaseModel):

    def __init__(
            self, lr=None, 
            load_name=None, load_id=None,
            verbose=True, name='model',
            input_shape=None, output_ndim=None,
            args : NamedTuple = None,
            **kwargs
        ):
        if args.flag_anneal_lr:
            linear_schedule = MySchedule(
                init_lr=args.init_lr,
                per=((args.data_size-1)//args.batch_size+1) * args.epochs,
                tot=args.iter_nums
            )
            lr = linear_schedule
        else: lr = args.init_lr
        super().__init__(lr, load_name, load_id, verbose, name, input_shape, output_ndim, **kwargs)

    def build_model(self):
        inputs = layers.Input(shape=(210,160,3), name='State')
        x = layers.Resizing(84, 84, name='Resize')(inputs)
        # # Block1
        # x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', name='Conv1')(x)
        # x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', name='Conv2')(x)
        # x = layers.MaxPool2D(2, strides=2, name='Pool1')(x)
        # # Block2
        # x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', name='Conv3')(x)
        # x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', name='Conv4')(x)
        # x = layers.MaxPool2D(2, strides=2, name='Pool2')(x)
        # # Block3
        # x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu', name='Conv5')(x)
        # x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu', name='Conv6')(x)
        # x = layers.MaxPool2D(2, strides=2, name='Pool3')(x)
        # # Block4
        # x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu', name='Conv7')(x)
        # x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu', name='Conv8')(x)
        # x = layers.MaxPool2D(2, strides=2, name='Pool4')(x)  # 8x8x512
        x = layers.Conv2D(32, kernel_size=8, strides=4, activation='relu', name='Conv1')(x)
        x = layers.Conv2D(64, kernel_size=4, strides=2, activation='relu', name='Conv2')(x)
        x = layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', name='Conv3')(x)
        feature = layers.Flatten(name='Feature')(x)
        # State Value FC
        x = layers.Dense(128, activation='relu', name='Dense1')(feature)
        x = layers.Dense(64, activation='relu', name='Dense2')(x)
        x = layers.Dense(16, activation='relu', name='Dense3')(x)
        value = layers.Dense(1, name='State-Value')(x)
        # Action Proba FC
        x = layers.Dense(128, activation='relu', name='Dense4')(feature)
        x = layers.Dense(64, activation='relu', name='Dense5')(x)
        x = layers.Dense(16, activation='relu', name='Dense6')(x)
        proba = layers.Dense(4, activation='softmax', name='Action-Proba')(x)
        return keras.Model(inputs, [value, proba], name=self.name)

    def build_optimizer(self, lr):
        return keras.optimizers.Adam(learning_rate=lr)

    def __call__(self, X):
        X = tf.cast(X, 'float32')
        X = X / 255.
        return super().__call__(X)
        