from agents.PPO import PPO
import agents.constants.PPO.breakout as const
from agents.models import Model, keras
from envs.gym_env import GymEnv
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

linear_schedule = MySchedule(
    init_lr=const.init_lr,
    per=const.data_size // const.batch_size * const.epochs,
    tot=const.iter_nums
)

class CNN(Model):

    def __init__(
            self, lr=linear_schedule if const.flag_anneal_lr else const.init_lr,
            load_id=None, verbose=True,
            name='model', **kwargs
        ):
        super().__init__(lr, load_id, verbose, name, **kwargs)

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

def PPO_breakout_train(start_idx=0, load_ids:dict={}):
    N = 30
    for idx in range(start_idx, start_idx + N):
        print(f"{idx}/{N}:")
        load_id = load_ids.get(idx)
        ppo = PPO(
            env=GymEnv(name='ALE/Breakout-v5', num_envs=const.actor_N),
            agent_name='PPO-breakout',
            model=CNN(load_id=load_id),
            agent_id=idx, **const.__dict__
        )
        # try:
        ppo.train()
        # except:
        #     print("GG: continue next training", idx+1)

def PPO_breakout_eval(agent_id, load_id, frames_M=int(1e4)):
    args = const.__dict__
    args['frames_M'] = frames_M
    ppo = PPO(
        env=GymEnv(
            name='Breakout-v4', num_envs=const.actor_N,
            capture_video=True
        ),
        agent_name='PPO-breakout',
        model=CNN(load_id=load_id),
        agent_id=agent_id,
        **args
    )
    ppo.evaluate()