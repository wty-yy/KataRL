from agents.DQN import DQN
from agents.models import Model
from envs.gym_env import GymEnv

import tensorflow as tf
keras = tf.keras
layers = keras.layers

class MLP(Model):

    def __init__(self, lr=1e-3, load_id=None, verbose=True, **kwargs):
        super().__init__(lr, load_id, verbose, **kwargs)

    def build_model(self):
        inputs = layers.Input(shape=(4,), name='State')
        x = layers.Dense(32, activation='relu', name='Dense1')(inputs)
        x = layers.Dense(32, activation='relu', name='Dense2')(x)
        outputs = layers.Dense(2, name='Q_Value')(x)
        return keras.Model(inputs, outputs)
    
    def build_optimizer(self, lr):
        return keras.optimizers.Adam(learning_rate=lr)

# DQN-hyper-args
args = {
    "batch_size": [1],
    "memory_size": 1e6,
    "start_fit_size": 1e4,
    "episodes": 1000
}

# DQN test on cartpole
def DQN_cartpole():
    start_idx = 0
    N = 3
    for batch_size in args['batch_size']:
        for idx in range(start_idx, start_idx + N):
            print(f"{idx}/{N}:")
            dqn = DQN(
                agent_name=f'DQN-{batch_size}',
                model=MLP(),
                env=GymEnv(name="CartPole-v1", render_mode="rgb_array"),
                verbose=False, agent_id=idx, episodes=1000, load_id=None,
                batch_size=batch_size
            )
            dqn.train()

if  __name__ == '__main__':
    DQN_test()