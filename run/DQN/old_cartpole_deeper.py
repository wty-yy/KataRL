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
        x = layers.Dense(128, activation='relu', name='Dense1')(inputs)
        x = layers.Dense(64, activation='relu', name='Dense2')(x)
        x = layers.Dense(16, activation='relu', name='Dense3')(x)
        outputs = layers.Dense(2, name='Q_Value')(x)
        return keras.Model(inputs, outputs, name='Q_Value')
    
    def build_optimizer(self, lr):
        return keras.optimizers.Adam(learning_rate=lr)

# DQN-hyper-args
args = {
    "batch_size": [4],
    "memory_size": 1e6,
    "start_fit_size": 1e4,
    "episodes": 1000
}

# DQN test on cartpole
def DQN_cartpole_train1(name="DQN-4-deeper"):
    start_idx = 0
    N = 30
    for batch_size in args['batch_size']:
        for idx in range(start_idx, start_idx + N):
            print(f"{idx}/{N}:")
            dqn = DQN(
                agent_name=name,
                model=MLP(),
                env=GymEnv(name="CartPole-v1"),
                verbose=False, agent_id=idx, episodes=1000,
                batch_size=batch_size
            )
            dqn.train()

def DQN_cartpole_eval1(agent_id, load_id, batch_size, episodes=10):
    dqn = DQN(
        agent_name=f'DQN-{batch_size}',
        model=MLP(load_id=load_id),
        env=GymEnv(name="CartPole-v1", capture_video=True),
        verbose=True, agent_id=agent_id, episodes=episodes,
        batch_size=batch_size
    )
    dqn.epsilon = 0
    dqn.evaluate()

# call in DQN_cartpole() '/main.py'