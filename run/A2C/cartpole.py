from agents.A2C import A2C
from agents.models import Model, keras
from envs.gym_env import GymEnv

import tensorflow as tf
keras = tf.keras
layers = keras.layers

class MLP(Model):

    def __init__(
            self, lr=0.001, load_id=None, verbose=True, name='model',
            is_value_model=False, **kwargs
        ):
        self.is_value_model = is_value_model
        super().__init__(lr, load_id, verbose, name, **kwargs)
    
    def build_model(self) -> Model:
        inputs = layers.Input(shape=(4,), name='State')
        x = layers.Dense(32, activation='relu', name='Dense1')(inputs)
        x = layers.Dense(32, activation='relu', name='Dense2')(x)
        if self.is_value_model:
            outputs = layers.Dense(1, name='State-Value')(x)
        else:  # is policy model
            outputs = layers.Dense(2, activation='softmax', name='Action-Proba')(x)
        return keras.Model(inputs, outputs, name=self.name)
    
    def build_optimizer(self, lr) -> keras.optimizers.Optimizer:
        return keras.optimizers.Adadelta(learning_rate=lr)

def A2C_cartpole():
    start_idx = 0
    N = 3
    for idx in range(start_idx, start_idx + N):
        print(f"{idx}/{N}:")
        a2c = A2C(
            agent_name='A2C',
            value_model=MLP(name='value-model', is_value_model=True),
            policy_model=MLP(name='policy-model', is_value_model=False),
            env=GymEnv(name='CartPole-v1', render_mode='rgb_array'),
            verbose=False, agent_id=idx, episodes=1000
        )
        a2c.train()
