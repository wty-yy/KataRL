# -*- coding: utf-8 -*-
'''
@File    : A2C/cartpole.py
@Time    : 2023/08/02 12:49:26
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : Advantage actor-critic on Cartpole environment.
Hyper parameters test:
lr_v: lr of value_model
lr_p: lr of policy_model

Deeper model

'''

if __name__ == '__main__':
    pass

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
        x = layers.Dense(128, activation='relu', name='Dense1')(inputs)
        x = layers.Dense(64, activation='relu', name='Dense2')(x)
        x = layers.Dense(16, activation='relu', name='Dense3')(x)
        if self.is_value_model:
            outputs = layers.Dense(1, name='State-Value')(x)
        else:  # is policy model
            outputs = layers.Dense(
                2, activation='softmax', name='Action-Proba'
            )(x)
        return keras.Model(inputs, outputs, name=self.name)
    
    def build_optimizer(self, lr) -> keras.optimizers.Optimizer:
        return keras.optimizers.Adam(learning_rate=lr)

def A2C_cartpole_train1(name="A2C-Adam-v3-p5-r20-deeper"):
    start_idx = 0
    N = 20
    for idx in range(start_idx, start_idx + N):
        print(f"{idx}/{N}:")
        a2c = A2C(
            agent_name=name,
            value_model=MLP(
                name='value-model', is_value_model=True,
                lr=1e-3
            ),
            policy_model=MLP(
                name='policy-model', is_value_model=False,
                lr=1e-5
            ),
            env=GymEnv(name='CartPole-v1', render_mode='rgb_array'),
            verbose=False, agent_id=idx, episodes=1000
        )
        a2c.train()

def A2C_cartpole_eval1(agent_id, load_id, episodes=10):
    a2c = A2C(
        agent_name='A2C',
        value_model=MLP(
            name='value-model', is_value_model=True,
            load_id=load_id,
        ),
        policy_model=MLP(
            name='policy-model', is_value_model=False,
            load_id=load_id,
        ),
        env=GymEnv(name='CartPole-v1', render_mode='rgb_array'),
        verbose=True, agent_id=agent_id, episodes=episodes
    )
    a2c.evaluate()
    