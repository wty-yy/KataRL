# -*- coding: utf-8 -*-
'''
@File    : PPO.py
@Time    : 2023/08/08 13:51:09
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
8.8. 完成基本框架，但是效率低，且value值计算错误
8.9. 使用env.vector重写Actor，删除copy weights
'''

from agents import Agent
from agents.constants.PPO import *
from agents.models import Model
from envs import Env
from utils.logs_manager import Logs
from utils import make_onehot, sample_from_proba
from tqdm import tqdm
import tensorflow as tf
keras = tf.keras
import numpy as np

def get_logs() -> Logs:
    return Logs(
        init_logs={
            'frame': 0,
            'max_step': 0,
            'v_value': keras.metrics.Mean(name='v_value'),
            'loss': keras.metrics.Mean(name='loss')
        }
    )

def expand_dim(state):
    return tf.expand_dims(tf.constant(state, dtype='float32'), axis=0)

class Actor:

    def __init__(self, env:Env, model:keras.Model, gamma, lambda_, step_T):
        self.env, self.model, self.gamma, self.lambda_ = \
            env, model, gamma, lambda_
        self.T, self.N = step_T, self.env.num_envs
        self.state = self.env.reset()
        self.data_size = self.T * self.N
        
    @tf.function
    def pred(self, state):
        return self.model(state)

    def act(self):
        action_shape = self.env.action_shape
        state_shape = self.env.state_shape
        S, A, R, S_, T, AD, V, LP = \
            np.zeros(shape=(self.T, self.N) + state_shape, dtype='float32'), \
            np.zeros(shape=(self.T, self.N) + action_shape, dtype='int32'), \
            np.zeros(shape=(self.T, self.N), dtype='float32'), \
            np.zeros(shape=(self.T, self.N) + state_shape, dtype='float32'), \
            np.zeros(shape=(self.T, self.N), dtype='bool'), \
            np.zeros(shape=(self.T, self.N), dtype='float32'), \
            np.zeros(shape=(self.T, self.N), dtype='float32'), \
            np.zeros(shape=(self.T, self.N), dtype='float32')
        max_step = 0
        for step in range(self.T):
            v, proba = self.pred(self.state)
            V[step] = v.numpy().squeeze()
            action = sample_from_proba(proba.numpy())
            action_one_hot = make_onehot(action, depth=self.env.action_size).astype('bool')
            LP[step] = np.log(proba[action_one_hot])
            state_, reward, terminal = self.env.step(action)
            action = action.reshape(-1, 1)
            S[step], A[step], R[step], S_[step], T[step] = \
                self.state, action, reward, state_, terminal
            self.state = state_
            max_step = int(max(max_step, self.env.step_count.max()))
        v_last, _ = self.pred(self.state)
        v_last = v_last.numpy().reshape(1, self.N)
        # Calc Delta
        AD = R + self.gamma * np.r_[V[1:,:], v_last] * (~T) - V
        for i in reversed(range(self.T-1)):
            AD[i] += self.gamma * self.lambda_ * AD[i+1] * (~T[i])
        # Target state value
        V += AD
        S, A, AD, V, LP = \
            S.reshape(self.data_size, -1), \
            A.reshape(self.data_size, -1), \
            AD.reshape(self.data_size, -1), \
            V.reshape(self.data_size, -1), \
            LP.reshape(self.data_size, -1)
        ds = tf.data.Dataset.from_tensor_slices((S,A,AD,V,LP))
        return ds, max_step

class PPO(Agent):

    def __init__(
            self, env: Env = None,
            agent_name='PPO', agent_id=0,
            episodes=None, 
            model: Model = None,
            gamma=gamma, lambda_=lambda_, epsilon=epsilon,
            actor_N=actor_N, frames_M=frames_M, step_T=step_T,
            epochs=epochs, batch_size=batch_size,
            **kwargs):
        models = [model]
        super().__init__(env, agent_name, agent_id, episodes, models, **kwargs)
        self.model, self.gamma, self.lambda_, self.epsilon, \
            self.actor_N, self.frames_M, self.step_T, \
            self.epochs, self.batch_size = \
            model, gamma, lambda_, epsilon,\
            actor_N, frames_M, step_T, epochs, batch_size
        self.data_size = actor_N * step_T
        self.logs = get_logs()
        # init actors
        self.actor = Actor(
            self.env, self.model,
            self.gamma, self.lambda_, self.step_T
        )
    
    def train(self):
        num_iters = (self.frames_M-1) // self.data_size + 1
        for i in tqdm(range(num_iters)):
            self.logs.reset()
            max_step = self.fit()
            frame = (i+1) * self.data_size
            self.logs.update(['frame', 'max_step'], [frame, max_step])
            self.update_history()
            if (i + 1) % 10 == 0:
                self.model.save_weights()
    
    def evaluate(self):
        num_iters = (self.frames_M-1) // self.data_size + 1
        for i in tqdm(range(num_iters)):
            self.logs.reset()
            _, max_step = self.actor.act()
            frame = (i+1) * self.data_size
            self.logs.update(['frame', 'max_step'], [frame, max_step])
            self.update_history()

    def fit(self):
        ds = None
        ds, max_step = self.actor.act()
        ds = ds.shuffle(1000).batch(self.batch_size)
        for _ in range(self.epochs):
            for s, a, ad, v, logpi in ds:
                # print(f"{ad=}")
                a = make_onehot(a.numpy(), depth=self.env.action_size).astype('bool')
                value, loss = self.train_step(s, a, ad, v, logpi)
                self.logs.update(['v_value', 'loss'], [value, loss])
        return max_step
    
    @tf.function
    def train_step(self, s, a, ad, v, logpi):
        mean, var = tf.nn.moments(ad, axes=[0])
        ad = (ad - mean) / (var + EPS)
        with tf.GradientTape() as tape:
            v_s, p_s = self.model(s)
            loss_v = tf.reduce_mean(tf.square(v_s-v)/2)
            logpi_now = tf.math.log(tf.reshape(p_s[a], (-1, 1)))
            lograte = logpi_now - logpi
            rate = tf.math.exp(lograte)
            loss_p_clip = tf.reduce_mean(
                tf.minimum(
                    rate*ad,
                    tf.clip_by_value(
                        rate,
                        clip_value_min=1-self.epsilon,
                        clip_value_max=1+self.epsilon
                    )*ad
                )
            )
            loss = loss_v - loss_p_clip
        grads = tape.gradient(loss, self.model.get_trainable_weights())
        self.model.apply_gradients(grads)
        return tf.reduce_mean(v_s), loss
    
    def update_history(self):
        self.best_episode.update_best(
            now=self.logs.logs['max_step'], logs=self.logs.to_dict()
        )
        self.history.update_dict(self.logs.to_dict())
