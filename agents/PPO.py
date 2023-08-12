# -*- coding: utf-8 -*-
'''
@File    : PPO.py
@Time    : 2023/08/08 13:51:09
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
2023.8.8. 完成基本框架，但是效率低，且value值计算错误
2023.8.9. 使用env.vector重写Actor，删除copy weights
2023.8.10. 完成PPO
2023.8.12. 修正max step，只在失败时进行记录
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
            'step': keras.metrics.Mean(name='step'),
            'v_value': keras.metrics.Mean(name='v_value'),
            'loss': keras.metrics.Mean(name='loss'),
            'loss_p': keras.metrics.Mean(name='loss_p'),
            'loss_v': keras.metrics.Mean(name='loss_v')
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
        terminal_steps = []
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
            terminal_steps += self.env.get_terminal_steps()
            # max_step = int(max(max_step, self.env.step_count.max()))
        v_last, _ = self.pred(self.state)
        v_last = v_last.numpy().reshape(1, self.N)
        # Calc Delta
        AD = R + self.gamma * np.r_[V[1:,:], v_last] * (~T) - V
        for i in reversed(range(self.T-1)):
            AD[i] += self.gamma * self.lambda_ * AD[i+1] * (~T[i])
        S, A, AD, V, LP = \
            S.reshape(self.data_size, -1), \
            A.reshape(self.data_size, -1), \
            AD.reshape(self.data_size, -1), \
            V.reshape(self.data_size, -1), \
            LP.reshape(self.data_size, -1)
        ds = tf.data.Dataset.from_tensor_slices((S,A,AD,V,LP))
        return ds, terminal_steps

class PPO(Agent):

    def __init__(
            self, env: Env = None,
            agent_name='PPO', agent_id=0,
            episodes=None, 
            model: Model = None,
            gamma=gamma, lambda_=lambda_,
            epsilon=epsilon, v_epsilon=v_epsilon,
            actor_N=actor_N, frames_M=frames_M, step_T=step_T,
            epochs=epochs, batch_size=batch_size,
            coef_value=coef_value,
            coef_entropy=coef_entropy,
            flag_ad_normal=flag_ad_normal,
            flag_clip_value=flag_clip_value,
            **kwargs):
        models = [model]
        super().__init__(env, agent_name, agent_id, episodes, models, **kwargs)
        self.model, self.gamma, self.lambda_, \
            self.epsilon, self.v_epsilon, \
            self.actor_N, self.frames_M, self.step_T, \
            self.epochs, self.batch_size, \
            self.coef_value, self.coef_entropy, \
            self.flag_clip_value, \
            self.flag_ad_normal = \
            model, gamma, lambda_, epsilon, v_epsilon, \
            actor_N, frames_M, step_T, epochs, batch_size, \
            coef_value, coef_entropy, \
            flag_clip_value, flag_ad_normal
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
            steps = self.fit()
            frame = (i+1) * self.data_size
            self.logs.update(['frame', 'step'], [frame, steps])
            self.update_history()
            if (i + 1) % 10 == 0:
                self.model.save_weights()
    
    def evaluate(self):
        num_iters = (self.frames_M-1) // self.data_size + 1
        for i in tqdm(range(num_iters)):
            self.logs.reset()
            _, steps = self.actor.act()
            frame = (i+1) * self.data_size
            self.logs.update(['frame', 'step'], [frame, steps])
            self.update_history()

    def fit(self):
        ds = None
        ds, steps = self.actor.act()
        ds = ds.shuffle(1000).batch(self.batch_size)
        for _ in range(self.epochs):
            for s, a, ad, v, logpi in ds:
                a = make_onehot(a.numpy(), depth=self.env.action_size).astype('bool')
                value, loss, loss_p, loss_v = self.train_step(s, a, ad, v, logpi)
                self.logs.update(
                    ['v_value', 'loss', 'loss_p', 'loss_v'],
                    [value, loss, loss_p, loss_v]
                )
        return steps
    
    @tf.function
    def train_step(self, s, a, ad, v, logpi):
        with tf.GradientTape() as tape:
            v_now, p_now = self.model(s)
            loss_v = tf.square(v_now-v-ad)
            if self.flag_clip_value:
                loss_v_clip = tf.square(
                    tf.clip_by_value(
                        v_now - v,
                        clip_value_min=-self.v_epsilon,
                        clip_value_max=self.v_epsilon
                    )-ad
                )
                loss_v = tf.maximum(loss_v, loss_v_clip)
            loss_v = tf.reduce_mean(loss_v / 2)
            # tf.print("v_now-v:", tf.reduce_mean(v_now - v))
            # tf.print("ad", tf.reduce_mean(ad))

            if self.flag_ad_normal:
                mean, var = tf.nn.moments(ad, axes=[0])
                ad = (ad - mean) / (var + EPS)

            logpi_now = tf.math.log(tf.reshape(p_now[a], (-1, 1)))
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
            loss_entropy = -tf.reduce_mean(
                tf.reduce_sum(p_now*tf.math.log(p_now), axis=1)
            )
            loss = - loss_p_clip \
                   + self.coef_value * loss_v \
                   - self.coef_entropy * loss_entropy
        grads = tape.gradient(loss, self.model.get_trainable_weights())
        self.model.apply_gradients(grads)
        # tf.print("clip:", loss_p_clip)
        # tf.print("value:", loss_v)
        # tf.print("entropy:", loss_entropy)
        return tf.reduce_mean(v_now), loss, loss_p_clip, loss_v
    
    def update_history(self):
        # self.best_episode.update_best(
        #     now=self.logs.logs['step'], logs=self.logs.to_dict()
        # )
        self.history.update_dict(self.logs.to_dict())
