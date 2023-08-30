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
2023.8.12. 修正max step，只在失败时进行记录，完成cartpole调参
2023.8.13. 加入新环境Breakout
2023.8.14,15. DEBUG 解决训练Breakout训练1.5e6总帧数时出现loss=None的问题
通过加入lr linear schedule和grad global norm clip，修改value_loss_coef=0.5解决
使用环境Breakout-v4最大达到170分
2023.8.16. 使用环境ALE/Breakout-v5再次训练，并使用更小的clip_epsilon=0.1
'''
from katarl.agents import BaseAgent
from katarl.agents.models.base.base_tf import TFModel
from katarl.envs import Env
from katarl.utils.logs import Logs, MeanMetric
from katarl.utils import make_onehot, sample_from_proba

from typing import NamedTuple
from tensorboardX import SummaryWriter
from tqdm import tqdm
import tensorflow as tf
keras = tf.keras
import numpy as np
import time

def get_logs() -> Logs:
    return Logs(
        init_logs={
            'episode_step': MeanMetric(),
            'episode_return': MeanMetric(),
            'v_value': MeanMetric(),
            'loss_p': MeanMetric(),
            'loss_v': MeanMetric(),
            'loss_ent': MeanMetric(),
            'advantage': MeanMetric(),
            'max_reward': 0,
            'learning_rate': 0,
        },
        folder2name={
            'charts': ['episode_step', 'episode_return', 'max_reward', 'learning_rate'],
            'metrics': ['v_value', 'loss_p', 'loss_v', 'loss_ent', 'advantage']
        }
    )

class Actor:

    def __init__(self, env:Env, model:TFModel, args:NamedTuple):
        self.env, self.model, self.args = env, model, args
        self.T, self.N = self.args.num_steps, self.args.num_envs
        self.state = self.env.reset()
        
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
        terminal_steps, terminal_rewards = [], []
        for step in range(self.T):
            v, proba = self.pred(self.state)
            V[step] = v.numpy().squeeze()
            action = sample_from_proba(proba)  # check
            action_one_hot = make_onehot(action, depth=self.env.action_ndim).astype('bool')
            LP[step] = np.log(proba[action_one_hot])
            state_, reward, terminal = self.env.step(action)
            action = action.reshape(-1, 1)
            S[step], A[step], R[step], S_[step], T[step] = \
                self.state, action, reward, state_, terminal
            self.state = state_
            terminal_steps += self.env.get_terminal_steps()
            terminal_rewards += self.env.get_terminal_reward()
            # max_step = int(max(max_step, self.env.step_count.max()))
        v_last, _ = self.pred(self.state)
        v_last = v_last.numpy().reshape(1, self.N)
        # Calc Delta
        AD = R + self.args.gamma * np.r_[V[1:,:], v_last] * (~T) - V
        for i in reversed(range(self.T-1)):
            AD[i] += self.args.gamma * self.args.coef_gae * AD[i+1] * (~T[i])
        S, A, AD, V, LP = \
            S.reshape(self.args.data_size, *state_shape), \
            A.reshape(self.args.data_size, *action_shape), \
            AD.reshape(self.args.data_size, 1), \
            V.reshape(self.args.data_size, 1), \
            LP.reshape(self.args.data_size, 1)
        ds = tf.data.Dataset.from_tensor_slices((S,A,AD,V,LP))
        return ds, terminal_steps, terminal_rewards

class Agent(BaseAgent):

    def __init__(
            self,
            agent_name: str = None,
            env: Env = None,
            models: list[TFModel] = None,
            writer: SummaryWriter = None,
            args: NamedTuple = None,
            model: TFModel = None
        ):
        self.model, models = model, [model]
        super().__init__(agent_name, env, models, writer, args)
        self.logs = get_logs()
        self.actor = Actor(self.env, self.model, args)
    
    def train(self):
        self.start_time = time.time()
        for i in tqdm(range(self.args.num_iters)):
            self.logs.reset()
            steps, rewards = self.fit()
            self.global_step = (i+1) * self.args.data_size
            max_reward = 0 if len(rewards) == 0 else int(np.max(rewards))
            if isinstance(self.model.lr, float):
                now_lr = self.model.lr
            else: now_lr = float(self.model.lr.lr.numpy())
            self.logs.update(
                ['episode_step', 'episode_return', 'max_reward', 'learning_rate'],
                [np.mean(steps), np.mean(rewards), max_reward, now_lr]
            )
            self.write_tensorboard()
            if (i + 1) % 10 == 0:
                self.model.save_weights()
    
    def evaluate(self):
        self.start_time = time.time()
        for i in tqdm(range(self.args.num_iters)):
            self.logs.reset()
            _, steps, rewards = self.actor.act()
            self.global_step = (i+1) * self.args.data_size
            max_reward = 0 if len(rewards) == 0 else int(np.max(rewards))
            self.logs.update(
                ['episode_step', 'episode_return', 'max_reward'],
                [np.mean(steps), np.mean(rewards), max_reward]
            )
            self.write_tensorboard()

    def fit(self):
        ds, steps, rewards = self.actor.act()
        ds = ds.shuffle(1000).batch(self.args.batch_size)
        self.ds = ds
        for _ in range(self.args.epochs):
            for s, a, ad, v, logpi in ds:
                a = make_onehot(a.numpy(), depth=self.env.action_ndim).astype('bool')
                value, loss_p, loss_v, loss_ent = \
                    self.train_step(s, a, ad, v, logpi)
                self.logs.update(
                    ['v_value', 'loss_p', 'loss_v', 'loss_ent', \
                     'advantage'],
                    [value, loss_p, loss_v, loss_ent, \
                     tf.reduce_mean(ad)]
                )
        return steps, rewards
    
    @tf.function()
    def train_step(self, s, a, ad, v, logpi):
        with tf.GradientTape() as tape:
            v_now, p_now = self.model(s)
            loss_v = tf.square(v_now-v-ad)
            if self.args.flag_clip_value:
                loss_v_clip = tf.square(
                    tf.clip_by_value(
                        v_now - v,
                        clip_value_min=-self.args.v_epsilon,
                        clip_value_max=self.args.v_epsilon
                    )-ad
                )
                loss_v = tf.maximum(loss_v, loss_v_clip)
            loss_v = tf.reduce_mean(loss_v / 2)

            if self.args.flag_ad_normal:
                mean, var = tf.nn.moments(ad, axes=[0])
                ad = (ad - mean) / (var + self.args.EPS)

            logpi_now = tf.math.log(tf.reshape(p_now[a], (-1, 1)))
            lograte = logpi_now - logpi
            rate = tf.math.exp(lograte)
            loss_p_clip = tf.reduce_mean(
                tf.minimum(
                    rate*ad,
                    tf.clip_by_value(
                        rate,
                        clip_value_min=1-self.args.epsilon,
                        clip_value_max=1+self.args.epsilon
                    )*ad
                )
            )
            loss_entropy = -tf.reduce_mean(  # Fix loss Nan: -0*log(0)=0
                tf.reduce_sum(p_now*tf.math.log(p_now+self.args.EPS), axis=1)
            )
            loss = - loss_p_clip \
                   + self.args.coef_value * loss_v \
                   - self.args.coef_entropy * loss_entropy
        grads = tape.gradient(loss, self.model.get_trainable_weights())
        # Fix loss Nan
        grads = tf.clip_by_global_norm(grads, clip_norm=self.args.max_grad_clip_norm)[0]
        self.model.apply_gradients(grads)
        return tf.reduce_mean(v_now), loss_p_clip, loss_v, loss_entropy
    
    def write_tensorboard(self):
        self.logs.writer_tensorboard(self.writer, self.global_step)
        self.writer.add_scalar(
            'charts/SPS',
            int(((self.args.data_size-1)//self.args.batch_size+1)*self.args.epochs/self.logs.get_time_length()),
            self.global_step
        )
        self.writer.add_scalar(
            'charts/SPS_avg',
            int(self.global_step / (time.time() - self.start_time)),
            self.global_step
        )
