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

from tensorboardX import SummaryWriter
from agents import BaseAgent
import agents.constants.ppo as const
from agents.models.base import BaseModel
from envs import Env
from utils.logs import Logs, MeanMetric
from utils import make_onehot, sample_from_proba
from tqdm import tqdm
import tensorflow as tf
keras = tf.keras
import numpy as np

chart_log_name = ['step', 'max_reward', 'learning_rate']
def get_logs() -> Logs:
    return Logs(
        init_logs={
            'frame': 0,
            'step': MeanMetric(),
            'v_value': MeanMetric(),
            'loss_p': MeanMetric(),
            'loss_v': MeanMetric(),
            'loss_ent': MeanMetric(),
            'advantage': MeanMetric(),
            'max_reward': 0,
            'learning_rate': 0,
        }
    )

class Actor:

    def __init__(self, env:Env, model:BaseModel, gamma, lambda_, step_T):
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
            terminal_rewards += self.env.get_terminal_rewrad()
            # max_step = int(max(max_step, self.env.step_count.max()))
        v_last, _ = self.pred(self.state)
        v_last = v_last.numpy().reshape(1, self.N)
        # Calc Delta
        AD = R + self.gamma * np.r_[V[1:,:], v_last] * (~T) - V
        for i in reversed(range(self.T-1)):
            AD[i] += self.gamma * self.lambda_ * AD[i+1] * (~T[i])
        S, A, AD, V, LP = \
            S.reshape(self.data_size, *state_shape), \
            A.reshape(self.data_size, *action_shape), \
            AD.reshape(self.data_size, 1), \
            V.reshape(self.data_size, 1), \
            LP.reshape(self.data_size, 1)
        ds = tf.data.Dataset.from_tensor_slices((S,A,AD,V,LP))
        return ds, terminal_steps, terminal_rewards

class PPO(BaseAgent):

    def __init__(
            self, agent_name=None,
            env: Env = None,
            model: BaseModel = None,
            writer: SummaryWriter = None,
            # hyperparameters
            gamma=const.gamma, lambda_=const.lambda_,
            epsilon=const.epsilon, v_epsilon=const.v_epsilon,
            actor_N=const.actor_N, frames_M=const.frames_M, step_T=const.step_T,
            epochs=const.epochs, batch_size=const.batch_size,
            coef_value=const.coef_value,
            coef_entropy=const.coef_entropy,
            flag_ad_normal=const.flag_ad_normal,
            flag_clip_value=const.flag_clip_value,
            max_clip_norm=const.max_clip_norm,
            **kwargs
        ):
        models = [model]
        super().__init__(agent_name, env, models, writer, **kwargs)
        self.model, self.gamma, self.lambda_, \
            self.epsilon, self.v_epsilon, \
            self.actor_N, self.frames_M, self.step_T, \
            self.epochs, self.batch_size, \
            self.coef_value, self.coef_entropy, \
            self.flag_clip_value, \
            self.flag_ad_normal, self.max_clip_norm = \
            model, gamma, lambda_, epsilon, v_epsilon, \
            actor_N, frames_M, step_T, epochs, batch_size, \
            coef_value, coef_entropy, \
            flag_clip_value, flag_ad_normal, max_clip_norm
        self.data_size = actor_N * step_T
        self.logs = get_logs()
        self.actor = Actor(
            self.env, self.model,
            self.gamma, self.lambda_, self.step_T
        )
    
    def train(self):
        iter_nums = (self.frames_M-1) // self.data_size + 1
        for i in tqdm(range(iter_nums)):
            self.logs.reset()
            steps, rewards = self.fit()
            frame = (i+1) * self.data_size
            max_reward = 0 if len(rewards) == 0 else int(np.max(rewards))
            if isinstance(self.model.lr, float):
                now_lr = self.model.lr
            else:
                now_lr = float(self.model.lr.lr.numpy())
            self.logs.update(
                ['frame', 'step', 'max_reward', 'learning_rate'],
                [frame, steps, max_reward, now_lr]
            )
            self.write_tensorboard()
            if (i + 1) % 10 == 0:
                self.model.save_weights()
    
    def evaluate(self):
        num_iters = (self.frames_M-1) // self.data_size + 1
        for i in tqdm(range(num_iters)):
            self.logs.reset()
            _, steps, rewards = self.actor.act()
            frame = (i+1) * self.data_size
            max_reward = 0 if len(rewards) == 0 else int(np.max(rewards))
            self.logs.update(
                ['frame', 'step', 'max_reward'],
                [frame, steps, max_reward])
            self.write_tensorboard()

    def fit(self):
        ds, steps, rewards = self.actor.act()
        ds = ds.shuffle(1000).batch(self.batch_size)
        self.ds = ds
        for _ in range(self.epochs):
            for s, a, ad, v, logpi in ds:
                a = make_onehot(a.numpy(), depth=self.env.action_ndim).astype('bool')
                value, loss_p, loss_v, loss_ent, loss = \
                    self.train_step(s, a, ad, v, logpi)
                self.logs.update(
                    ['v_value', 'loss_p', 'loss_v', 'loss_ent', \
                     'advantage'],
                    [value, loss_p, loss_v, loss_ent, \
                     tf.reduce_mean(ad)]
                )
        return steps, rewards
    
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

            if self.flag_ad_normal:
                mean, var = tf.nn.moments(ad, axes=[0])
                ad = (ad - mean) / (var + const.EPS)

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
            loss_entropy = -tf.reduce_mean(  # Fix loss Nan: -0*log(0)=0
                tf.reduce_sum(p_now*tf.math.log(p_now+const.EPS), axis=1)
            )
            loss = - loss_p_clip \
                   + self.coef_value * loss_v \
                   - self.coef_entropy * loss_entropy
        grads = tape.gradient(loss, self.model.get_trainable_weights())
        # Fix loss Nan
        grads = tf.clip_by_global_norm(grads, clip_norm=self.max_clip_norm)[0]
        self.model.apply_gradients(grads)
        return tf.reduce_mean(v_now), loss_p_clip, loss_v, loss_entropy, loss
    
    def write_tensorboard(self):
        frame = self.logs.logs['frame']
        d = self.logs.to_dict(drops=['frame'])
        for key, value in d.items():
            if key in chart_log_name: name = 'charts/' + key
            else: name = 'metrics/' + key
            if value is not None:
                self.writer.add_scalar(name, value, frame)
        self.writer.add_scalar(
            'charts/SPS',
            int(((self.data_size-1)//self.batch_size+1)*self.epochs/self.logs.get_time_length()),
            frame
        )
