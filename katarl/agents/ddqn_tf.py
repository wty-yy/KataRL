# -*- coding: utf-8 -*-
'''
@File    : dqn.py
@Time    : 2023/08/26 12:02:07
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : None
2023.8.26. 在DQN基础上加入target network

'''

if __name__ == '__main__':
    pass

from typing import NamedTuple
from tensorboardX import SummaryWriter
from agents import BaseAgent
import agents.constants.dqn.ddqn as const
from agents.models.base import BaseModel
from envs.gym_env import GymEnv
from utils.logs import Logs, MeanMetric
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import time

from envs import Env
keras = tf.keras

def init_logs() -> Logs:
    return Logs(
        init_logs = {
            'episode_step': MeanMetric(),
            'episode_return': MeanMetric(),
            'q_value': MeanMetric(),
            'loss': MeanMetric(),
        },
        folder2name = {
            'charts': ['episode_step', 'episode_return'],
            'metrics': ['q_value', 'loss']
        }
    )

class WeightsCopy(keras.optimizers.Optimizer):
    
    def __init__(self, tau=1, **kwargs):
        super().__init__(name='copy', **kwargs)
        self._learning_rate = 0
        self.tau = tf.Variable(tau, dtype='float32', trainable=False)
    
    def update_step(self, target, variable):
        variable.assign(self.tau * target + (1-self.tau) * variable)
    
class MemoryCache:
    """
    save the memory cache of (S,A,R,S',T):
        type name      |  shape             |  type
        ---------------|--------------------|---------
        S(state)       |  env.state_shape   |  float32
        A(action)      |  env.action_shape  |  int32
        R(reward)      |  (1,)              |  int32
        S'(next state) |  env.state_shape   |  float32
        T(terminal)    |  (1,)              |  bool
    """

    def __init__(self, state_shape, action_shape, memory_size):
        self.state_shape, self.action_shape = state_shape, action_shape
        self.count = 0
        self.total = memory_size
        # use np.ndarray could sample by indexs easily
        self.s = np.zeros([self.total, *self.state_shape], dtype='float32')
        self.a = np.zeros([self.total], dtype='int32')
        self.r = np.zeros([self.total], dtype='float32')
        self.s_ = np.zeros([self.total, *self.state_shape], dtype='float32')
        self.t = np.zeros([self.total], dtype='bool')
        self.memory = [self.s, self.a, self.r, self.s_, self.t]
    
    def update(self, item:tuple):  # (S,A,R,S',T)
        start = self.count % self.total
        N = item[0].shape[0]  # actor_N
        for value, array in zip(item, self.memory):
            len1 = min(self.total - start, N)
            len2 = N - len1
            array[start:start+len1] = value[:len1]
            if len2 != 0:
                array[:len2] = value[-len2:]
        self.count += N
    
    def sample(self, num=1):
        size = min(self.count, self.total)
        indexs = np.random.choice(size, num)
        return [array[indexs] for array in self.memory]

class Agent(BaseAgent):

    def __init__(
            self,
            agent_name: str = None,
            env: Env = None,
            models: list[BaseModel] = None,
            writer: SummaryWriter = None,
            args: NamedTuple = None,
            # hyper-parameters
            model: BaseModel = None,
        ):
        models = [model]
        super().__init__(agent_name, env, models, writer, args)
        np.random.seed(self.args.seed)
        self.model, self.logs = model, init_logs()
        self.loss_fn = keras.losses.MeanSquaredError()
        self.memory = MemoryCache(env.state_shape, env.action_shape, self.args.memory_size)

        self.weights_copy = WeightsCopy(tau=self.args.tau)
        self.target_model = self.model.build_model()
    
    @tf.function
    def update_target_model(self):
        self.weights_copy.apply_gradients(
            zip(self.model.get_trainable_weights(), self.target_model.trainable_weights)
        )
    
    def update_epsilon(self):
        self.epsilon = max(self.args.epsilon_max + self.args.slope * self.global_step, self.args.epsilon_min)
    
    def train(self):
        state = self.env.reset()
        self.start_time, self.global_step = time.time(), 0
        for i in tqdm(range(self.args.num_iters)):
            self.logs.reset()
            self.update_epsilon()
            action = self.act(state)
            state_, reward, terminal = self.env.step(action)
            self.remember(state, action, reward, state_, terminal)
            self.global_step += self.args.num_envs

            if self.global_step > self.args.start_fit_size:
                if self.global_step % self.args.train_frequency < self.args.num_envs:
                    loss, q_value = self.fit()
                    self.logs.update(['q_value', 'loss'], [q_value, loss])
                    
                if self.global_step % self.args.target_model_update_frequency < self.args.num_envs:
                    self.update_target_model()

            state = state_
            self.logs.update(
                ['episode_step', 'episode_return'],
                [self.env.get_terminal_length(), self.env.get_terminal_reward()]
            )
            # if episode end, then we plot it
            self.logs.writer_tensorboard(self.writer, self.global_step, drops=['q_value', 'loss'])

            if (i+1) % self.args.write_logs_frequency == 0:
                self.write_tensorboard()
            if (i+1) % (self.args.num_iters // self.args.num_model_save) < self.args.num_envs or i == self.args.num_iters - 1:
                print(f"Save weights at global step:", self.global_step)
                self.model.save_weights()

    def evaluate(self):
        state = self.env.reset()
        self.start_time, self.global_step = time.time(), 0
        self.epsilon = 0
        for i in tqdm(range(self.args.num_iters)):
            self.logs.reset()
            action = self.act(state)
            state_, _, _ = self.env.step(action)
            self.global_step += self.args.num_envs
            state = state_
            self.logs.update(
                ['episode_step', 'episode_return'],
                [self.env.get_terminal_length(), self.env.get_terminal_reward()]
            )
            if i % self.args.write_logs_frequency == 0:
                self.write_tensorboard()
        
    @tf.function
    def predict(self, state):
        return self.model(state)

    @tf.function
    def target_predict(self, state):
        return self.target_model(state)

    def act(self, state):  # epsilon choice policy
        rand = np.random.rand()
        if rand <= self.epsilon:
            return np.random.randint(0, self.env.action_ndim, self.args.num_envs)
        else:
            q_pred = self.predict(state)
            action = np.argmax(q_pred, axis=-1)
            return action

    def remember(self, state, action, reward, state_, terminal):
        self.memory.update((state, action, reward, state_, terminal))

    @tf.function
    def train_step(self, X, y):
        with tf.GradientTape() as tape:
            q_state = self.model(X)
            loss = self.loss_fn(y, q_state)
        grads = tape.gradient(loss, self.model.get_trainable_weights())
        self.model.apply_gradients(grads)
        return loss, tf.reduce_mean(q_state)

    def fit(self):
        s, a, r, s_, t = self.memory.sample(self.args.batch_size)
        r, t = r.squeeze(), t.squeeze()

        td_target = r + self.args.gamma * np.max(self.target_predict(s_), axis=-1) * (1-t)
        q_state = self.model(s)
        q_target = q_state.numpy()
        q_target[np.arange(self.args.batch_size), a] = td_target

        loss, q_value = self.train_step(s, q_target)
        return tf.reduce_mean(loss), tf.reduce_mean(q_value)
    
    def write_tensorboard(self):
        self.logs.writer_tensorboard(self.writer, self.global_step, drops=['episode_step', 'episode_return'])
        self.writer.add_scalar('charts/epsilon', self.epsilon, self.global_step)
        self.writer.add_scalar(
            'charts/SPS_avg',
            int(self.global_step / (time.time() - self.start_time)),
            self.global_step
        )
    