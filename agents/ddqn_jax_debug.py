# -*- coding: utf-8 -*-
'''
@File    : dqn.py
@Time    : 2023/08/26 12:02:07
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : None
2023.8.26. 用Jax实现DDQN
2023.8.27. 成功实现，5e5时间步，num-envs=1用时7mins, num-envs=4用时4mins

'''

if __name__ == '__main__':
    pass

from tensorboardX import SummaryWriter
from agents import BaseAgent
import agents.constants.dqn.ddqn as const
from agents.models.base import BaseModel
from utils.logs import Logs, MeanMetric
from tqdm import tqdm
import time
from envs import Env
import jax
import numpy as np
import jax.numpy as jnp
from flax.struct import dataclass
from flax.training.train_state import TrainState
from functools import partial

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

# @dataclass
# class Storage:
#     s: jax.Array
#     a: jax.Array
#     r: jax.Array
#     s_: jax.Array
#     t: jax.Array
# 
# class MemoryCache1:
#     """
#     save the memory cache of (S,A,R,S',T):
#         type name      |  shape             |  type
#         ---------------|--------------------|---------
#         S(state)       |  env.state_shape   |  float32
#         A(action)      |  env.action_shape  |  int32
#         R(reward)      |  (1,)              |  int32
#         S'(next state) |  env.state_shape   |  float32
#         T(terminal)    |  (1,)              |  bool
#     """
# 
#     def __init__(self, state_shape, action_shape, memory_size, num_envs, batch_size):
#         self.state_shape, self.action_shape, self.num_envs, self.batch_size = state_shape, action_shape, num_envs, batch_size
#         self.count = 0
#         self.num_sample_rows = (batch_size-1) // self.num_envs + 1
#         self.sample_size = self.num_sample_rows * self.num_envs
#         self.total = (memory_size-1) // self.num_envs + 1
#         # use np.ndarray could sample by indexs easily
#         self.memory = Storage(
#             s=jnp.zeros([self.total, self.num_envs, *self.state_shape], dtype='float32'),
#             a=jnp.zeros([self.total, self.num_envs], dtype='int32'),
#             r=jnp.zeros([self.total, self.num_envs], dtype='float32'),
#             s_=jnp.zeros([self.total, self.num_envs, *self.state_shape], dtype='float32'),
#             t=jnp.zeros([self.total, self.num_envs], dtype='int32')
#         )
#     
#     @partial(jax.jit, static_argnums=0)
#     def get_memory(self, start:int, memory:Storage, s, a, r, s_, t):
#         memory = memory.replace(
#             s=memory.s.at[start].set(s),  # matrix
#             a=memory.a.at[start].set(a),  # vector
#             r=memory.r.at[start].set(r),  # vector
#             s_=memory.s_.at[start].set(s_),  # matrix
#             t=memory.t.at[start].set(t),  # vector
#         )
#         return memory
# 
#     def update(self, items:tuple):  # (S,A,R,S',T)
#         start = self.count % self.total
#         self.memory = self.get_memory(start, self.memory, *items)
#         self.count += 1
# 
#     @partial(jax.jit, static_argnums=[0,2])
#     def get_sample(  # BUG2: here use numpy is slow
#         self, 
#         key:jax.random.KeyArray,
#         size,
#         memory:Storage
#     ):
#         key, subkey = jax.random.split(key)
#         indexs = jax.random.choice(subkey, size, shape=(self.num_sample_rows,))
#         return (
#             key,
#             memory.s[indexs].reshape(self.sample_size, -1),
#             memory.a[indexs].reshape(self.sample_size, -1),
#             memory.r[indexs].reshape(self.sample_size, -1),
#             memory.s_[indexs].reshape(self.sample_size, -1),
#             memory.t[indexs].reshape(self.sample_size, -1),
#         )
#    
#     def sample(self, key):
#         return self.get_sample(key, min(self.count, self.total), self.memory)

class MemoryCache:
    """
    save the memory cache of (S,A,R,S',T):
        type name      |  shape             |  type
        ---------------|--------------------|---------
        S(state)       |  env.state_shape   |  float32
        A(action)      |  env.action_shape  |  int32
        R(reward)      |  (1,)              |  float32
        S'(next state) |  env.state_shape   |  float32
        T(terminal)    |  (1,)              |  bool
    """

    def __init__(self, state_shape, action_shape, memory_size, num_envs, batch_size):
        self.state_shape, self.action_shape, self.num_envs, self.batch_size = state_shape, action_shape, num_envs, batch_size
        self.count = 0
        self.num_sample_rows = (batch_size-1) // self.num_envs + 1
        self.sample_size = self.num_sample_rows * self.num_envs
        self.total = (memory_size-1) // self.num_envs + 1
        # use np.ndarray could sample by indexs easily
        self.memory = [
            np.zeros([self.total, self.num_envs, *self.state_shape], dtype='float32'),
            np.zeros([self.total, self.num_envs], dtype='int32'),
            np.zeros([self.total, self.num_envs], dtype='float32'),
            np.zeros([self.total, self.num_envs, *self.state_shape], dtype='float32'),
            np.zeros([self.total, self.num_envs], dtype='bool')
        ]
    
    def update(self, items:tuple):  # (S,A,R,S',T)
        start = self.count % self.total
        for a, item in zip(self.memory, items):
            a[start] = item
        self.count += 1

    def sample(self):
        idxs = np.random.choice(min(self.count, self.total), self.num_sample_rows)
        return [a[idxs].reshape(self.sample_size, -1) for a in self.memory]

class Agent(BaseAgent):

    def __init__(
            self, agent_name=None,
            env: Env = None,
            models: list = None,
            writer: SummaryWriter = None, 
            seed: int = 1,
            # hyper-parameters
            model: BaseModel = None,
            total_timesteps=const.total_timesteps,
            num_envs=const.num_envs,
            batch_size=const.batch_size,
            gamma=const.gamma,
            memory_size=const.memory_size,
            start_fit_size=const.start_fit_size,
            epsilon_max=const.epsilon_max,
            epsilon_min=const.epsilon_min,
            exporation_proportion=const.exporation_proportion,
            train_frequency=const.train_frequency,
            target_model_update_frequency=const.target_model_update_frequency,
            tau=const.tau,
            write_logs_frequency=const.write_logs_frequency,
            **kwargs
        ):
        models = [model]
        super().__init__(agent_name, env, models, writer, seed, **kwargs)

        # set key
        # self.key = jax.random.PRNGKey(self.seed)
        np.random.seed(self.seed)
        
        self.model, self.logs = model, init_logs()
        self.epsilon_max, self.epsilon_min, self.exporation_proportion, self.train_frequency, self.tau, self.target_model_update_frequency, self.write_logs_frequency = \
            epsilon_max, epsilon_min, exporation_proportion, train_frequency, tau, target_model_update_frequency, write_logs_frequency
        self.total_timesteps, self.num_envs, self.batch_size, self.gamma, self.memory_size, self.start_fit_size = \
            total_timesteps, num_envs, batch_size, gamma, memory_size, start_fit_size
        self.memory = MemoryCache(env.state_shape, env.action_shape, self.memory_size, self.num_envs, self.batch_size)
        self.slope = (self.epsilon_min - self.epsilon_max) / (self.total_timesteps * self.exporation_proportion)

        self.target_model_params = self.model.state.params.copy()
    
    @partial(jax.jit, static_argnums=0)
    def update_target_model(self, current_params, target_params):
        return jax.tree_map(
            lambda x, y: self.tau * x + (1-self.tau) * y, 
            current_params, target_params
        )
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_max + self.slope * self.global_step, self.epsilon_min)
    
    def train(self):
        state = self.env.reset()
        self.start_time, self.global_step = time.time(), 0
        num_iters = (self.total_timesteps-1)//self.num_envs+1
        # for i in tqdm(range(num_iters)):
        for i in range(num_iters):
            time1 = time.time()
            self.logs.reset()
            self.update_epsilon()
            action = self.act(state)
            state_, reward, terminal = self.env.step(action)
            self.remember(state, action, reward, state_, terminal)
            self.global_step += self.num_envs
            print("time1:", time.time() - time1)

            if self.global_step > self.start_fit_size:
                if self.global_step % self.train_frequency < self.num_envs:
                    time2 = time.time()
                    batch = self.memory.sample()

                    self.model.state, loss, q_value = self.fit(self.target_model_params, self.model.state, *batch)
                    self.logs.update(['q_value', 'loss'], [q_value, loss])
                    print("time2:", time.time() - time2)
                    
                if self.global_step % self.target_model_update_frequency < self.num_envs:
                    time3 = time.time()
                    self.target_model_params = self.update_target_model(self.model.state.params, self.target_model_params)
                    print("time3:", time.time() - time3)

            state = state_
            self.logs.update(
                ['episode_step', 'episode_return'],
                [self.env.get_terminal_steps(), self.env.get_terminal_rewrad()]
            )
            # if episode end, then we plot it
            self.logs.writer_tensorboard(self.writer, self.global_step, drops=['q_value', 'loss'])
            # print("time3:", time.time() - time3)

            if (i+1) % self.write_logs_frequency == 0:
                self.write_tensorboard()
            if self.global_step % int(1e4) < self.num_envs:
                self.model.save_weights()

    def evaluate(self):
        state = self.env.reset()
        self.start_time, self.global_step = time.time(), 0
        self.epsilon = 0
        num_iters = (self.total_timesteps-1)//self.num_envs+1
        for i in tqdm(range(num_iters)):
            self.logs.reset()
            action = self.act(state)
            state_, _, _ = self.env.step(action)
            self.global_step += self.num_envs
            state = state_
            self.logs.update(
                ['episode_step', 'episode_return'],
                [self.env.get_terminal_steps(), self.env.get_terminal_rewrad()]
            )
            if i % self.write_logs_frequency == 0:
                self.write_tensorboard()
    
    @partial(jax.jit, static_argnums=0)
    def get_model_action(self, params, s):
        q_pred = self.model.state.apply_fn(params, s)
        return jnp.argmax(q_pred, axis=-1)

    def act(self, s):  # epsilon choice policy
        rand = np.random.uniform(size=(1,))[0]  # BUG1: there use np is better than jax!
        if rand <= self.epsilon:
            action = np.random.randint(low=0, high=self.env.action_ndim, size=(self.num_envs,))
        else:
            action = jax.device_get(self.get_model_action(self.model.state.params, s))
        return action

    def remember(self, state, action, reward, state_, terminal):
        self.memory.update((state, action, reward, state_, terminal))

    
    @partial(jax.jit, static_argnums=0)
    def fit(self, target_model_params, state:TrainState, s, a, r, s_, t):
        a, r, t = a.flatten(), r.flatten(), t.flatten()
        td_target = r + self.gamma * jnp.max(
            self.model.state.apply_fn(target_model_params, s_), axis=-1
        ) * (1-t)
        # q_target = state.apply_fn(state.params, s)
        # q_target = q_target.at[jnp.arange(self.batch_size), a.squeeze()].set(td_target.squeeze())

        # def train_step(state:TrainState, X, y):
        def loss_fn(params):
            logits = state.apply_fn(params, s)
            logits = logits[jnp.arange(self.batch_size), a]
            return ((logits - td_target) ** 2).mean(), logits
            # return loss, logits

        loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss_val, q_value), grads = loss_grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        # return state, loss_val, q_value

        # state, loss, q_value = train_step(state, s, q_target)
        return state, loss_val, q_value
    
    def write_tensorboard(self):
        self.logs.writer_tensorboard(self.writer, self.global_step, drops=['episode_step', 'episode_return'])
        self.writer.add_scalar('charts/epsilon', self.epsilon, self.global_step)
        self.writer.add_scalar(
            'charts/SPS_avg',
            int(self.global_step / (time.time() - self.start_time)),
            self.global_step
        )
    