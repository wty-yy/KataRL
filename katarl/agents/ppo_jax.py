# -*- coding: utf-8 -*-
'''
@File    : ppo_jax.py
@Time    : 2023/08/30 13:44:53
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 完成ppo的Jax实现
'''

from katarl.agents import BaseAgent
from katarl.agents.models.base.base_jax import JaxModel
from katarl.envs import Env
from katarl.utils.logs import Logs, MeanMetric

from typing import NamedTuple
from tensorboardX import SummaryWriter
from tqdm import tqdm
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from functools import partial
import numpy as np
import time

def get_logs() -> Logs:
    return Logs(
        init_logs={
            'episode_step': MeanMetric(),
            'episode_return': MeanMetric(),
            'v_value': MeanMetric(),
            'ad_value': MeanMetric(),
            'loss_p': MeanMetric(),
            'loss_v': MeanMetric(),
            'loss_ent': MeanMetric(),
            # 'advantage': MeanMetric(),
            'max_reward': 0,
            'learning_rate': 0,
        },
        folder2name={
            'charts': ['episode_step', 'episode_return', 'max_reward', 'learning_rate'],
            'metrics': ['v_value', 'ad_value', 'loss_p', 'loss_v', 'loss_ent']
        }
    )

class Actor:

    def __init__(self, env:Env, model:JaxModel, args:NamedTuple):
        self.env, self.model, self.args = env, model, args
        self.row, self.col = self.args.num_steps, self.args.num_envs
        self.key = jax.random.PRNGKey(self.args.seed)
        self.state = self.env.reset()
        self.S, self.A, self.R, self.S_, self.T, self.AD, self.V, self.LP = \
            np.zeros(shape=(self.row, self.col, *self.env.state_shape), dtype='float32'), \
            np.zeros(shape=(self.row, self.col), dtype='int32'), \
            np.zeros(shape=(self.row, self.col), dtype='float32'), \
            np.zeros(shape=(self.row, self.col, *self.env.state_shape), dtype='float32'), \
            np.zeros(shape=(self.row, self.col), dtype='bool'), \
            np.zeros(shape=(self.row, self.col), dtype='float32'), \
            np.zeros(shape=(self.row, self.col), dtype='float32'), \
            np.zeros(shape=(self.row, self.col), dtype='float32')
        
    @partial(jax.jit, static_argnums=0)
    def pred(self, params, x):
        return self.model.state.apply_fn(params, x)
    
    @partial(jax.jit, static_argnums=0)
    def get_action(self, key, params, x):
        v, logits = self.pred(params, x)
        key, subkey = jax.random.split(key)
        # Gumbel-softmax trick: https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        log_proba = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        return key, v, log_proba, action

    def act(self):
        S, A, R, S_, T, AD, V, LP = self.S, self.A, self.R, self.S_, self.T, self.AD, self.V, self.LP
        terminal_steps, terminal_rewards = [], []
        for i in range(self.row):
            self.key, v, log_proba, action = self.get_action(self.key, self.model.state.params, self.state)
            V[i], LP[i] = v.flatten(), log_proba
            state_, reward, terminal = self.env.step(action.tolist())
            S[i], A[i], R[i], S_[i], T[i] = \
                self.state, action, reward, state_, terminal
            self.state = state_
            terminal_steps += self.env.get_terminal_steps()
            terminal_rewards += self.env.get_terminal_reward()
        v_last, _ = self.pred(self.model.state.params, self.state)
        v_last = v_last.reshape(1, self.col)
        # Calc Delta
        AD = R + self.args.gamma * np.r_[V[1:,:], v_last] * (~T) - V
        for i in reversed(range(self.row-1)):
            AD[i] += self.args.gamma * self.args.coef_gae * AD[i+1] * (~T[i])
        S, A, AD, V, LP = \
            S.reshape(self.args.data_size, *self.env.state_shape), \
            A.reshape(self.args.data_size, 1), \
            AD.reshape(self.args.data_size, 1), \
            V.reshape(self.args.data_size, 1), \
            LP.reshape(self.args.data_size, 1)
        dataset = (S, A, AD, V, LP)
        return dataset, terminal_steps, terminal_rewards

class Agent(BaseAgent):

    def __init__(
            self,
            agent_name: str = None,
            env: Env = None,
            models: list[JaxModel] = None,
            writer: SummaryWriter = None,
            args: NamedTuple = None,
            model: JaxModel = None
        ):
        self.model, models = model, [model]
        super().__init__(agent_name, env, models, writer, args)
        self.logs = get_logs()
        self.actor = Actor(self.env, self.model, args)

        np.random.seed(self.args.seed)  # use np for random the dataset training idxs
    
    def train(self):
        self.start_time = time.time()
        for i in tqdm(range(self.args.num_iters)):
            self.logs.reset()
            steps, rewards = self.fit()
            self.global_step = (i+1) * self.args.data_size
            max_reward = None if len(rewards) == 0 else int(np.max(rewards))
            self.logs.update(
                ['episode_step', 'episode_return', 'max_reward', 'learning_rate'],
                [steps, rewards, max_reward, self.model.state.opt_state[1][1]['learning_rate']]
            )
            self.write_tensorboard()
            if (i + 1) % 10 == 0:
                self.model.save_weights()
    
    def evaluate(self):
        self.start_time = time.time()
        for i in tqdm(range(self.args.iter_nums)):
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
        dataset, steps, rewards = self.actor.act()
        for _ in range(self.args.epochs):
            idxs = np.random.permutation(self.args.data_size)
            for i in range(0, self.args.data_size, self.args.batch_size):
                self.model.state, (v, ad, loss_p, loss_v, loss_ent) = \
                    self.train_step(self.model.state, dataset, idxs[i:i+self.args.batch_size])
                self.logs.update(
                    ['v_value', 'ad_value', 'loss_p', 'loss_v', 'loss_ent'],
                    [v, ad, loss_p, loss_v, loss_ent]
                )
        return steps, rewards
    
    @partial(jax.jit, static_argnums=0)
    def train_step(self, state:TrainState, dataset, idxs):
        def loss_fn(params, dataset, idxs):
            s, a, ad, v, logpi = jax.tree_map(lambda x: x[idxs], dataset)
            v_now, logits = self.model.state.apply_fn(params, s)
            loss_v = ((v_now - v - ad) ** 2).mean() / 2

            if self.args.flag_ad_normal:
                ad = (ad - ad.mean()) / (ad.std() + self.args.EPS)
            
            logpi_now = jax.nn.log_softmax(logits)[jnp.arange(a.shape[0]), a.flatten()].reshape(-1, 1)
            rate = jnp.exp(logpi_now - logpi)
            loss_p = jnp.minimum(
                ad * rate,
                ad * jnp.clip(
                    rate,
                    1 - self.args.epsilon,
                    1 + self.args.epsilon
                )
            ).mean()

            # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/            
            log_p = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
            log_p = log_p.clip(min=jnp.finfo(log_p.dtype).min)
            loss_entropy = - (log_p * jax.nn.softmax(logits)).sum(-1).mean()
        
            loss = - loss_p \
                   + self.args.coef_value * loss_v \
                   - self.args.coef_entropy * loss_entropy
            return loss, (v, ad, loss_p, loss_v, loss_entropy)
        
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, dataset, idxs)
        state = state.apply_gradients(grads=grads)
        return state, metrics
        
    
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
