# -*- coding: utf-8 -*-
'''
@File    : ppo_torch.py
@Time    : 2023/10/03 15:13:30
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
2023/10/03: Add pytorch to complish PPO.
'''
from katarl.agents import BaseAgent
from katarl.agents.models.base.base_torch import TorchModel
from katarl.envs import Env
from katarl.utils.logs import Logs, MeanMetric

from typing import NamedTuple
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import time

def get_logs() -> Logs:
    return Logs(
        init_logs={
            'terminal_length': MeanMetric(),
            'terminal_rewards': MeanMetric(),
            'episode_length': MeanMetric(),
            'episode_rewards': MeanMetric(),
            'v_value': MeanMetric(),
            'ad_value': MeanMetric(),
            'loss_p': MeanMetric(),
            'loss_v': MeanMetric(),
            'loss_ent': MeanMetric(),
            'max_terminal_reward': 0,
            'learning_rate': 0,
        },
        folder2name={
            'charts': ['terminal_length', 'terminal_rewards', 'episode_length', 'episode_rewards', 'max_terminal_reward', 'learning_rate'],
            'metrics': ['v_value', 'ad_value', 'loss_p', 'loss_v', 'loss_ent']
        }
    )

class Actor:

    def __init__(self, env:Env, model:TorchModel, args:NamedTuple):
        self.env, self.model, self.args = env, model, args
        self.row, self.col = self.args.num_steps, self.args.num_envs
        self.state = torch.tensor(self.env.reset())
        self.S, self.A, self.R, self.S_, self.T, self.AD, self.V, self.LP = (
            torch.zeros(size=(self.row, self.col, *self.env.state_shape), dtype=torch.float32),
            torch.zeros(size=(self.row, self.col), dtype=torch.int32),
            torch.zeros(size=(self.row, self.col), dtype=torch.float32),
            torch.zeros(size=(self.row, self.col, *self.env.state_shape), dtype=torch.float32),
            torch.zeros(size=(self.row, self.col), dtype=torch.bool),
            torch.zeros(size=(self.row, self.col), dtype=torch.float32),
            torch.zeros(size=(self.row, self.col), dtype=torch.float32),
            torch.zeros(size=(self.row, self.col), dtype=torch.float32)
        )
        
    def get_action(self, x):
        v, logits = self.model(x)
        # Gumbel-softmax trick: https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html
        u = torch.rand(size=logits.shape)
        action = torch.argmax(logits - torch.log(-torch.log(u)), dim=-1)
        log_proba = torch.log_softmax(logits, dim=-1)[torch.arange(action.shape[0]), action]
        return v, log_proba, action

    def act(self):
        S, A, R, S_, T, AD, V, LP = self.S, self.A, self.R, self.S_, self.T, self.AD, self.V, self.LP
        terminal_length, terminal_rewards, episode_length, episode_rewards = [], [], [], []
        for i in range(self.row):
            v, log_proba, action = self.get_action(self.state)
            V[i], LP[i] = v.flatten(), log_proba
            state_, reward, terminal = self.env.step(action.tolist())
            state_, reward, terminal = torch.tensor(state_), torch.tensor(reward), torch.tensor(terminal)
            S[i], A[i], R[i], S_[i], T[i] = \
                self.state, action, reward, state_, terminal
            self.state = state_
            terminal_length += self.env.get_terminal_length()
            terminal_rewards += self.env.get_terminal_reward()
            episode_length += self.env.get_episode_length()
            episode_rewards += self.env.get_episode_reward()
        v_last, _ = self.model(self.state)
        v_last = v_last.reshape(1, self.col)
        # Calc Delta
        AD = R + self.args.gamma * torch.cat([V[1:,:], v_last.reshape(1,-1)], dim=0) * (~T) - V
        for i in reversed(range(self.row-1)):
            AD[i] += self.args.gamma * self.args.coef_gae * AD[i+1] * (~T[i])
        S, A, AD, V, LP = \
            S.reshape(self.args.data_size, *self.env.state_shape), \
            A.reshape(self.args.data_size, 1), \
            AD.reshape(self.args.data_size, 1), \
            V.reshape(self.args.data_size, 1), \
            LP.reshape(self.args.data_size, 1)
        dataset = (S, A, AD, V, LP)
        return dataset, terminal_length, terminal_rewards, episode_length, episode_rewards

class Agent(BaseAgent):

    def __init__(
            self,
            agent_name: str = None,
            env: Env = None,
            models: list[TorchModel] = None,
            writer: SummaryWriter = None,
            args: NamedTuple = None,
            model: TorchModel = None
        ):
        self.model, models = model, [model]
        super().__init__(agent_name, env, models, writer, args)
        self.logs = get_logs()
        self.actor = Actor(self.env, self.model, args)

        np.random.seed(self.args.seed)  # use np for random the dataset training idxs
        torch.manual_seed(self.args.seed)
    
    def train(self):
        self.start_time = time.time()
        for i in tqdm(range(self.args.num_iters)):
            self.logs.reset()
            terminal_length, terminal_rewards, episode_length, episode_rewards = self.fit()
            self.global_step = (i+1) * self.args.data_size
            max_reward = None if len(terminal_rewards) == 0 else np.max(terminal_rewards)
            self.logs.update(
                ['terminal_length', 'terminal_rewards', 'episode_length', 'episode_rewards', 'max_terminal_reward', 'learning_rate'],
                [terminal_length, terminal_rewards, episode_length, episode_rewards, max_reward, self.model.optimizer.param_groups[0]['lr']]
            )
            self.write_tensorboard()
            if (i+1) % (self.args.num_iters // self.args.num_model_save) == 0 or i == self.args.num_iters - 1:
                print(f"Save weights at global step:", self.global_step)
                for model in self.models: model.save_weights()
    
    def evaluate(self):
        self.start_time = time.time()
        for i in tqdm(range(self.args.iter_nums)):
            self.logs.reset()
            with torch.no_grad():
                _, steps, rewards = self.actor.act()
            self.global_step = (i+1) * self.args.data_size
            max_reward = 0 if len(rewards) == 0 else int(np.max(rewards))
            self.logs.update(
                ['episode_step', 'episode_return', 'max_reward'],
                [np.mean(steps), np.mean(rewards), max_reward]
            )
            self.write_tensorboard()

    def fit(self):
        with torch.no_grad():
            dataset, *info = self.actor.act()
        for _ in range(self.args.epochs):
            idxs = np.random.permutation(self.args.data_size)
            for i in range(0, self.args.data_size, self.args.batch_size):
                batch = (dataset[k][idxs[i:i+self.args.batch_size]] for k in range(len(dataset)))

                s, a, ad, v, logpi = batch
                v_now, logits = self.model(s)
                loss_v = ((v_now - v - ad) ** 2).mean() / 2

                if self.args.flag_ad_normal:
                    ad = (ad - ad.mean()) / (ad.std() + self.args.EPS)
                
                logpi_now = torch.log_softmax(logits, dim=-1)[torch.arange(a.shape[0]), a.flatten()].reshape(-1, 1)
                rate = torch.exp(logpi_now - logpi)
                loss_p = torch.minimum(
                    ad * rate,
                    ad * torch.clip(
                        rate,
                        1 - self.args.epsilon,
                        1 + self.args.epsilon
                    )
                ).mean()

                # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/            
                # log_p = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
                # log_p = log_p.clip(min=jnp.finfo(log_p.dtype).min)
                loss_entropy = - (torch.log_softmax(logits, dim=-1) * torch.softmax(logits, dim=-1)).sum(-1).mean()
            
                loss = - loss_p \
                    + self.args.coef_value * loss_v \
                    - self.args.coef_entropy * loss_entropy
                # return loss, (v, ad, loss_p, loss_v, loss_entropy)
                
                # (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, dataset, idxs)
                # state = state.apply_gradients(grads=grads)
                if self.args.flag_anneal_lr:
                    per = self.args.data_size // self.args.batch_size * self.args.epochs
                    frac = 1.0 - self.model.step // per / self.args.num_iters
                    lrnow = self.args.learning_rate * frac
                    self.model.optimizer.param_groups[0]['lr'] = lrnow
                self.model.step += 1

                loss.backward()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()

                self.logs.update(
                    ['v_value', 'ad_value', 'loss_p', 'loss_v', 'loss_ent'],
                    [v.detach().numpy(), ad.detach().numpy(), loss_p.detach().numpy(), loss_v.detach().numpy(), loss_entropy.detach().numpy()]
                )
        return info
    
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
