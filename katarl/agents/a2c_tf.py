from katarl.agents import BaseAgent
from katarl.agents.models.base.base_tf import TFModel
from katarl.envs import Env
from katarl.utils.logs import Logs, MeanMetric

from typing import NamedTuple
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
import tensorflow as tf
import numpy as np
keras = tf.keras

def get_logs() -> Logs:
    return Logs(
        init_logs={
            'episode_step': MeanMetric(),
            'episode_return': MeanMetric(),
            'v_value': MeanMetric(),
            'loss': MeanMetric(),
        },
        folder2name={
            'charts': ['episode_step', 'episode_return'],
            'metrics': ['v_value', 'loss']
        }
    )

class Agent(BaseAgent):

    def __init__(
            self,
            agent_name: str = None,
            env: Env = None,
            models: list[TFModel] = None,
            writer: SummaryWriter = None,
            args: NamedTuple = None,
            # hyper
            value_model: TFModel = None,
            policy_model: TFModel = None
        ):
        models = [value_model, policy_model]
        self.value_model, self.policy_model = value_model, policy_model
        super().__init__(agent_name, env, models, writer, args)
        self.logs = get_logs()
    
    def train(self):
        state = self.env.reset()
        self.start_time = time.time()
        for self.global_step in tqdm(range(self.args.total_timesteps)):
            self.logs.reset()
            action = self.act(state)
            state_, reward, terminal = self.env.step(action)
            loss, v_value = self.fit(state, action, reward, state_, terminal)
            self.logs.update(['v_value', 'loss'], [v_value, loss])
            state = state_
            self.logs.update(
                ['episode_step', 'episode_return'],
                [self.env.get_terminal_steps(), self.env.get_terminal_reward()]
            )
            self.logs.writer_tensorboard(self.writer, self.global_step, drops=['v_value', 'loss'])
            if (self.global_step + 1) % self.args.write_logs_frequency == 0:
                self.write_tensorboard()
            if (self.global_step + 1) % (self.args.total_timesteps // self.args.num_model_save) == 0 or self.global_step == self.args.total_timesteps - 1:
                    self.value_model.save_weights()
                    self.policy_model.save_weights()
    
    def evaluate(self):
        for episode in tqdm(range(self.episodes)):
            self.logs.reset()
            state = self.env.reset()
            for step in range(self.env.max_step):
                action = self.act(state)
                state_, _, terminal = self.env.step(action)
                state = state_
                if terminal: break
            self.logs.update(['episode', 'step'], [episode, step])
            self.write_tensorboard()
    
    def act(self, state):
        action_proba = self.policy_model(state)[0]
        action = np.random.choice(self.env.action_ndim, size=(1,), p=action_proba.numpy())
        return action
        # return tf.argmax(action_proba).numpy()
    
    @tf.function
    def train_step(self, state, action, y):
        with tf.GradientTape() as tape:
            v = self.value_model(state)[0]
        delta = v - y  # TD error
        g_v = tape.gradient(v, self.value_model.get_trainable_weights())
        for i in range(len(g_v)): g_v[i] *= delta
        self.value_model.apply_gradients(g_v)

        with tf.GradientTape() as tape:
            logit = tf.math.log(self.policy_model(state)[0][action[0]])
        g_p = tape.gradient(logit, self.policy_model.get_trainable_weights())
        for i in range(len(g_p)): g_p[i] *= delta
        self.policy_model.apply_gradients(g_p)
        return tf.square(delta), v

    def fit(self, state, action, reward, state_, terminal):
        y = reward
        if not terminal: y += self.args.gamma * self.value_model(state_)[0]
        loss, v_value = self.train_step(state, action, y)
        return loss, v_value
    
    def write_tensorboard(self):
        self.logs.writer_tensorboard(self.writer, self.global_step, drops=['episode_step', 'episode_return'])
        self.writer.add_scalar('charts/SPS_avg', int(self.global_step / (time.time()-self.start_time)), self.global_step)
