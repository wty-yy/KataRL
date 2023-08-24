from tensorboardX import SummaryWriter
from agents import Agent
import agents.constants.A2C as const
from agents.models import BaseModel
from envs import Env
from utils.logs import Logs, MeanMetric
from tqdm import tqdm
import tensorflow as tf
import numpy as np
keras = tf.keras

def get_logs() -> Logs:
    return Logs(
        init_logs={
            'episode': 0,
            'step': 0,
            'v_value': MeanMetric(),
            'loss': MeanMetric(),
        }
    )

def expand_dim(state):
    return tf.expand_dims(tf.constant(state), axis=0)

class A2C(Agent):

    def __init__(
            self, agent_name=None,
            env: Env = None,
            value_model: BaseModel = None, policy_model: BaseModel = None,
            writer: SummaryWriter = None,
            # hyperparameters
            episodes=const.episodes,
            gamma=const.gamma,
            **kwargs
        ):
        models = [value_model, policy_model]
        super().__init__(agent_name, env, models, writer, **kwargs)
        self.value_model, self.policy_model = value_model, policy_model
        self.episodes, self.gamma = episodes, gamma
        self.logs = get_logs()
    
    def train(self):
        for episode in tqdm(range(self.episodes)):
            self.logs.reset()
            state = self.env.reset()
            for step in range(self.env.max_step):
                action = self.act(state)
                state_, reward, terminal = self.env.step(action)
                loss, v_value = self.fit(state, action, reward, state_, terminal)
                self.logs.update(['v_value', 'loss'], [v_value, loss])
                state = state_
                if terminal: break
            self.logs.update(['episode', 'step'], [episode, step])
            self.write_tensorboard()
            if (episode + 1) % 100 == 0:
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
        state = expand_dim(state)
        action_proba = self.policy_model(state)[0]
        # print(action_proba)
        action = np.random.choice(self.env.action_ndim, p=action_proba.numpy())
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
            logit = tf.math.log(self.policy_model(state)[0][action])
        g_p = tape.gradient(logit, self.policy_model.get_trainable_weights())
        for i in range(len(g_p)): g_p[i] *= delta
        self.policy_model.apply_gradients(g_p)
        return tf.pow(delta, 2), v

    def fit(self, state, action, reward, state_, terminal):
        y = reward
        state, state_ = expand_dim(state), expand_dim(state_)
        if not terminal: y += self.gamma * self.value_model(state_)[0]
        # print(state, action)
        loss, v_value = self.train_step(state, action, y)
        return loss.numpy(), v_value
    
    def write_tensorboard(self):
        episode = self.logs.logs['episode']
        d = self.logs.to_dict(drops=['episode'])
        for key, value in d.items():
            if key in ['step']: name = 'charts/' + key
            else: name = 'metrics/' + key
            if value is not None:
                self.writer.add_scalar(name, value, episode)
        if d['loss'] is not None:
            self.writer.add_scalar(
                'charts/SPS',
                int(d['step'] / self.logs.get_time_length()),
                episode
            )
