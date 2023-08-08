from agents import Agent
from agents.constants.PPO import *
from agents.models import Model
from envs import Env
from utils.logs_manager import Logs
from utils import make_onehot
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
            'loss': keras.metrics.Mean(name='loss'),
            'plot_frame': []
        }
    )

def expand_dim(state):
    return np.array(state).reshape(1, -1)

class Copyer(keras.optimizers.Optimizer):
    
    def __init__(self, **kwargs):
        super().__init__(name='copy', **kwargs)
        self._learning_rate = 0
    
    def update_step(self, target, variable):
        variable.assign(target)

copyer = Copyer()

class Actor:

    def __init__(self, env:Env, model:keras.Model, gamma, lambda_):
        self.env, self.model, self.gamma, self.lambda_ = \
            env, model, gamma, lambda_
        self.reset()
    
    def reset(self):
        self.state = self.env.reset()
        self.total_step = 0
    
    def get_action(self, proba):
        rand = np.random.rand()
        for i in range(len(proba)):
            if rand <= proba[i] + EPS: return i
            else: rand -= proba[i]

    def act(self, step_T):
        action_shape = self.env.action_shape
        state_shape = self.env.state_shape
        S, A, R, S_, D = \
            np.zeros(shape=(step_T, *state_shape), dtype='float32'), \
            np.zeros(shape=(step_T, *action_shape), dtype='int32'), \
            np.zeros(shape=(step_T, 1), dtype='float32'), \
            np.zeros(shape=(step_T, *state_shape), dtype='float32'), \
            np.zeros(shape=(step_T, 1), dtype='float32')
        max_step = 0
        for step in step_T:
            v, proba = self.model(expand_dim(self.state))
            D[step] = v[0][0].numpy()
            action = self.get_action(proba.numpy())
            state_, reward, terminal = self.env.step(action)
            S[step], A[step], R[step], S_[step] = \
                self.state, action, reward, state_
            self.state = state_
            self.total_step += 1
            max_step = max(max_step, self.total_step)
            if terminal: self.reset()
        # Calc Advantage Value: D
        D = R + self.gamma * np.r_[D[1:,0],0].reshape(-1,1) - D
        for i in range(len(D)-2, -1, -1):
            D[i,0] += self.gamma * self.lambda_ * D[i+1,0]
        ds = tf.data.Dataset.from_tensor_slices((S,A,R,S_,D))
        return ds, max_step

class PPO(Agent):

    def __init__(
            self, env: Env = None, verbose=False,
            agent_name=None, agent_id=0,
            episodes=None, models: list = None,
            model: Model = None,
            gamma=gamma, lambda_=lambda_,
            actor_N=actor_N, iter_M=iter_M, step_T=step_T,
            epochs=epochs, batch_size=batch_size,
            **kwargs):
        models = [model]
        super().__init__(env, verbose, agent_name, agent_id, episodes, models, **kwargs)
        self.model, self.gamma, self.lambda_, \
            self.actor_N, self.iter_M, self.step_T, \
            self.epochs, self.batch_size = model, gamma, lambda_, \
            actor_N, iter_M, step_T, epochs, batch_size
        self.logs = get_logs()
        # build old model init as model
        self.model_old = self.model.build_model()
        self.copy_weights()
        # init actors
        self.actors = [
            Actor(self.env, self.model_old, self.gamma, self.lambda_) \
            for _ in range(self.actor_N)
        ]
    
    def copy_weights(self):
        copyer.apply_gradients(zip(self.model.get_trainable_weights(),
                                   self.model_old.trainable_weights))

    def train(self):
        for i in range(self.iter_M):
            self.logs.reset()
            max_step = self.fit()
            frame = (i+1) * self.iter_M
            self.logs.update(['frame', 'max_step'], [frame, max_step])
            self.update_history()
            if (i + 1) % 10 == 0:
                self.model.save_weights()
    
    def fit(self):
        ds = None
        for actor in self.actors:
            ds_new = actor.act(self.step_T)
            ds = ds_new if ds is None else ds.concatenate(ds_new)
        ds = ds.shuffle(10000).batch(self.batch_size)
        for epoch in self.epochs:
            for s, a, r, s_, A in ds:
                a = make_onehot(a, depth=self.env.action_shape[0])
    
    def update_history(self):
        self.best_episode.update_best(
            now=self.logs.logs['max_step'], logs=self.logs.to_dict()
        )
        self.history.update_dict(self.logs.to_dict(drops=['plot_frame']))
