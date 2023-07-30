from agents import Agent
from agents.constants import *
from agents.constants.DQN import *
from agents.models.utils import build_model
from envs.gym_env import GymEnv
from utils import make_onehot
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from envs import Env
keras = tf.keras

class Logs:
    def __init__(self):
        self.logs = {
            'episode': 0,
            'step': 0,
            'q_value': keras.metrics.Mean(name='q_value'),
            'loss': keras.metrics.Mean(name='loss'),
            'frame': []
        }
    
    def reset(self):
        self.logs['episode'] = 0
        self.logs['step'] = 0
        self.logs['frame'] = []
        self.logs['q_value'].reset_state()
        self.logs['loss'].reset_state()
    
    def update(self, keys, values):
        for key, value in zip(keys, values):
            if value is not None:
                target = self.logs[key]
                if isinstance(target, keras.metrics.Metric):
                    target.update_state(value)
                elif isinstance(target, list):
                    target.append(value)
                elif isinstance(target, int):
                    self.logs[key] = value
    
    def to_dict(self, show_frame=False):
        ret = self.logs.copy()
        for key, value in ret.items():
            if value is not None:
                target = self.logs[key]
                if isinstance(target, keras.metrics.Metric):
                    ret[key] = round(float(target.result()), 5) if target.count else None
        if not show_frame: ret.pop('frame')
        return ret
    
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

    def __init__(self, state_shape, action_shape):
        self.state_shape, self.action_shape = state_shape, action_shape
        self.count = 0
        self.total = memory_size
        # use np.ndarray could sample by indexs easily
        self.s = np.zeros([self.total, *self.state_shape], dtype='float32')
        self.a = np.zeros([self.total, *self.action_shape], dtype='int32')
        self.r = np.zeros([self.total, 1], dtype='float32')
        self.s_ = np.zeros([self.total, *self.state_shape], dtype='float32')
        self.t = np.zeros([self.total, 1], dtype='bool')
        self.memory = [self.s, self.a, self.r, self.s_, self.t]
    
    def update(self, item:tuple):  # (S,A,R,S',T)
        for value, array in zip(item, self.memory):
            array[self.count % self.total] = value
        self.count += 1
    
    def sample(self, num=1):
        size = min(self.count, self.total)
        indexs = np.random.choice(size, num)
        return [array[indexs] for array in self.memory]

class DQN(Agent):

    def __init__(
            self, env:Env=None, verbose=False,
            agent_name='DQN', agent_id=0,
            model_name='MLP', load_id=None,
            episodes=100, **kwargs
        ):
        super().__init__(env, verbose, agent_name, agent_id, model_name, load_id, episodes, **kwargs)
        self.model = build_model(model_name, self.env.state_shape, load_id=load_id)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = keras.losses.MeanSquaredError()
        self.logs = Logs()
        self.epsilon = epsilon_max
        self.memory = MemoryCache(env.state_shape, env.action_shape)
    
    def train(self):
        for episode in tqdm(range(self.episodes)):
            self.logs.reset()
            state = self.env.reset()
            for step in range(self.env.max_step):
                action = self.act(state)
                state_, reward, terminal = self.env.step(action)
                self.remember(state, action, reward, state_, terminal)
                loss, q_value = self.fit()
                frame = self.env.render() if self.verbose else None
                self.logs.update(['q_value', 'loss', 'frame'], [q_value, loss, frame])
                state = state_
                if terminal: break
            self.logs.update(['episode', 'step'], [episode, step])
            self.update_history()
            if (episode + 1) % 100 == 0:
                self.model.save_weights()

    def evaluate(self):
        pass

    def act(self, state):  # epsilon choice policy
        rand = np.random.rand()
        if rand <= self.epsilon:
            return np.random.randint(0, 2)
        else:
            q_pred = self.model(tf.constant([state]))[0]
            action = np.argmax(q_pred)
            # print(self.epsilon)
            # print(q_pred)
            # print(action)
            return action

    def remember(self, state, action, reward, state_, terminal):
        self.memory.update((state, action, reward, state_, terminal))

    @tf.function
    def train_step(self, X, y):
        with tf.GradientTape() as tape:
            q_state = self.model(X)
            loss = self.loss_fn(y, q_state)
            # loss = keras.losses.MSE(y, q_state)
            # loss = tf.reduce_sum(tf.square(q_state - y)) / batch_size
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss, tf.reduce_mean(q_state)

    def fit(self):
        if self.memory.count < start_fit_size: return None, None
        s, a, r, s_, t = self.memory.sample(batch_size)
        r, t = r.squeeze(), t.squeeze()
        a_onehot = make_onehot(a, depth=2).astype('bool')

        td_target = r + gamma * np.max(self.model(s_), axis=-1) * (1-t)
        q_state = self.model(s)
        q_target = q_state.numpy()
        q_target[a_onehot] = td_target

        # print(f"{r=}\n{td_target=}\n{a=}\n{q_state=}\n{q_target=}")

        loss, q_value = self.train_step(s, q_target)
        # decrease epsilon after one fit!
        self.epsilon = max(self.epsilon * epsilon_decay, epsilon_min)
        return loss, q_value

    def update_history(self):
        self.best_episode.update_best(now=self.logs.logs['step'], logs=self.logs.to_dict(show_frame=True))
        self.history.update_dict(self.logs.to_dict())
    
if __name__ == '__main__':
    dqn = DQN(
        env=GymEnv(name="CartPole-v1", render_mode="rgb_array"),
        verbose=True, agent_id=0, episodes=5
    )
    dqn.train()