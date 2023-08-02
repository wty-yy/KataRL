from agents import Agent
from agents.constants import PATH
from agents.constants.A2C import gamma
from agents.models import Model
from envs import Env
from utils.logs_manager import Logs
from tqdm import tqdm
import tensorflow as tf
keras = tf.keras

def get_logs() -> Logs:
    return Logs(
        init_logs={
            'episode': 0,
            'step': 0,
            'v_value': keras.metrics.Mean(name='v_value'),
            'loss': keras.metrics.Mean(name='loss'),
            'frame': []
        }
    )

def expand_dim(state):
    return tf.expand_dims(tf.constant(state), axis=0)

class A2C(Agent):
    
    def __init__(
            self, env: Env = None, verbose=False,
            agent_name='A2C', agent_id=0,
            value_model: Model = None, policy_model: Model = None,
            episodes=1000, gamma=gamma,  # constants.A2C
            **kwargs
        ):
        models = [value_model, policy_model]
        super().__init__(env, verbose, agent_name, agent_id, episodes, models, **kwargs)
        self.value_model, self.policy_model = value_model, policy_model
        self.gamma = gamma
        self.logs = get_logs()
    
    def train(self):
        for episode in tqdm(range(self.episodes)):
            self.logs.reset()
            state = self.env.reset()
            for step in range(self.env.max_step):
                action = self.act(state)
                state_, reward, terminal = self.env.step(action)
                loss, v_value = self.fit(state, action, reward, state_, terminal)
                frame = self.env.render() if self.verbose else None
                self.logs.update(['v_value', 'loss', 'frame'], [v_value, loss, frame])
                state = state_
                if terminal: break
            self.logs.update(['episode', 'step'], [episode, step])
            self.update_history()
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
                frame = self.env.render() if self.verbose else None
                self.logs.update(['frame'], [frame])
                state = state_
                if terminal: break
            self.logs.update(['episode', 'step'], [episode, step])
            self.update_history()
    
    def act(self, state):
        state = expand_dim(state)
        action_proba = self.policy_model(state)[0]
        return tf.argmax(action_proba).numpy()
    
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
    
    def update_history(self):
        self.best_episode.update_best(
            now=self.logs.logs['step'], logs=self.logs.to_dict()
        )
        self.history.update_dict(self.logs.to_dict(drops=['frame']))
