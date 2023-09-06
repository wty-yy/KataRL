from katarl.agents import BaseAgent
from katarl.agents.models.base.base_jax import JaxModel
from katarl.envs import Env
from katarl.utils.logs import Logs, MeanMetric

from typing import NamedTuple
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm

from functools import partial
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

def get_logs() -> Logs:
    return Logs(
        init_logs={
            'terminal_length': MeanMetric(),
            'terminal_rewards': MeanMetric(),
            'v_value': MeanMetric(),
            'loss': MeanMetric(),
        },
        folder2name={
            'charts': ['terminal_length', 'terminal_rewards'],
            'metrics': ['v_value', 'loss']
        }
    )

class Agent(BaseAgent):

    def __init__(
            self,
            agent_name: str = None,
            env: Env = None,
            models: list[JaxModel] = None,
            writer: SummaryWriter = None,
            args: NamedTuple = None,
            # hyper
            value_model: JaxModel = None,
            policy_model: JaxModel = None
        ):
        models = [value_model, policy_model]
        self.value_model, self.policy_model = value_model, policy_model
        super().__init__(agent_name, env, models, writer, args)
        self.logs = get_logs()

        self.key = jax.random.PRNGKey(self.args.seed)
        self.target_model_params = self.value_model.state.params.copy()

    @partial(jax.jit, static_argnums=0)
    def update_target_model(self, current_params, target_params):
        return jax.tree_map(
            lambda x, y: self.args.tau * x + (1-self.args.tau) * y, 
            current_params, target_params
        )

    def train(self):
        state = self.env.reset()
        self.start_time = time.time()
        for self.global_step in tqdm(range(self.args.total_timesteps)):
            self.logs.reset()
            action = self.act(state)
            state_, reward, terminal = self.env.step(action)
            self.value_model.state, self.policy_model.state, loss, v_value = \
                self.fit(self.target_model_params, self.value_model.state, self.policy_model.state, state, action, reward, state_, terminal)
            state = state_
            self.logs.update(
                ['terminal_length', 'terminal_rewards', 'v_value', 'loss'],
                [self.env.get_terminal_length(), self.env.get_terminal_reward(), v_value, loss]
            )
            self.logs.writer_tensorboard(self.writer, self.global_step, drops=['v_value', 'loss'])
            if (self.global_step + 1) % self.args.target_model_update_frequency == 0:
                self.target_model_params = self.update_target_model(self.value_model.state.params, self.target_model_params)
            if (self.global_step + 1) % self.args.write_logs_frequency == 0:
                self.write_tensorboard()
            if (self.global_step + 1) % (self.args.total_timesteps // self.args.num_model_save) == 0 or self.global_step == self.args.total_timesteps - 1:
                print(f"Save weights at global step:", self.global_step)
                self.value_model.save_weights()
                self.policy_model.save_weights()
    
    def evaluate(self):
        state = self.env.reset()
        self.start_time = time.time()
        for self.global_step in tqdm(range(self.args.total_timesteps)):
            self.logs.reset()
            action = self.act(state)
            state_, _, terminal = self.env.step(action)
            state = state_
            self.logs.update(
                ['terminal_length', 'terminal_rewards'],
                [self.env.get_terminal_length(), self.env.get_terminal_reward()]
            )
            self.logs.writer_tensorboard(self.writer, self.global_step, drops=['v_value', 'loss'])
            if (self.global_step + 1) % self.args.write_logs_frequency == 0:
                self.write_tensorboard()
    
    def policy_model_predict(self, params, x):
        return self.policy_model.state.apply_fn(params, x)

    @partial(jax.jit, static_argnums=0)
    def _act(self, key, state_p:TrainState, state):
        key, subkey = jax.random.split(key)
        action_proba = state_p.apply_fn(state_p.params, state)[0]
        action = jax.random.choice(subkey, self.env.action_ndim, shape=(1,), p=action_proba)
        return key, action
    
    def act(self, state):
        self.key, action = self._act(self.key, self.policy_model.state, state)
        return jax.device_get(action)
    
    @partial(jax.jit, static_argnums=0)
    def fit(
        self, target_params, state_v:TrainState, state_p:TrainState,
        s, a, r, s_, t
    ):
        r += self.args.gamma * self.value_model.state.apply_fn(target_params, s_) * (1-t)
        def calc_v(params, s):
            return self.value_model.state.apply_fn(params, s)[0][0]
        v, g_v = jax.value_and_grad(calc_v)(state_v.params, s)
        delta = v - r[0]
        g_v = jax.tree_map(lambda x: x * delta, g_v)
        state_v = state_v.apply_gradients(grads=g_v)

        def calc_ln_p(params, s, a):
            proba = self.policy_model.state.apply_fn(params, s)[0][a[0]]
            return jnp.log(proba)
        g_p = jax.grad(calc_ln_p)(state_p.params, s, a)
        g_p = jax.tree_map(lambda x: x * delta, g_p)
        state_p = state_p.apply_gradients(grads=g_p)
        return state_v, state_p, jnp.square(delta), v
    
    def write_tensorboard(self):
        self.logs.writer_tensorboard(self.writer, self.global_step, drops=['terminal_length', 'terminal_rewards'])
        self.writer.add_scalar('charts/SPS_avg', int(self.global_step / (time.time()-self.start_time)), self.global_step)
        self.writer.add_scalar(
            'charts/learning_rate_v',
            self.value_model.state.opt_state[1]['learning_rate'].item(),
            self.global_step
        )
        self.writer.add_scalar(
            'charts/learning_rate_p',
            self.policy_model.state.opt_state[1]['learning_rate'].item(),
            self.global_step
        )
