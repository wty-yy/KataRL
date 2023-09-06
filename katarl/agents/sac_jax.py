from katarl.agents import BaseAgent
from katarl.agents.models.base.base_jax import JaxModel
from katarl.utils.logs import Logs, MeanMetric
from katarl.envs import Env

from typing import NamedTuple
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import jax
import numpy as np
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.struct import dataclass
import optax
from functools import partial

def init_logs() -> Logs:
    return Logs(
        init_logs = {
            'episode_step': MeanMetric(),
            'episode_return': MeanMetric(),
            'q1_loss': MeanMetric(),
            'q2_loss': MeanMetric(),
            'q1_value': MeanMetric(),
            'q2_value': MeanMetric(),
            'p_loss': MeanMetric(),
            'entropy': MeanMetric(),
        },
        folder2name = {
            'charts': ['episode_step', 'episode_return'],
            'metrics': ['q1_loss', 'q2_loss', 'q1_value', 'q2_value', 'p_loss', 'entropy']
        }
    )

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

    def __init__(self, state_shape, action_shape, args: NamedTuple):
        self.state_shape, self.action_shape, self.num_envs, self.batch_size = state_shape, action_shape, args.num_envs, args.batch_size
        self.count = 0
        self.sample_row = (self.batch_size-1) // self.num_envs + 1
        self.sample_size = self.sample_row * self.num_envs
        self.row = (args.memory_size-1) // self.num_envs + 1
        self.memory = [
            np.zeros([self.row, self.num_envs, *self.state_shape], dtype='float32'),
            np.zeros([self.row, self.num_envs], dtype='int32'),
            np.zeros([self.row, self.num_envs], dtype='float32'),
            np.zeros([self.row, self.num_envs, *self.state_shape], dtype='float32'),
            np.zeros([self.row, self.num_envs], dtype='bool')
        ]
    
    def update(self, items:tuple):  # (S,A,R,S',T)
        start = self.count % self.row
        for a, item in zip(self.memory, items):
            a[start] = item
        self.count += 1

    def sample(self):
        idxs = np.random.choice(min(self.count, self.row), self.sample_row)
        return [a[idxs].reshape(self.sample_size, -1) for a in self.memory]
    
@dataclass
class ModelState:
    q1: TrainState
    q2: TrainState
    p: TrainState
    q1_target_params: dict
    q2_target_params: dict
    log_alpha: TrainState

class Agent(BaseAgent):

    def __init__(
            self,
            agent_name: str = None,
            env: Env = None,
            writer: SummaryWriter = None, 
            args: NamedTuple = None,
            # models
            q1_model: JaxModel = None,
            q2_model: JaxModel = None,
            p_model: JaxModel = None,
        ):
        models = [q1_model, q2_model, p_model]
        super().__init__(agent_name, env, models, writer, args)

        self.key = jax.random.PRNGKey(self.args.seed)  # action choose
        np.random.seed(self.args.seed)  # memory sample
        
        self.logs = init_logs()
        self.memory = MemoryCache(env.state_shape, env.action_shape, args)

        self.model = ModelState(
            q1=q1_model.state,
            q2=q2_model.state,
            p=p_model.state,
            q1_target_params=q1_model.state.params.copy(),
            q2_target_params=q2_model.state.params.copy(),
            log_alpha=TrainState.create(
                apply_fn=None,
                params=[np.log(self.args.alpha)],
                tx=optax.adam(learning_rate=self.args.learning_rate_alpha)
            )
        )
    
    @partial(jax.jit, static_argnums=0)
    def update_target_model(self, current_params, target_params):
        return jax.tree_map(
            lambda x, y: self.args.tau * x + (1-self.args.tau) * y, 
            current_params, target_params
        )
    
    def train(self):
        state = self.env.reset()
        self.start_time, self.global_step = time.time(), 0
        for i in tqdm(range(self.args.num_iters)):
            self.logs.reset()
            action = self.act(state)
            state_, reward, terminal = self.env.step(action)
            self.memory.update((state, action, reward, state_, terminal))
            self.global_step += self.args.num_envs

            if self.global_step > self.args.start_fit_size:
                if self.global_step % self.args.train_frequency < self.args.num_envs:
                    batch = self.memory.sample()

                    self.model, metrics = self.fit(self.model, *batch)
                    self.logs.update(['q1_loss', 'q2_loss', 'q1_value', 'q2_value', 'p_loss', 'entropy'], metrics)
                    
                if self.global_step % self.args.target_model_update_frequency < self.args.num_envs:
                    self.model = self.model.replace(
                        q1_target_params = self.update_target_model(self.model.q1.params, self.model.q1_target_params),
                        q2_target_params = self.update_target_model(self.model.q2.params, self.model.q2_target_params),
                    )

            state = state_
            self.logs.update(
                ['episode_step', 'episode_return'],
                [self.env.get_terminal_length(), self.env.get_terminal_reward()]
            )
            self.logs.writer_tensorboard(self.writer, self.global_step, drops=['q_value', 'loss'])

            if (i+1) % self.args.write_logs_frequency == 0:
                self.write_tensorboard()
            if (i+1) % (self.args.num_iters // self.args.num_model_save) == 0 or i == self.args.num_iters - 1:
                print(f"Save weights at global step:", self.global_step)
                for model in self.models: model.save_weights()

    def evaluate(self):
        self.epsilon = 0
        state = self.env.reset()
        self.start_time, self.global_step = time.time(), 0
        for i in tqdm(range(self.args.num_iters)):
            self.logs.reset()
            action = self.act(state)
            state_, _, _= self.env.step(action)
            self.global_step += self.args.num_envs
            state = state_
            self.logs.update(
                ['episode_step', 'episode_return'],
                [self.env.get_terminal_length(), self.env.get_terminal_reward()]
            )
            self.logs.writer_tensorboard(self.writer, self.global_step, drops=['q_value', 'loss'])

            if (i+1) % self.args.write_logs_frequency == 0:
                self.write_tensorboard()
    
    @partial(jax.jit, static_argnums=0)
    def get_model_action(self, key, params, x):
        logits = self.model.p.apply_fn(params, x)
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=-1)
        return key, action

    def act(self, s):  # epsilon choice policy
        self.key, action = self.get_model_action(self.key, self.model.p.params, s)
        return jax.device_get(action)

    @partial(jax.jit, static_argnums=0)
    def fit(self, model:ModelState, s, a, r, s_, t):
        a, r, t = a.flatten(), r.flatten(), t.flatten()
        alpha = jnp.exp(model.log_alpha.params[0])
        # 1. q value part
        logits = model.p.apply_fn(model.p.params, s_)
        q1_target = model.q1.apply_fn(model.q1_target_params, s_)
        q2_target = model.q2.apply_fn(model.q2_target_params, s_)
        td_target = r + (1-t) * self.args.gamma * (jax.nn.softmax(logits) * (
            jnp.minimum(q1_target, q2_target) - alpha * jax.nn.log_softmax(logits)
        )).sum(-1)
        def q_loss_fn(params, apply_fn, x, td_target, action):
            q_pred = apply_fn(params, x)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), action]
            return ((q_pred - td_target) ** 2 / 2).mean(), q_pred
        (q1_loss, q1_value), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(model.q1.params, model.q1.apply_fn, s, td_target, a)
        (q2_loss, q2_value), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(model.q2.params, model.q2.apply_fn, s, td_target, a)
        model = model.replace(
            q1 = model.q1.apply_gradients(grads=q1_grads),
            q2 = model.q2.apply_gradients(grads=q2_grads)
        )
        # 2. policy part
        q1 = model.q1.apply_fn(model.q1.params, s)
        q2 = model.q2.apply_fn(model.q2.params, s)
        def p_loss_fn(params, q_min, alpha):
            logits = model.p.apply_fn(params, s)
            proba = jax.nn.softmax(logits)
            log_p = jax.nn.log_softmax(logits)
            entropy = -(proba * log_p).sum(-1).mean()
            return (proba * (alpha * log_p - q_min)).sum(-1).mean(), entropy
        (p_loss, entropy), p_grads = jax.value_and_grad(p_loss_fn, has_aux=True)(model.p.params, jnp.minimum(q1, q2), alpha)
        model = model.replace(p=model.p.apply_gradients(grads=p_grads))
        # 3. autotune alpha
        if self.args.flag_autotune_alpha:
            alpha_grad = (entropy - self.args.coef_target_entropy * jnp.log(self.env.action_ndim))
            model = model.replace(log_alpha=model.log_alpha.apply_gradients(grads=[alpha_grad]))
        return model, (q1_loss, q2_loss, q1_value, q2_value, p_loss, entropy)
    
    def write_tensorboard(self):
        self.logs.writer_tensorboard(self.writer, self.global_step, drops=['episode_step', 'episode_return'])
        self.writer.add_scalar('charts/alpha', np.exp(self.model.log_alpha.params[0]), self.global_step)
        self.writer.add_scalar(
            'charts/SPS_avg',
            int(self.global_step / (time.time() - self.start_time)),
            self.global_step
        )
    