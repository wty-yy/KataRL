from agents.PPO import PPO
from agents.constants.PPO import actor_N
from agents.models import Model, keras
from envs.gym_env import GymEnv
import tensorflow as tf
keras = tf.keras
layers = keras.layers

class MLP(Model):

    def __init__(
            self, lr=0.001, load_id=None, verbose=True,
            name='model', **kwargs
        ):
        super().__init__(lr, load_id, verbose, name, **kwargs)

    def build_model(self):
        inputs = layers.Input(shape=(4,), name='State')
        x = layers.Dense(128, activation='relu', name='Dense1')(inputs)
        x = layers.Dense(64, activation='relu', name='Dense2')(x)
        x = layers.Dense(16, activation='relu', name='Dense3')(x)
        value = layers.Dense(1, name='State-Value')(x)
        x = layers.Dense(128, activation='relu', name='Dense4')(inputs)
        x = layers.Dense(64, activation='relu', name='Dense5')(x)
        x = layers.Dense(16, activation='relu', name='Dense6')(x)
        proba = layers.Dense(2, activation='softmax', name='Action-Proba')(x)
        return keras.Model(inputs, [value, proba], name=self.name)

    def build_optimizer(self, lr):
        return keras.optimizers.Adam(learning_rate=lr)

def build_env():
    return GymEnv(name='CartPole-v1', render_mode='rgb_array')

def PPO_cartpole_train():
    start_idx = 0
    N = 10
    for idx in range(start_idx, start_idx + N):
        print(f"{idx}/{N}:")
        ppo = PPO(
            env=GymEnv(name='CartPole-v1', num_envs=actor_N),
            agent_name='PPO-no',
            model=MLP(lr=3e-4),
            agent_id=idx
        )
        ppo.train()

def PPO_cartpole_eval(agent_id, load_id, frames_M=int(1e4)):
    ppo = PPO(
        env=GymEnv(
            name='CartPole-v1', num_envs=actor_N,
            capture_video=True
        ),
        agent_name='PPO',
        model=MLP(lr=3e-4, load_id=load_id),
        agent_id=agent_id,
        frames_M=frames_M
    )
    ppo.evaluate()