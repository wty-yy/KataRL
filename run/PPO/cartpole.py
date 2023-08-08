from agents.PPO import PPO
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

def PPO_cartpole_train():
    start_idx = 0
    N = 1
    for idx in range(start_idx, start_idx + N):
        print(f"{idx}/{N}:")
        ppo = PPO(
            env=GymEnv(name='CartPole-v1', render_mode='rgb_array'),
            agent_name='PPO',
            model=MLP(lr=1e-3),
            agent_id=idx, iter_M=10
        )
        ppo.train()



