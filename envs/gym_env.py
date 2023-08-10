import gymnasium as gym
from envs import Env
import numpy as np

# If joining a Env, the following four parameters must be given.
max_step = {
    "CartPole-v1": 500
}
state_shape = {
    "CartPole-v1": (4,)
}
action_shape = {
    "CartPole-v1": (1,)
}
action_size = {
    "CartPole-v1": 2
}
rewards = {
    "positive": {
        "CartPole-v1": 1,
    },
    "negative": {
        "CartPole-v1": -20,
    }
}

class GymEnv(Env):
    """
    The OpenAI Gymnasium, create a new Env by the Env.name
    """
    
    def __init__(self, name, seed=None, num_envs=1, capture_video=False, **kwargs):
        if max_step.get(name) is None:
            raise Exception(f"Don't know the max_step of the environment: '{name}'")
        if state_shape.get(name) is None:
            raise Exception(f"Don't know the state_shape of the environment: '{name}'")
        if action_shape.get(name) is None:
            raise Exception(f"Don't know the action_shape of the environment: '{name}'")
        if action_size.get(name) is None:
            raise Exception(f"Don't know the action_size of the environment: '{name}'")
        if rewards['positive'].get(name) is None:
            raise Exception(f"Don't know the positive reward of the environment: '{name}'")
        if rewards['negative'].get(name) is None:
            raise Exception(f"Don't know the negative reward of the environment: '{name}'")

        super().__init__(
            name, seed=seed,
            num_envs=num_envs,
            capture_video=capture_video,
            max_step=max_step[name],
            state_shape=state_shape[name],
            action_shape=action_shape[name],
            action_size=action_size[name],
            **kwargs
        )
        if seed is not None: kwargs['seed'] = seed
        self.envs = gym.vector.AsyncVectorEnv([
            self.make_env(i) for i in range(self.num_envs)
        ])
        self.reset()
    
    def make_env(self, idx):
        def thunk():
            env = gym.make(self.name, render_mode='rgb_array')
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if self.capture_video:
                env = gym.wrappers.RecordVideo(
                    env, "logs/videos",
                    episode_trigger=lambda _: True,  # capture every episode
                    name_prefix=f"{idx}"
                )
            return env
        return thunk
    
    def step(self, action):
        if self.last_terminal is not None:
            self.step_count[self.last_terminal] = 0
        if not isinstance(action, list) and not isinstance(action, np.ndarray):
            action = [action]
        self.step_count += 1
        state, reward, terminal, truncated, _ = self.envs.step(action)
        terminal = terminal | truncated
        reward = \
            np.full_like(reward, rewards['positive'][self.name], dtype='float32')
        reward[terminal & (self.step_count != self.max_step)] = \
            rewards['negative'][self.name]
        if self.num_envs == 1:
            state, reward, terminal = state[0], reward[0], terminal[0]
        self.last_terminal = terminal.copy()
        return state, reward, terminal

    def reset(self):
        super().reset()  # reset step_count
        if self.seed is not None:
            state, _ = self.envs.reset(seed=self.seed)
        else: state, _ = self.envs.reset()
        if self.num_envs == 1: state = state[0]
        return state

    def render(self):
        return self.envs.render()
    
    def close(self):
        self.envs.close()
