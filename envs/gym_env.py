import gymnasium as gym
from envs import Env

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
    
    def __init__(self, name, seed=None, **kwargs):
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
            max_step=max_step[name],
            state_shape=state_shape[name],
            action_shape=action_shape[name],
            action_size=action_size[name],
            **kwargs
        )
        if seed is not None: kwargs['seed'] = seed
        self.env = gym.make(name, **kwargs)
        self.step_count = 0
    
    def step(self, action):
        self.step_count += 1
        state, reward, terminal, _, _ = self.env.step(action)
        if self.step_count != self.max_step and terminal:
            reward = rewards['negative']['CartPole-v1']
        else: reward = rewards['positive']['CartPole-v1']
        return state, reward, terminal
    
    def reset(self):
        if self.seed is not None:
            state, _ = self.env.reset(seed=self.seed)
        else: state, _ = self.env.reset()
        self.step_count = 0
        return state

    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
