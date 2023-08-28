import gymnasium as gym
from envs import Env
import numpy as np
from envs.atari_wrappers import (
    FireResetWrapper, 
    EpisodeLifeWrapper,
)

INF = int(1e18)

atari_envs = [
    "Breakout-v4",
    "ALE/Breakout-v5"
]

# If joining a Env, the following four parameters must be given.
max_step = {
    "CartPole-v1": 500,
    "Breakout-v4": INF,
    "ALE/Breakout-v5": INF,
}
state_shape = {
    "CartPole-v1": (4,),
    # DELIT: Atari env has GrayColor and 4 frames stack
    "Breakout-v4": (210, 160, 3),
    "ALE/Breakout-v5": (210, 160, 3),
}
action_shape = {
    "CartPole-v1": (1,),
    "Breakout-v4": (1,),
    "ALE/Breakout-v5": (1,),
}
action_ndim = {
    "CartPole-v1": 2,
    "Breakout-v4": 4,
    "ALE/Breakout-v5": 4,
}
rewards = {
    "positive": {
        "CartPole-v1": 1,
        "Breakout-v4": None,
        "ALE/Breakout-v5": None,
    },
    "negative": {
        "CartPole-v1": 1,
        "Breakout-v4": None,
        "ALE/Breakout-v5": None,
    }
}

class GymEnv(Env):
    """
    The OpenAI Gymnasium, create a new Env by the Env.name
    """
    
    def __init__(
            self, name, seed=1, num_envs=1,
            capture_video=False,
            **kwargs
        ):
        if max_step.get(name) is None:
            raise Exception(f"Don't know the max_step of the environment: '{name}'")
        if state_shape.get(name) is None:
            raise Exception(f"Don't know the state_shape of the environment: '{name}'")
        if action_shape.get(name) is None:
            raise Exception(f"Don't know the action_shape of the environment: '{name}'")
        if action_ndim.get(name) is None:
            raise Exception(f"Don't know the action_size of the environment: '{name}'")
        # if rewards['positive'].get(name) is None:
        #     raise Exception(f"Don't know the positive reward of the environment: '{name}'")
        # if rewards['negative'].get(name) is None:
        #     raise Exception(f"Don't know the negative reward of the environment: '{name}'")

        super().__init__(
            name, seed=seed,
            num_envs=num_envs,
            capture_video=capture_video,
            max_step=max_step[name],
            state_shape=state_shape[name],
            action_shape=action_shape[name],
            action_ndim=action_ndim[name],
            **kwargs
        )
        self.use_atari_wrapper = self.name in atari_envs
        if seed is not None: kwargs['seed'] = seed
        self.envs = gym.vector.SyncVectorEnv([  # FIX: AsyncVectorEnv is slower than SyncVectorEnv
            self.make_env(i) for i in range(self.num_envs)
        ])
        if kwargs.get('neg_rewards'):
            self.neg_rewards = kwargs['neg_rewards']
        else: self.neg_rewards = rewards['negative'][self.name]
        self.reset()
    
    def make_env(self, idx):
        def thunk():
            if self.capture_video and idx == 0:
                env = gym.make(self.name, render_mode='rgb_array')  # FIX: render_mods is slower
                env = gym.wrappers.RecordVideo(
                    env, "logs/videos",
                    episode_trigger=lambda episode: True if (episode+1)%10==0 or episode==0 else False,  # capture 10,20,...
                    name_prefix=f"{idx}"
                )
            else:
                env = gym.make(self.name)
            env.action_space.seed(self.seed)
            if self.use_atari_wrapper:
                env = FireResetWrapper(env)
                env = EpisodeLifeWrapper(env)
                # env = gym.wrappers.GrayScaleObservation(env)
                # env = gym.wrappers.FrameStack(env, 4)
            return env
        return thunk
    
    def step(self, action):
        self.reset_history()
        state, reward, terminal, truncated, _ = self.envs.step(action)
        terminal = terminal | truncated
        if rewards['positive'][self.name] is not None:
            reward = \
                np.full_like(reward, rewards['positive'][self.name], dtype='float32')
        if self.neg_rewards is not None:
            reward[terminal ^ truncated] = self.neg_rewards
        self.add_history(['step_count', 'sum_reward'], [1, reward])
        state = self.check_state_shape(state)
        self.last_terminal = terminal
        return state, reward, terminal
    
    def reset(self):
        super().reset()  # reset step_count
        state, _ = self.envs.reset(
            seed=[self.seed+i for i in range(self.num_envs)]
        )
        # if self.num_envs == 1: state = state[0]
        # print("state.shape=",state.shape)
        state = self.check_state_shape(state)
        return state
    
    def check_state_shape(self, state):
        if self.use_atari_wrapper:
            # (8, 4, 210, 160) -> (8, 210, 160, 4)
            state = state.reshape(self.num_envs, *self.state_shape)
        return state

    def render(self):
        return self.envs.render()
    
    def close(self):
        self.envs.close()
