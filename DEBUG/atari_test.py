import gymnasium as gym
import numpy as np

def make_env(idx):
    def thunk() -> gym.Env:
        env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(
            env, "videos", episode_trigger=lambda _: True,
            name_prefix=f"{idx}")
        return env
    return thunk

N = 1

# envs = gym.vector.AsyncVectorEnv([
#     make_env(i) for i in range(N)
# ])

class FireResetEnv(gym.Wrapper):
    """
    Some Atari game need "Fire" action to start game.
    Such like: Breakout
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        state, _, terminal, truncated, info = self.env.step(1)
        if terminal or truncated:
            self.env.reset(**kwargs)
        return state, info
    
class EpisodeLifeEnv(gym.Wrapper):
    """
    Most atari games have many lives, you can get current 
    lives info from: `info['lives']` or `env.unwrapped.ale.lives()`.
    One life loss means one episode terminal, but only 
    all lives used, the env will be reseted, that is `game over`.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.game_over = True
    
    def step(self, action):
        state, reward, terminal, truncated, info = self.env.step(action)
        self.game_over = terminal | truncated
        lives = info['lives']
        # Loss one life, episode is terminal
        if 0 < lives < self.lives:
            terminal = True
        self.lives = lives  # update lives
        return state, reward, terminal, truncated, info
    
    def reset(self, **kwargs):
        if self.game_over:
            # No lives anymore, you need reset game.
            state, info = self.env.reset(**kwargs)
        else:
            # Still have lives, wait for next episode.
            state, _, terminal, truncated, info = self.env.step(1)
            if terminal or truncated:
                state, info = self.env.reset(**kwargs)
        self.lives = info['lives']
        return state, info

env = make_env(1)()
env = FireResetEnv(env)
env = EpisodeLifeEnv(env)
# env = gym.wrappers.FrameStack(env, 4)
env.reset()
rewards = 0
for _ in range(1000):
    action = env.action_space.sample()
    # print(action)
    state, reward, terminal, truncated, info = env.step(action)
    # print(state.shape)
    # print(reward)
    rewards += reward
    # print(info)
    # print(env.unwrapped.ale.lives())
    terminal |= truncated
    if terminal:
        print(rewards)
        rewards = 0
        env.reset()
    # print(terminal, truncated)
env.close()