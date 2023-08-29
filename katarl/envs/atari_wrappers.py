import gymnasium as gym
import numpy as np

"""
Wrappers learning from
`stable-baselines3/common/atari_wrappers.py`
"""

class FireResetWrapper(gym.Wrapper):
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
            return self.env.reset(**kwargs)
        return state, info
    
class EpisodeLifeWrapper(gym.Wrapper):
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