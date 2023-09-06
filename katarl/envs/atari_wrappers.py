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
    
    def __init__(self, env: gym.Env, anneal_reward=False):
        super().__init__(env)
        self.lives = 0
        self.game_over = True
        self.anneal_reward = anneal_reward
        
    
    def step(self, action):
        state, reward, terminal, truncated, info = self.env.step(action)
        self.game_over = terminal | truncated
        lives = info['lives']
        # Loss one life, episode is terminal
        if 0 < lives < self.lives:
            terminal = True
        self.lives = lives  # update lives
        if self.anneal_reward: reward *= 1 / (6-self.lives)
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

class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)

    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._state_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, done, information
        """
        total_reward = 0.0
        for i in range(self._skip):
            # obs, reward, done, info = self.env.step(action)
            state, reward, terminal, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._state_buffer[0] = state
            if i == self._skip - 1:
                self._state_buffer[1] = state
            total_reward += reward
            if terminal or truncated:
                break
        max_frame = self._state_buffer.max(axis=0)

        return max_frame, total_reward, terminal, truncated, info