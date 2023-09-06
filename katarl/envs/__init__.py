import numpy as np
from typing import NamedTuple

class Env():
    """
    The virtual environment for Agent training.

    Initialize:
    -   name: The name of Env.
    -   seed: The random seed of Env at each 'reset'.
    -   num_envs: The number of environment in AsyncEnvs.
    -   capture_video: Save frames to mp4.
    -   state_shape: State shape of Env.
    -   action_shape: Action shape of Env.
    -   sum_length: Save the current terminal length.
    -   sum_reward: Save the current terminal length.

    Function:
    -   step(action): Do 'action' in Env.

    -   reset(): Reset Env.

    -   render(): Return the render frame of Env.
    """

    def __init__(
            self,
            args: NamedTuple = None,
            state_shape: tuple = None,
            action_shape: tuple = None,
            action_ndim: int = None,
        ):
        self.args, self.state_shape, self.action_shape, self.action_ndim = args, state_shape, action_shape, action_ndim
        self.name, self.num_envs = args.env_name, self.args.num_envs if vars(args).get('num_envs') else 1
        self.history = {
            'sum_length': np.zeros(self.num_envs, dtype='int32'),
            'sum_reward': np.zeros(self.num_envs, dtype='float32'),
        }
        self.last_terminal, self.last_info = None, None
    
    def step(self, action):
        """
        return: state, reward, terminal
        """
        pass
    
    def reset(self):
        """
        return: init_state
        """
        for key, value in self.history.items():
            self.history[key] = np.zeros_like(value)

    def render(self):
        """
        return: frame
        """
        pass

    def add_history(self, keys, values):
        for key, value in zip(keys, values):
            self.history[key] += value
    
    def reset_history(self):
        if self.last_terminal is not None:
            for key in self.history.keys():
                self.history[key][self.last_terminal] = 0
    
    def get_terminal_length(self) -> list:
        return self.history['sum_length'][self.last_terminal].tolist()

    def get_terminal_reward(self) -> list:
        return self.history['sum_reward'][self.last_terminal].tolist()
    
    def get_info(self, key) -> list:
        ret = []
        if 'final_info' in self.last_info.keys():
            for info in self.last_info['final_info']:
                if info is not None and 'episode' in info.keys():
                    ret += info['episode'][key].tolist()
        return ret

    def get_episode_length(self) -> list:
        return self.get_info('l')
        
    def get_episode_reward(self) -> list:
        return self.get_info('r')
    
    def close(self):
        pass