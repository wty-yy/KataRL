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
    -   max_step: The maximum step limit in Env.
    -   state_shape: State shape of Env.
    -   action_shape: Action shape of Env.
    -   step_count: Save the current episode length.

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
        self.name = args.env_name
        self.history = {
            'step_count': np.zeros(self.args.num_envs, dtype='int32'),
            'sum_reward': np.zeros(self.args.num_envs, dtype='float32'),
        }
        self.last_terminal = None
    
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
    
    def get_terminal_steps(self) -> list:
        return self.history['step_count'][self.last_terminal].tolist()

    def get_terminal_rewrad(self) -> list:
        return self.history['sum_reward'][self.last_terminal].tolist()
    
    def close(self):
        pass