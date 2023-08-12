import numpy as np

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
            self, name, seed=None, num_envs=1, capture_video=False,
            max_step=None,
            state_shape=None, action_shape=None,
            action_size=None, **kwargs
        ):
        self.name, self.seed, self.num_envs, self.capture_video, self.max_step = \
            name, seed, num_envs, capture_video, max_step
        self.state_shape, self.action_shape, self.action_size = \
            state_shape, action_shape, action_size
        self.step_count = np.zeros(self.num_envs, dtype='int32')
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
        self.step_count = np.zeros(self.num_envs, dtype='int32')
        pass

    def render(self):
        """
        return: frame
        """
        pass
    
    def get_terminal_steps(self) -> list:
        return self.step_count[self.last_terminal].tolist()
