class Env():
    """
    The virtual environment for Agent training.

    Initialize:
    -   name: The name of Env.
    -   seed: The random seed of Env at each 'reset'.
    -   max_step: The maximum step limit in Env.
    -   state_shape: State shape of Env.
    -   action_shape: Action shape of Env.

    Function:
    -   step(action): Do 'action' in Env.

    -   reset(): Reset Env.

    -   render(): Return the render frame of Env.
    """
    
    def __init__(
            self, name, seed=None, max_step=None,
            state_shape=None, action_shape=None, **kwargs
        ):
        self.name, self.seed, self.max_step = name, seed, max_step
        self.state_shape, self.action_shape = \
            state_shape, action_shape
    
    def step(self, action):
        """
        return: state, reward, terminal
        """
        pass
    
    def reset(self):
        """
        return: init_state
        """
        pass

    def render(self):
        """
        return: frame
        """
        pass
