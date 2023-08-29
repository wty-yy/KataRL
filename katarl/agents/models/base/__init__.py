from katarl.agents.constants import PATH
from typing import NamedTuple

class BaseModel:
    """
    Model base class.
    You just need rewrite build function, 
    and instantiate this class when pass to the Agent.

    Initialize:
    -   load_id: Load the model from 'PATH.CHECKPOINTS/cp-{load_id}'.
    -   lr: The learning rate of optimizer.
    -   verbose: Whether print the model struct figure.
    -   save_id: save_id += 1 at each call of save_weights().
    -   model: Return Model from build_model().
    -   optimizer: Return Optimizer from build_optimizer().

    Function:
    -   build_model(): **Return the model the Agent using.**

    -   build_optimizer(): **Return the optimizer the model using.**

    -   plot_model():
        Use keras.utils.plot_model to plot model,
        it will be called at init when verbose=True.
    
    -   __call__(X): Return self.model(X)

    -   save_weights():
        Save model's weights at
        'PATH.CHECKPOINTS/cp-{save_id}', and save_id += 1
    
    -   load_weights():
        Load model's weights at 'PATH.CHECKPOINTS/cp-{load_id}',
        be auto called if load_id is not None.
    """

    def __init__(
            self, name='model',
            input_shape=None, output_ndim=None,
            args: NamedTuple = None
        ):
        self.name, self.input_shape, self.output_ndim, self.args = name, input_shape, output_ndim, args
        self.set_seed()
        self.save_id = 0
        self.model = self.build_model()
        if self.args.load_id is not None:
            self.save_id = self.args.load_id + 1
        self.plot_model(PATH.FIGURES.joinpath(f'{self.name}.png'))

        self.load_path = None
        if self.args.load_name is not None and self.args.load_id is not None:
            self.load_path = PATH.LOGS.joinpath(self.args.load_name+"/checkpoints").joinpath(f"{self.name}-{self.args.load_id:04}")

    def plot_model(self, path):
        pass
    
    def build_model(self):
        pass
    
    def __call__(self, X):
        pass
    
    def save_weights(self):
        pass
    
    def load_weights(self):
        pass
    
    def set_seed(self):
        pass
