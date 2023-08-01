from agents.constants import PATH
import tensorflow as tf
keras = tf.keras

class Model:
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

    def __init__(self, lr=1e-3, load_id=None, verbose=True, **kwargs):
        self.lr, self.load_id, self.verbose = lr, load_id, verbose
        self.save_id = 0
        self.model = self.build_model()
        self.optimizer = self.build_optimizer(self.lr)
        if self.load_id is not None:
            self.load_weights()
            self.save_id = self.load_id + 1
        if verbose: self.plot_model()

    def plot_model(self):
        path = PATH.AGENT.joinpath('model.png')
        print(f"plot model struct png at '{path.absolute()}'")
        keras.utils.plot_model(self.model, to_file=path, show_shapes=True)
    
    def build_model(self) -> keras.Model:
        pass
    
    def build_optimizer(self, lr) -> keras.optimizers.Optimizer:
        pass

    def __call__(self, X):
        return self.model(X)
    
    def save_weights(self):
        path = PATH.CHECKPOINTS.joinpath(f"{self.save_id:04}")
        self.model.save_weights(path)
        self.save_id += 1
    
    def load_weights(self):
        path = PATH.CHECKPOINTS.joinpath(f"{self.load_id:04}")
        print(f"Load weight from '{path.absolute()}'")
        self.model.load_weights(path)
    
    def get_trainable_weights(self):
        return self.model.trainable_weights
    
    def apply_gradients(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

