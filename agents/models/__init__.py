from agents.constants import PATH
import tensorflow as tf
keras = tf.keras

class BasicModel:
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
            self, lr=1e-3, load_name=None, load_id=None, verbose=True, name='model', 
            input_shape=None, output_ndim=None,
            **kwargs
        ):
        self.lr, self.load_name, self.load_id, self.verbose, self.name, \
        self.input_shape, self.output_ndim = \
            lr, load_name, load_id, verbose, name, input_shape, output_ndim
        self.save_id = 0
        self.model = self.build_model()
        self.optimizer = self.build_optimizer(self.lr)
        if self.load_id is not None:
            self.save_id = self.load_id + 1
        if verbose: self.plot_model(); self.model.summary()

        self.load_path = None
        if self.load_name is not None and self.load_id is not None:
            self.load_path = PATH.LOGS.joinpath(self.load_name).joinpath(f"{self.load_id:04}")

    def plot_model(self):
        path = PATH.FIGURES.joinpath(f'{self.name}.png')
        print(f"plot model struct png at '{path.absolute()}'")
        keras.utils.plot_model(self.model, to_file=path, show_shapes=True)
    
    def build_model(self) -> keras.Model:
        pass
    
    def build_optimizer(self, lr) -> keras.optimizers.Optimizer:
        pass

    def __call__(self, X):
        return self.model(X)
    
    def save_weights(self, prefix_name=""):
        if len(prefix_name) != 0: prefix_name += '-'
        path = PATH.CHECKPOINTS.joinpath(prefix_name+f"{self.save_id:04}")
        self.model.save_weights(path)
        self.save_id += 1
    
    def load_weights(self):
        if self.load_path is None: return
        print(f"Load weight from '{self.load_path.absolute()}'")
        self.model.load_weights(self.load_path)
    
    def get_trainable_weights(self):
        return self.model.trainable_weights
    
    def apply_gradients(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

