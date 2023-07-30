from agents.constants import PATH
import tensorflow as tf
keras = tf.keras

class Model:

    def __init__(self, input_shape, load_id=None, verbose=True, **kwargs):
        self.input_shape, self.load_id = input_shape, load_id
        self.save_id = 0
        self.model = self.build()
        self.trainable_weights = self.model.trainable_weights
        if self.load_id is not None:
            self.load_weights()
            self.save_id = self.load_id + 1
        if verbose: self.plot_model()

    def plot_model(self):
        path = PATH.AGENT.joinpath('model.png')
        print(f"plot model struct png at '{path.absolute()}'")
        keras.utils.plot_model(self.model, to_file=path, show_shapes=True)
    
    def build(self) -> keras.Model:
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

