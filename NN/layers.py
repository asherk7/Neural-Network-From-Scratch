import numpy as np

class Layer():
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        """
        raise NotImplementedError("Forward method not implemented")

    def backward(self, grad_output):
        raise NotImplementedError("Backward method not implemented")

    def update(self, learning_rate):
        pass

class Dense(Layer):
    pass

class Conv2D(Layer):
    pass

class MaxPool2D(Layer):
    pass

class Flatten(Layer):
    pass
