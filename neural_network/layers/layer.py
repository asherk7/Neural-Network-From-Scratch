import numpy as np

class Layer:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        """
        raise NotImplementedError("Forward method not implemented")

    def backward(self, grad_output):
        """
        Backward pass through the layer. 
        Computes the gradient of the loss with respect to the weight, bias, and input of the layer. 
        """
        raise NotImplementedError("Backward method not implemented")
    
    def update(self, learning_rate):
        """
        Update the layer's parameters using the gradients.
        """
        raise NotImplementedError("Update method not implemented")