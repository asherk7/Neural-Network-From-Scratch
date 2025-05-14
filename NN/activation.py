import numpy as np

class Activation():
    """
    Base class for activation functions.
    
    Methods:
        forward(x: np.ndarray) -> np.ndarray:
            Applies the activation function to the input data (W*X + b).
        backward(gradient: np.ndarray) -> np.ndarray:
            Multiplies the gradient being backpropagated by the derivative of the activation function
            which is evaluated at the input data (W*X + b).
    """
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the activation function.
        Args:
            x (np.ndarray): Input data.
        Returns:
            np.ndarray: Output data after applying the activation function.
        """
        raise NotImplementedError("Forward method not implemented")

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass through the activation function.
        Args:
            gradient: Gradient of the next layer being backpropagated with respect to the loss.
        Returns:
            np.ndarray: Gradient of the current layer with respect to the loss.
        """
        raise NotImplementedError("Backward method not implemented")

class Linear(Activation):
    """
    Linear activation function.
    
    This function is defined as f(x) = x, which means it does not change the input.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        return gradient

class Sigmoid(Activation):
    """
    Sigmoid activation function.
    
    This function maps the input to a value between 0 and 1.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x # Store the input for backward pass
        return 1 / (1 + np.exp(-x))

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        #sigmoid = self.forward(gradient)
        #return gradient * sigmoid * (1 - sigmoid)
        sigmoid = (np.exp(self.x) / ((1 + np.exp(self.x))**2))
        return gradient * sigmoid

class ReLU(Activation):
    """
    ReLU (Rectified Linear Unit) activation function.
    
    This function maps negative values to 0 and positive values to themselves.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x # Store the input for backward pass
        return np.maximum(0, x)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        relu = self.x
        relu[relu > 0] = 1
        relu[relu <= 0] = 0
        return gradient * relu

class LeakyReLU(Activation):
    """
    Leaky ReLU activation function.

    This function allows a small, non-zero gradient when the input is negative.
    """
    pass

class Tanh(Activation):
    """
    Tanh activation function.
    This function maps the input to a value between -1 and 1.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x # Store the input for backward pass
        return np.tanh(x)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        tanh = (1 - np.tanh(self.x)**2)
        return gradient * tanh

class Softmax(Activation):
    """
    Softmax activation function.
    This function converts the input into a probability distribution.
    """
    pass

