class Activation():
    """
    Base class for activation functions.
    """

    def __init__(self):
        pass

    def forward(self, x):
        """
        Forward pass of the activation function.
        """
        raise NotImplementedError("Forward pass not implemented.")

    def backward(self, x):
        """
        Backward pass of the activation function.
        """
        raise NotImplementedError("Backward pass not implemented.")
    def __call__(self, x):
        """
        Call the forward pass of the activation function.
        """
        return self.forward(x)


class Sigmoid(Activation):
    pass

class Tanh(Activation):
    pass

class ReLU(Activation):
    pass

class Softmax(Activation):
    pass

