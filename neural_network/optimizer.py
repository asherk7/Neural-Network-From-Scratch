import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update_weights(self, layer):
        """
        Update the weights of the layer using the optimizer's learning rate.
        """
        raise NotImplementedError("Update weights method not implemented")
    
    def update_biases(self, layer):
        """
        Update the biases of the layer using the optimizer's learning rate.
        """
        raise NotImplementedError("Update biases method not implemented")

class SGD():
    """
    Stochastic Gradient Descent (SGD) optimizer.
    Args:
        learning_rate (float): Learning rate for the optimizer.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_weights(self, layer):
        """
        Update the weights of the layer using the optimizer's learning rate.
        """
        layer.weights -= self.learning_rate * layer.gradient_weights

    def update_biases(self, layer):
        """
        Update the biases of the layer using the optimizer's learning rate.
        """
        layer.biases -= self.learning_rate * layer.gradient_biases

class Adam():
    """
    Adam optimizer.
    Args:
        learning_rate (float): Learning rate for the optimizer.
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = 0
        self.v = 0
    
    def update_weights(self, layer):
        """
        Update the weights of the layer using the optimizer's learning rate.
        """
        self.m = self.beta1 * self.m + (1 - self.beta1) * layer.gradient_weights
        self.v = self.beta2 * self.v + (1 - self.beta2) * (layer.gradient_weights ** 2)
        m_hat = self.m / (1 - self.beta1)
        v_hat = self.v / (1 - self.beta2)
        layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

    def update_biases(self, layer):
        """
        Update the biases of the layer using the optimizer's learning rate.
        """
        self.m = self.beta1 * self.m + (1 - self.beta1) * layer.gradient_biases
        self.v = self.beta2 * self.v + (1 - self.beta2) * (layer.gradient_biases ** 2)
        m_hat = self.m / (1 - self.beta1)
        v_hat = self.v / (1 - self.beta2)
        layer.biases -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
