import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update_weights(self, layer, gradient_weights):
        """
        Update the weights of the layer using the optimizer's learning rate.
        """
        raise NotImplementedError("Update weights method not implemented")
    
    def update_biases(self, layer, gradient_biases):
        """
        Update the biases of the layer using the optimizer's learning rate.
        """
        raise NotImplementedError("Update biases method not implemented")

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    Args:
        learning_rate (float): Learning rate for the optimizer.
    """
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def update_weights(self, layer, gradient_weights):
        layer.weights -= self.learning_rate * gradient_weights

    def update_biases(self, layer, gradient_biases):
        layer.biases -= self.learning_rate * gradient_biases

class Adam(Optimizer):
    """
    Adam optimizer.
    Args:
        learning_rate (float): Learning rate for the optimizer.
        beta1 (float): Exponential decay rate for the moving average of the gradient.
        beta2 (float): Exponential decay rate for the moving average of the squared gradient.
        epsilon (float): Small constant to prevent division by zero.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = {}
        self.v_w = {}
        self.m_b = {}
        self.v_b = {}
        self.t = {}

    def _init_moments(self, layer):
        layer_id = id(layer)
        if layer_id not in self.m_w:
            self.m_w[layer_id] = np.zeros_like(layer.weights)
            self.v_w[layer_id] = np.zeros_like(layer.weights)
            self.m_b[layer_id] = np.zeros_like(layer.biases)
            self.v_b[layer_id] = np.zeros_like(layer.biases)
            self.t[layer_id] = 0

    def update_weights(self, layer, gradient_weights):
        layer_id = id(layer)
        self._init_moments(layer)
        self.t[layer_id] += 1

        self.m_w[layer_id] = self.beta1 * self.m_w[layer_id] + (1 - self.beta1) * gradient_weights
        self.v_w[layer_id] = self.beta2 * self.v_w[layer_id] + (1 - self.beta2) * (gradient_weights ** 2)

        m_hat = self.m_w[layer_id] / (1 - self.beta1 ** self.t[layer_id])
        v_hat = self.v_w[layer_id] / (1 - self.beta2 ** self.t[layer_id])

        layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def update_biases(self, layer, gradient_biases):
        layer_id = id(layer)
        self._init_moments(layer)

        self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * gradient_biases
        self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * (gradient_biases ** 2)

        m_hat = self.m_b[layer_id] / (1 - self.beta1 ** self.t[layer_id])
        v_hat = self.v_b[layer_id] / (1 - self.beta2 ** self.t[layer_id])

        layer.biases -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)