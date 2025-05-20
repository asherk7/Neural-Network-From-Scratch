import numpy as np

class Optimizer:
    """
    Base class for optimization algorithms.

    This class defines the interface for optimizers that update weights
    and biases of a layer during training.
    
    Attributes:
        learning_rate (float): Step size used for updates.
    """

    def __init__(self, learning_rate=0.01):
        """
        Initializes the optimizer with a learning rate.

        Args:
            learning_rate (float): The step size for parameter updates.
        """
        self.learning_rate = learning_rate

    def update_weights(self, layer, gradient_weights):
        """
        Updates the weights of a layer (abstract method).

        Args:
            layer: The layer whose weights are to be updated.
            gradient_weights: The gradient of the loss w.r.t. the weights.

        Raises:
            NotImplementedError: If method is not implemented in a subclass.
        """
        raise NotImplementedError("Update weights method not implemented")

    def update_biases(self, layer, gradient_biases):
        """
        Updates the biases of a layer (abstract method).

        Args:
            layer: The layer whose biases are to be updated.
            gradient_biases: The gradient of the loss w.r.t. the biases.

        Raises:
            NotImplementedError: If method is not implemented in a subclass.
        """
        raise NotImplementedError("Update biases method not implemented")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    SGD updates model parameters by moving them in the direction
    opposite to the gradient of the loss with respect to each parameter.
    This is done by subtracting the parameter by the gradient multiplied by the learning rate.
    """

    def __init__(self, learning_rate=0.01):
        """
        Initializes SGD with a learning rate.

        Args:
            learning_rate (float): The step size for parameter updates.
        """
        super().__init__(learning_rate)

    def update_weights(self, layer, gradient_weights):
        """
        Applies gradient descent to update weights.

        Args:
            layer: The layer whose weights are being updated.
            gradient_weights: The gradient of the loss w.r.t. the weights.
        """
        layer.weights -= self.learning_rate * gradient_weights

    def update_biases(self, layer, gradient_biases):
        """
        Applies gradient descent to update biases.

        Args:
            layer: The layer whose biases are being updated.
            gradient_biases: The gradient of the loss w.r.t. the biases.
        """
        layer.biases -= self.learning_rate * gradient_biases


class Adam(Optimizer):
    """
    Adam optimizer.

    Adam (Adaptive Moment Estimation) combines SGD and RMSProp. It maintains running
    averages (average of recent gradients) of both the gradients (first moment) 
    and the squared gradients (second moment), which are used to adapt the learning rate 
    for each parameter individually. These averages are bias-corrected to account
    for their initialization at zero. This allows Adam to converge faster
    and more reliably on deep networks, especially in sparse or noisy settings.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initializes Adam optimizer with hyperparameters.

        Args:
            learning_rate (float): Base step size for updates.
            beta1 (float): Decay rate for the moving average of gradients.
            beta2 (float): Decay rate for the moving average of squared gradients.
            epsilon (float): A small constant to improve numerical stability (prevent division by zero).
        """
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
        """
        Initializes moment vectors for a new layer.

        Args:
            layer: The layer for which to initialize moments.
        """
        layer_id = id(layer)
        if layer_id not in self.m_w:
            self.m_w[layer_id] = np.zeros_like(layer.weights)
            self.v_w[layer_id] = np.zeros_like(layer.weights)
            self.m_b[layer_id] = np.zeros_like(layer.biases)
            self.v_b[layer_id] = np.zeros_like(layer.biases)
            self.t[layer_id] = 0

    def update_weights(self, layer, gradient_weights):
        """
        Updates weights using Adam optimization algorithm.

        Tracks moving averages of gradients and squared gradients to adaptively
        adjust the learning rate for each weight. Applies bias correction to
        these averages before updating the weights for more stable convergence.

        Args:
            layer: The layer whose weights are being updated.
            gradient_weights: The gradient of the loss w.r.t. the weights.
        """
        layer_id = id(layer)
        self._init_moments(layer)
        self.t[layer_id] += 1

        self.m_w[layer_id] = self.beta1 * self.m_w[layer_id] + (1 - self.beta1) * gradient_weights
        self.v_w[layer_id] = self.beta2 * self.v_w[layer_id] + (1 - self.beta2) * (gradient_weights ** 2)

        m_hat = self.m_w[layer_id] / (1 - self.beta1 ** self.t[layer_id])
        v_hat = self.v_w[layer_id] / (1 - self.beta2 ** self.t[layer_id])

        layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def update_biases(self, layer, gradient_biases):
        """
        Updates biases using Adam optimization algorithm.

        Tracks moving averages of gradients and squared gradients to adaptively
        adjust the learning rate for each bias. Applies bias correction to
        these averages before updating the biases for more stable convergence.

        Args:
            layer: The layer whose biases are being updated.
            gradient_biases: The gradient of the loss w.r.t. the biases.
        """
        layer_id = id(layer)
        self._init_moments(layer)

        self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * gradient_biases
        self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * (gradient_biases ** 2)

        m_hat = self.m_b[layer_id] / (1 - self.beta1 ** self.t[layer_id])
        v_hat = self.v_b[layer_id] / (1 - self.beta2 ** self.t[layer_id])

        layer.biases -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
