import numpy as np

class Layer:
    """
    Base class for neural network layers.

    Provides a template for implementing custom layers, including
    methods for forward pass, backward pass, and parameter updates.
    """

    def __init__(self):
        """
        Initializes the layer.
        """
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass through the layer.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Output of the layer.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Forward method not implemented")

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass through the layer.

        Computes gradients with respect to inputs and stores gradients for parameters (if any).

        Args:
            grad_output (np.ndarray): Gradient from the next layer.

        Returns:
            np.ndarray: Gradient with respect to the input.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Backward method not implemented")

    def update(self, learning_rate: float):
        """
        Updates layer parameters using the stored gradients.

        Args:
            learning_rate (float): Learning rate for the update step.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Update method not implemented")
