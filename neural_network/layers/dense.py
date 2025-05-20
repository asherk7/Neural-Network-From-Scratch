import numpy as np

from neural_network.layers.layer import Layer
from neural_network.optimizer import Optimizer

class Dense(Layer):
    """
    Fully connected (dense) layer.

    This layer performs a linear transformation on the input using 
    weights and biases: output = xW + b. 
    """

    def __init__(self, input_neurons, output_neurons):
        """
        Initializes the dense layer with Xavier/Glorot initialization, by creating a matrix of weights, mapping
        input neurons to output neurons. 
        
        Each row of the weight matrix corresponds to an input neuron, and each column corresponds to an output neuron.

        The weights are initialized using a normal distribution with mean 0 and standard deviation 1/sqrt(input_neurons).
        The biases are initialized to an array of zeros.

        Args:
            input_neurons (int): Number of input features.
            output_neurons (int): Number of output features (neurons).
        """
        self.weights = np.random.randn(input_neurons, output_neurons) * np.sqrt(1 / input_neurons)
        self.biases = np.zeros((1, output_neurons))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass for the dense layer.

        Args:
            x (np.ndarray): Input data of shape (batch_size, input_neurons).

        Returns:
            np.ndarray: Output of the linear transformation (batch_size, output_neurons).
        """
        self.x = x  # Store input for backward pass
        return np.dot(x, self.weights) + self.biases

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes the gradients for backpropagation. 
        The math can be found in the 2nd reference in the README.

        Args:
            gradient (np.ndarray): Gradient from the next layer.

        Returns:
            np.ndarray: Gradient with respect to the input.
        """
        self.gradient_weights = np.dot(self.x.T, gradient)
        self.gradient_biases = np.sum(gradient, axis=0, keepdims=True)
        return np.dot(gradient, self.weights.T)

    def update(self, optimizer: Optimizer):
        """
        Updates weights and biases using the given optimizer.

        Args:
            optimizer (Optimizer): Optimizer to apply parameter updates.
        """
        optimizer.update_weights(self, self.gradient_weights)
        optimizer.update_biases(self, self.gradient_biases)
