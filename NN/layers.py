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

class Dense(Layer):
    def __init__(self, input_neurons, output_neurons):
        """
        Initialize the layer by creating a matrix of weights, mapping input neurons to output neurons.
        The weights are initialized using a normal distribution with mean 0 and standard deviation 1/sqrt(input_neurons).
        The biases are initialized to an array of zeros.
        """
        self.weights = np.random.randn(input_neurons, output_neurons) * np.sqrt(1 / input_neurons)
        self.biases = np.zeros((1, output_neurons))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x # Store the input for backward pass
        return np.dot(x, self.weights) + self.biases
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        gradient_weights: Gradient of the loss with respect to the weights (how much we need to change the weights).
        gradient_biases: Gradient of the loss with respect to the biases (how much we need to change the biases).
        gradient_layer: Gradient of the loss with respect to the input of this layer (the gradient that will be passed to the previous layer).
        The math can be found in the Reference in the README.
        """
        self.gradient_weights = np.dot(self.x.T, gradient)
        self.gradient_biases = np.sum(gradient, axis=0, keepdims=True)
        gradient_layer = np.dot(gradient, self.weights.T)
        return gradient_layer
    
    def update(self, learning_rate: float):
        self.weights -= learning_rate * self.gradient_weights
        self.biases -= learning_rate * self.gradient_biases

class Conv2D(Layer):
    pass

class MaxPool2D(Layer):
    pass

class Flatten(Layer):
    pass
