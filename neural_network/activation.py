import numpy as np

class Activation:
    """
    Base class for activation functions.

    Defines the interface for forward and backward passes of activations.
    """

    def __init__(self):
        """
        Initializes the activation function.
        """
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the activation.

        Args:
            x (np.ndarray): Input array to the activation function.

        Returns:
            np.ndarray: Output after applying the activation.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Forward method not implemented")

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass (gradient) of the activation.
        This is done by taking the derivative of the activation function with respect to its input
        and multiplying it by the gradient of the loss with respect to the output.

        Args:
            gradient (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Backward method not implemented")


class Linear(Activation):
    """
    Linear activation function (identity).

    Returns input as output without any modification, f(x) = x.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass returns input unchanged.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Same as input.
        """
        return x

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass returns the gradient unchanged.

        Args:
            gradient (np.ndarray): Gradient of the loss w.r.t output.

        Returns:
            np.ndarray: Same as input gradient.
        """
        return gradient


class Sigmoid(Activation):
    """
    Sigmoid activation function.

    Maps input values to the range (0, 1), useful for binary classification.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes sigmoid activation.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying sigmoid.
        """
        self.x = x  # Store input for backward pass
        return 1 / (1 + np.exp(-x))

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes gradient of sigmoid activation using stored input.

        Args:
            gradient (np.ndarray): Gradient of the loss w.r.t output.

        Returns:
            np.ndarray: Gradient of the loss w.r.t input.
        """
        #sigmoid = self.forward(gradient)
        #return gradient * sigmoid * (1 - sigmoid)
        sigmoid_derivative = np.exp(self.x) / ((1 + np.exp(self.x))**2)
        return gradient * sigmoid_derivative


class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU) activation function.

    Outputs input directly if positive; otherwise outputs zero.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies ReLU activation element-wise.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying ReLU.
        """
        self.x = x  # Store input for backward pass
        relu = np.maximum(0, x)
        return relu

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes gradient of ReLU.

        Args:
            gradient (np.ndarray): Gradient of the loss w.r.t output.

        Returns:
            np.ndarray: Gradient of the loss w.r.t input, zero where input <= 0.
        """
        relu_grad = self.x.copy()
        relu_grad[relu_grad > 0] = 1
        relu_grad[relu_grad <= 0] = 0
        return gradient * relu_grad


class Tanh(Activation):
    """
    Hyperbolic tangent (tanh) activation function.

    Maps input values to the range (-1, 1), centered around zero.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes tanh activation.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying tanh.
        """
        self.x = x  # Store input for backward pass
        return np.tanh(x)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes gradient of tanh activation.

        Args:
            gradient (np.ndarray): Gradient of the loss w.r.t output.

        Returns:
            np.ndarray: Gradient of the loss w.r.t input.
        """
        tanh_derivative = 1 - np.tanh(self.x)**2
        return gradient * tanh_derivative


class LeakyReLU(Activation):
    """
    Leaky ReLU activation function.

    Similar to ReLU but allows a small gradient (0.01x) for negative inputs to avoid dying neurons.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies Leaky ReLU activation element-wise.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying Leaky ReLU.
        """
        self.x = x  # Store input for backward pass
        return np.where(x > 0, x, 0.01 * x) 

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes gradient of Leaky ReLU.

        Args:
            gradient (np.ndarray): Gradient of the loss w.r.t output.

        Returns:
            np.ndarray: Gradient of the loss w.r.t input.
        """
        leaky_relu_grad = np.where(self.x > 0, 1, 0.01)
        return gradient * leaky_relu_grad


class Softmax(Activation):
    """
    Softmax activation function.

    Converts raw scores into probabilities summing to 1 across classes, used mainly for classification.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes softmax probabilities with numerical stability adjustment.

        Args:
            x (np.ndarray): Input array (batch_size x num_classes).

        Returns:
            np.ndarray: Probability distribution across classes.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # Subtract by max for numerical stability (avoid overflow on exponential)
        probabilities = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.output = probabilities  # Store for backward pass
        return probabilities

    def backward(self, y_true: np.ndarray) -> np.ndarray:
        """
        Computes gradient of softmax combined with cross-entropy loss.

        Args:
            y_true (np.ndarray): One-hot encoded true labels.

        Returns:
            np.ndarray: Gradient of the loss w.r.t input logits.
        """

        """
        # Gradient of the softmax function using the Jacobian matrix
        # The Jacobian matrix is a square matrix of partial derivatives
        self.dinputs = np.empty_like(gradient)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, gradient)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)
        """

        return self.output - y_true
