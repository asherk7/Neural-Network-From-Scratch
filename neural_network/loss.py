import numpy as np

class Loss:
    """
    Base class for loss functions.

    Defines the interface for forward and backward computation of losses.
    """

    def __init__(self):
        """
        Initializes the loss function.
        """
        pass

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the loss value.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: Computed loss.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Forward method not implemented")

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted values.

        Returns:
            np.ndarray: Gradient of the loss with respect to predictions.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Backward method not implemented")


class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) loss.

    Used for regression tasks. Measures average squared difference between predictions and actual values.

    MSE = 1/N * sum([(y_true - y_pred)^2])
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the mean squared error loss.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: Mean squared error.
        """
        diff = y_true - y_pred
        diff_squared = np.square(diff)
        mean_squared_error = np.mean(diff_squared)
        return mean_squared_error

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the MSE loss.
        Gradient of MSE is 2 * (y_pred - y_true) / N, where N is the number of samples

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            np.ndarray: Gradient of the loss with respect to predictions.
        """
        diff = y_pred - y_true
        return (2 * diff) / y_true.size


class BinaryCrossEntropy(Loss):
    """
    Binary Cross Entropy loss.

    Used for binary classification tasks.

    BCE = -1/N * sum([y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)])
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the binary cross entropy loss.

        Args:
            y_true (np.ndarray): Ground truth binary labels.
            y_pred (np.ndarray): Predicted probabilities.

        Returns:
            float: Binary cross entropy loss.
        """
        epsilon = 1e-15 # to avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # to avoid log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the binary cross entropy loss.

        Args:
            y_true (np.ndarray): Ground truth binary labels.
            y_pred (np.ndarray): Predicted probabilities.

        Returns:
            np.ndarray: Gradient of the loss with respect to predictions.
        """
        epsilon = 1e-15 # to avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # to avoid log(0)
        return -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross Entropy loss.

    Used for multi-class classification with one-hot encoded labels.

    CCE = -1/N * sum([y_true * log(y_pred)])
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the categorical cross entropy loss.

        Args:
            y_true (np.ndarray): One-hot encoded ground truth labels.
            y_pred (np.ndarray): Predicted probabilities.

        Returns:
            float: Categorical cross entropy loss.
        """
        epsilon = 1e-15 # to avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # to avoid log(0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the categorical cross entropy loss.

        Args:
            y_true (np.ndarray): One-hot encoded ground truth labels.
            y_pred (np.ndarray): Predicted probabilities.

        Returns:
            np.ndarray: Gradient of the loss with respect to predictions.
        """
        epsilon = 1e-15 # to avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # to avoid log(0)
        return -(y_true / y_pred) / y_true.shape[0]

