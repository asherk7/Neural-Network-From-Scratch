import numpy as np

class Loss:
    def __init__(self):
        pass

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Forward pass through the loss function.
        Args:
            y_true (np.ndarray): True labels, often one-hot encoded for classification tasks.
            y_pred (np.ndarray): Predicted labels.
        Returns:
            float: Loss value.
        """
        raise NotImplementedError("Forward method not implemented")
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Backward pass through the loss function.
        Args:
            y_true (np.ndarray): True labels, often one-hot encoded for classification tasks.
            y_pred (np.ndarray): Predicted labels. (Note: this is a variable, not a constant when doing the derivative for backpropagation)
        Returns:
            np.ndarray: Gradient of the loss with respect to the predictions.
        """
        raise NotImplementedError("Backward method not implemented")

class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) loss function.
    This is used for regression problems, where the goal is to minimize the difference between predicted and true values.
    """
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        diff = y_true - y_pred
        diff_squared = np.square(diff)
        mean_squared_error = np.mean(diff_squared)
        return mean_squared_error
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        diff = y_pred - y_true
        # Gradient of MSE is 2 * (y_pred - y_true) / N, where N is the number of samples
        return (2 * diff) / y_true.size

class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy loss function.
    This is used for binary classification problems (e.g., 0 or 1).

    BCE = -1/N * sum([y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)])
    """
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-15  # Small value to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15  # Small value to prevent division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid division by zero
        return -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
    
class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy loss function.
    This is used for multi-class classification problems (e.g., 0, 1, 2, ..., n).

    CCE = -1/N * sum([y_true * log(y_pred)])
    """
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-15  # Small value to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15  # Small value to prevent division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid division by zero
        return -(y_true / y_pred) / y_true.shape[0]

