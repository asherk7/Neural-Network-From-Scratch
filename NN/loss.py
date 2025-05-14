import numpy as np

class Loss():
    def __init__(self):
        pass

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Forward pass through the loss function.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        Returns:
            float: Loss value.
        """
        raise NotImplementedError("Forward method not implemented")
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Backward pass through the loss function.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        Returns:
            np.ndarray: Gradient of the loss with respect to the predictions.
        """
        raise NotImplementedError("Backward method not implemented")

class MeanSquaredError(Loss):
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the Mean Squared Error (MSE) loss.
        """
        return np.mean(np.square(y_true - y_pred))
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the MSE loss with respect to the predictions.
        """
        return 2 * (y_pred - y_true) / y_true.size

class BinaryCrossEntropy(Loss):
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the Binary Cross-Entropy loss.
        """
        epsilon = 1e-15  # Small value to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Binary Cross-Entropy loss with respect to the predictions.
        """
        epsilon = 1e-15  # Small value to prevent division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid division by zero
        return -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
    
class CategoricalCrossEntropy(Loss):
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the Categorical Cross-Entropy loss.
        """
        epsilon = 1e-15  # Small value to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Categorical Cross-Entropy loss with respect to the predictions.
        """
        epsilon = 1e-15  # Small value to prevent division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid division by zero
        return -(y_true / y_pred) / y_true.shape[0]

