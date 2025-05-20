import numpy as np

from neural_network.layers.layer import Layer

class GlobalAvgPool2D(Layer):
    """
    Global Average Pooling 2D layer.

    This layer performs spatial average pooling over each channel of the input,
    reducing the spatial dimensions (H, W) to a single value per channel.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass for global average pooling.

        Args:
            x (np.ndarray): Input tensor of shape (N, C, H, W) where
                            N = batch size, C = channels, H = height, W = width.

        Returns:
            np.ndarray: Output tensor of shape (N, C), where each channel is 
                        averaged over the spatial dimensions.
        """
        self.input_shape = x.shape
        return np.mean(x, axis=(2, 3), keepdims=False)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to the input.

        Args:
            gradient (np.ndarray): Upstream gradient of shape (N, C).

        Returns:
            np.ndarray: Gradient with respect to the input, shape (N, C, H, W).
        """
        N, C = gradient.shape
        _, _, H, W = self.input_shape
        # Distribute the gradient equally across spatial dimensions
        dx = gradient[:, :, None, None] * np.ones((N, C, H, W)) / (H * W) 
        return dx
