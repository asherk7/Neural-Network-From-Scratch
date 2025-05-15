import numpy as np

from neural_network.layers.layer import Layer

class GlobalAvgPool2D(Layer):
    """
    global average pooling layer for 2D convolutional outputs.

    Reduces each channel of the input feature map to a single value by averaging over
    all spatial locations (height and width). 

    Forward pass:
        - Averages each channel across spatial dimensions (H, W)

    Backward pass:
        - Distributes the upstream gradient equally across all spatial locations per channel

    Input shape: (batch_size, channels, height, width)
    Output shape: (batch_size, channels)
    """
    def forward(self, x):
        self.input_shape = x.shape
        return np.mean(x, axis=(2, 3), keepdims=False)

    def backward(self, gradient):
        N, C = gradient.shape
        _, _, H, W = self.input_shape

        dx = gradient[:, :, None, None] * np.ones((N, C, H, W)) / (H * W)
        return dx