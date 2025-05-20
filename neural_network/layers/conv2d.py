import numpy as np

from neural_network.layers.layer import Layer
from neural_network.optimizer import Optimizer

class Conv2D(Layer):
    """
    2D convolutional layer.

    This layer applies a set of learnable filters (also known as kernels) to the input tensor
    in order to extract spatial features. Each filter slides over the input spatially,
    performing an element-wise multiplication and summing the result, effectively scanning
    for specific patterns like edges, textures, or shapes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initializes the Conv2D layer.

        Args:
            in_channels (int): Number of channels in the input volume.
            out_channels (int): Number of filters (output channels).
            kernel_size (int or tuple): Size of each square filter, or (height, width).
            stride (int): Step size with which the filter is applied.
            padding (int): Zero-padding added to all four sides of the input.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization for weights
        scale = np.sqrt(2 / (in_channels * np.prod(self.kernel_size)))
        self.weights = np.random.randn(out_channels, in_channels, *self.kernel_size) * scale
        self.biases = np.zeros(out_channels)

        # Placeholders for gradients
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_biases = np.zeros_like(self.biases)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the convolution operation on the input.

        Slides each filter across the (padded) input tensor, computes
        element-wise multiplications and sums, then adds the bias.

        Args:
            x (np.ndarray): Input tensor of shape (N, C_in, H, W).
            N: Batch size
            C_in: Number of input channels (layer depth)
            H: Input height
            W: Input width

        Returns:
            np.ndarray: Output tensor of shape (N, C_out, H_out, W_out).
        """
        self.input = x
        N, C, H, W = x.shape
        KH, KW = self.kernel_size
        OH = (H + 2 * self.padding - KH) // self.stride + 1
        OW = (W + 2 * self.padding - KW) // self.stride + 1

        # Pad input
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0),
             (self.padding, self.padding),
             (self.padding, self.padding)),
            mode='constant'
        )
        self.x_padded = x_padded

        # Perform convolution
        out = np.zeros((N, self.out_channels, OH, OW))
        for n in range(N):
            for c_out in range(self.out_channels):
                for h in range(OH):
                    for w in range(OW):
                        h0 = h * self.stride
                        w0 = w * self.stride
                        region = x_padded[n, :, h0:h0+KH, w0:w0+KW]
                        out[n, c_out, h, w] = np.sum(region * self.weights[c_out]) + self.biases[c_out]

        return out

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes gradients for backpropagation through this layer.

        Accumulates gradients with respect to weights and biases by sliding
        the upstream gradient over the stored padded input. Computes
        gradient with respect to the input for passing to earlier layers.

        Args:
            gradient (np.ndarray): Upstream gradient of shape (N, C_out, OH, OW).

        Returns:
            np.ndarray: Gradient with respect to the original input
                        (shape (N, C_in, H, W)).
        """
        N, C_out, OH, OW = gradient.shape
        KH, KW = self.kernel_size
        _, C_in, H, W = self.input.shape

        # Reset gradients
        self.gradient_weights.fill(0)
        self.gradient_biases.fill(0)
        dx_padded = np.zeros_like(self.x_padded)

        # Compute gradients
        for n in range(N):
            for c_out in range(C_out):
                for h in range(OH):
                    for w in range(OW):
                        h0 = h * self.stride
                        w0 = w * self.stride
                        grad_val = gradient[n, c_out, h, w]
                        region = self.x_padded[n, :, h0:h0+KH, w0:w0+KW]

                        self.gradient_weights[c_out] += region * grad_val
                        self.gradient_biases[c_out] += grad_val
                        dx_padded[n, :, h0:h0+KH, w0:w0+KW] += self.weights[c_out] * grad_val

        # Remove padding
        if self.padding == 0:
            return dx_padded
        return dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

    def update(self, optimizer: Optimizer):
        """
        Updates the layer's weights and biases via the given optimizer.

        Args:
            optimizer (Optimizer): Optimizer to apply parameter updates.
        """
        optimizer.update_weights(self, self.gradient_weights)
        optimizer.update_biases(self, self.gradient_biases)
