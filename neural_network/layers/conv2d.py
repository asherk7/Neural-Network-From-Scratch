import numpy as np

import neural_network.layers.layer as Layer

class Conv2D(Layer):
    """
    A basic 2D convolutional layer
    Applies a filter (kernel) to the input tensor, to extract features from the input data.

    Forward pass:
        - Pads the input (if needed)
        - Slides the filter(s) across the input with specified stride
        - Computes dot products between filters and local regions of the input

    Backward pass:
        - Computes gradients with respect to weights, biases, and inpute

    Parameters:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of filters (output channels)
        kernel_size (int or tuple): Size of the convolutional kernel (e.g., 3 or (3, 3))
        stride (int): Stride of the convolution (default: 1)
        padding (int): Amount of zero-padding around input (default: 0)

    Attributes:
        weights (ndarray): Learnable filters of shape (out_channels, in_channels, kernel_height, kernel_width)
        biases (ndarray): Learnable biases of shape (out_channels,)
        gradient_weights (ndarray): Gradients of the weights
        gradient_biases (ndarray): Gradients of the biases
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(2 / (in_channels * np.prod(self.kernel_size)))
        self.weights = np.random.randn(out_channels, in_channels, *self.kernel_size) * scale
        self.biases = np.zeros(out_channels)

        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_biases = np.zeros_like(self.biases)

    def forward(self, x):
        self.input = x
        N, C, H, W = x.shape
        KH, KW = self.kernel_size
        OH = (H + 2 * self.padding - KH) // self.stride + 1
        OW = (W + 2 * self.padding - KW) // self.stride + 1

        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        self.x_padded = x_padded
        out = np.zeros((N, self.out_channels, OH, OW))

        for n in range(N):
            for c_out in range(self.out_channels):
                for h in range(OH):
                    for w in range(OW):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + KH
                        w_end = w_start + KW

                        region = x_padded[n, :, h_start:h_end, w_start:w_end]
                        out[n, c_out, h, w] = np.sum(region * self.weights[c_out]) + self.biases[c_out]

        return out

    def backward(self, gradient):
        N, C_out, OH, OW = gradient.shape
        KH, KW = self.kernel_size
        _, C_in, H, W = self.input.shape

        self.gradient_weights.fill(0)
        self.gradient_biases.fill(0)
        dx_padded = np.zeros_like(self.x_padded)

        for n in range(N):
            for c_out in range(self.out_channels):
                for h in range(OH):
                    for w in range(OW):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + KH
                        w_end = w_start + KW

                        region = self.x_padded[n, :, h_start:h_end, w_start:w_end]
                        self.gradient_weights[c_out] += region * gradient[n, c_out, h, w]
                        self.gradient_biases[c_out] += gradient[n, c_out, h, w]
                        dx_padded[n, :, h_start:h_end, w_start:w_end] += self.weights[c_out] * gradient[n, c_out, h, w]

        if self.padding == 0:
            dx = dx_padded
        else:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return dx