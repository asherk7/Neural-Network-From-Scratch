import numpy as np

from neural_network.optimizer import Optimizer

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
    
    def update(self, optimizer: Optimizer):
        optimizer.update_weights(self, self.gradient_weights)
        optimizer.update_biases(self, self.gradient_biases)

class Conv2D:
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

class GlobalAvgPool2D:
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
