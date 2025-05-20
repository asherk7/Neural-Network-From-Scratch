from neural_network.layers.layer import Layer
from neural_network.layers.conv2d import Conv2D

class ResidualBlock(Layer):
    """
    Residual block consisting of two Conv2D layers with an optional downsampling layer.

    Implements a basic residual connection: output of the block is the sum of the
    transformed input and the original input (possibly downsampled). Applies
    an activation function after each convolution and after the residual sum.
    """

    def __init__(self, in_channels, out_channels, stride=1, activation=None, downsample=None):
        """
        Initializes the residual block with two convolutional layers and an optional downsample layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first convolution layer. Defaults to 1.
            activation (Activation): Activation function to use after each conv and sum.
            downsample (Layer or None): Optional downsampling layer to match dimensions
                                        of identity to output when channels or stride differ.
        """

        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = activation
        self.downsample = downsample

    def forward(self, x):
        """
        Forward pass of the residual block.

        Applies first convolution, activation, second convolution, adds skip connection
        (with optional downsampling), then applies activation again.

        Args:
            x (np.ndarray): Input tensor of shape (N, C_in, H, W).

        Returns:
            np.ndarray: Output tensor of shape (N, C_out, H_out, W_out).
        """

        identity = x
        out = self.conv1.forward(x)
        out = self.activation.forward(out)
        out = self.conv2.forward(out)

        if self.downsample:
            identity = self.downsample.forward(x)

        out += identity
        out = self.activation.forward(out)
        return out

    def backward(self, gradient):
        """
        Backward pass of the residual block.

        Computes gradients flowing back through the activation, convolutions,
        and skip connection (with optional downsampling), then sums gradients.

        Args:
            gradient (np.ndarray): Upstream gradient of shape matching block output.

        Returns:
            np.ndarray: Gradient with respect to the input tensor x.
        """
        
        gradient = self.activation.backward(gradient)
        gradient_skip = gradient.copy()

        gradient = self.conv2.backward(gradient)
        gradient = self.activation.backward(gradient)
        gradient = self.conv1.backward(gradient)

        if self.downsample:
            gradient_skip = self.downsample.backward(gradient_skip)

        return gradient + gradient_skip
