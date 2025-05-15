from neural_network.layers.layer import Layer
from neural_network.layers.conv2d import Conv2D

class ResidualBlock(Layer):
    def __init__(self, in_channels, out_channels, stride=1, activation=None, downsample=None):
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = activation
        self.downsample = downsample

    def forward(self, x):
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
        gradient = self.activation.backward(gradient)
        gradient_skip = gradient.copy()

        gradient = self.conv2.backward(gradient)
        gradient = self.activation.backward(gradient)
        gradient = self.conv1.backward(gradient)

        if self.downsample:
            gradient_skip = self.downsample.backward(gradient_skip)

        return gradient + gradient_skip
