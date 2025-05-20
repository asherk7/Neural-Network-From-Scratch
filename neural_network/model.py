class Model:
    """
    A neural network model composed of multiple layers, trained with a specified loss function and optimizer.

    Args:
        layers (list): List of layer instances forming the model.
        loss (Loss): Loss function instance used to compute the loss and its gradient.
        optimizer (Optimizer): Optimizer instance used to update model parameters.
    """

    def __init__(self, layers, loss, optimizer):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, x):
        """
        Perform a forward pass through all layers of the model.

        Args:
            x (np.ndarray): Input data batch.

        Returns:
            np.ndarray: Output predictions from the final layer.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, gradient):
        """
        Perform a backward pass through all layers to propagate gradients.

        Args:
            gradient (np.ndarray): Gradient of the loss with respect to model output.

        Returns:
            np.ndarray: Gradient with respect to the input data (useful for debugging or advanced usage).
        """
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def update(self):
        """
        Update weights and biases of all layers that have parameters using the optimizer.
        """
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                layer.update(self.optimizer)

    def train_batch(self, x_batch, y_batch):
        """
        Train the model on a single batch of data.

        Performs forward pass, computes loss, backpropagates gradients, and updates parameters.

        Args:
            x_batch (np.ndarray): Batch of input data.
            y_batch (np.ndarray): Batch of true labels.

        Returns:
            float: Computed loss value for the batch.
        """
        y_pred = self.forward(x_batch)
        loss_val = self.loss.forward(y_pred, y_batch)

        grad_loss = self.loss.backward(y_pred, y_batch)
        self.backward(grad_loss)

        self.update()
        return loss_val

    def evaluate(self, x_val, y_val):
        """
        Evaluate the model on validation data without updating parameters.

        Args:
            x_val (np.ndarray): Validation input data.
            y_val (np.ndarray): Validation true labels.

        Returns:
            float: Computed loss value on validation data.
        """
        y_pred = self.forward(x_val)
        loss_val = self.loss.forward(y_pred, y_val)
        return loss_val
