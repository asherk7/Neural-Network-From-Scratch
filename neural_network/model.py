class Model:
    def __init__(self, layers, loss, optimizer):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def update(self):
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                self.optimizer.update_weights(layer, layer.dweights)
            if hasattr(layer, 'biases'):
                self.optimizer.update_biases(layer, layer.dbiases)

    def train_batch(self, x_batch, y_batch):
        # Forward pass
        y_pred = self.forward(x_batch)
        loss_val = self.loss.forward(y_pred, y_batch)

        # Backward propagation
        grad_loss = self.loss.backward(y_pred, y_batch)
        self.backward(grad_loss)

        self.update()
        return loss_val

    def evaluate(self, x_val, y_val):
        y_pred = self.forward(x_val)
        loss_val = self.loss.forward(y_pred, y_val)
        return loss_val
