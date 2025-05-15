from neural_network.layers.conv2d import Conv2D
from neural_network.layers.dense import Dense
from neural_network.layers.globalavgpool2d import GlobalAvgPool2D
from neural_network.layers.residual import ResidualBlock
from neural_network.activation import ReLU
from neural_network.model import Model
from neural_network.loss import CategoricalCrossEntropy
from neural_network.optimizer import Adam
import numpy as np
import pickle

def create_model():
    relu = ReLU()
    downsample = Conv2D(in_channels=16, out_channels=32, kernel_size=1, stride=2, padding=0) 

    layers = [
        Conv2D(3, 16, kernel_size=3, stride=1, padding=1),
        relu,
        ResidualBlock(16, 16, stride=1, activation=relu),
        ResidualBlock(16, 32, stride=2, activation=relu, downsample=downsample),
        GlobalAvgPool2D(),
        Dense(32, 10), 
    ]

    model = Model(layers=layers, loss=CategoricalCrossEntropy(), optimizer=Adam(lr=0.001))
    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
    """
    Train the model on the CIFAR-10 dataset.
    Args:
        model (Model): The model to train.
        x_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        x_val (np.ndarray): Validation data.
        y_val (np.ndarray): Validation labels.
        epochs (int): Number of epochs to train.
        batch_size (int): Size of each batch.
    """
    num_samples = x_train.shape[0]
    for epoch in range(epochs):
        for i in range(0, num_samples, batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            loss = model.train_batch(x_batch, y_batch)
            print(f"Epoch {epoch + 1}/{epochs}, Batch {i // batch_size + 1}, Loss: {loss:.4f}")

        val_loss = model.evaluate(x_val, y_val)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")

def load_batch(filename):
    """
    Load a single batch of CIFAR-10 data.
    Args:
        filename (str): Path to the batch file.
    Returns:
        tuple: Tuple containing the data and labels.
    """
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')  
    data = batch[b'data']      
    labels = batch[b'labels']  
    return data, labels

def load_cifar10(data_dir):
    data_list = []
    labels_list = []
    for i in range(1, 6):
        data_batch, labels_batch = load_batch(f'{data_dir}/data_batch_{i}')
        data_list.append(data_batch)
        labels_list.append(labels_batch)
    
    X_train = np.concatenate(data_list)
    y_train = np.concatenate(labels_list)
    
    X_test, y_test = load_batch(f'{data_dir}/test_batch')
    
    return X_train, y_train, X_test, y_test

def preprocess(X):
    X = X.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return X

def one_hot_encode(labels, num_classes=10):
    encoded = np.zeros((len(labels), num_classes))
    encoded[np.arange(len(labels)), labels] = 1
    return encoded

def main():
    model = create_model()
    data_dir = './data'

    X_train, y_train, X_test, y_test = load_cifar10(data_dir)

    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    y_train_oh = one_hot_encode(y_train)
    y_test_oh = one_hot_encode(y_test)
    
    #train_model(model, X_train, y_train_oh, X_test, y_test_oh, epochs=10, batch_size=32)
    
    X_test_sample = X_test[:8]  # 8 images
    y_test_sample = y_test_oh[:8]  # corresponding one-hot labels

    # Forward pass
    logits = model.forward(X_test_sample)

    # Calculate loss
    test_loss = model.loss.forward(logits, y_test_sample)
    print(f'Test Loss: {test_loss}')

    # Calculate accuracy (simple)
    pred_classes = np.argmax(logits, axis=1)
    true_classes = np.argmax(y_test_sample, axis=1)
    accuracy = np.mean(pred_classes == true_classes)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')


if __name__ == "__main__":
    main()