from sklearn.model_selection import train_test_split
import numpy as np
import pickle

from neural_network.layers.conv2d import Conv2D
from neural_network.layers.dense import Dense
from neural_network.layers.globalavgpool2d import GlobalAvgPool2D
from neural_network.layers.residual import ResidualBlock
from neural_network.activation import ReLU, Softmax
from neural_network.model import Model
from neural_network.loss import CategoricalCrossEntropy
from neural_network.optimizer import Adam

def create_model():
    """
    Constructs and returns a convolutional neural network with residual connections.

    Returns:
        Model: An instance of the custom neural network model with predefined architecture.
    """
    downsample = Conv2D(in_channels=16, out_channels=32, kernel_size=1, stride=2, padding=0) 

    layers = [
        Conv2D(3, 16, kernel_size=3, stride=1, padding=1),
        ReLU(),
        ResidualBlock(16, 16, stride=1, activation=ReLU()),
        ResidualBlock(16, 16, stride=1, activation=ReLU()),
        ResidualBlock(16, 16, stride=1, activation=ReLU()),
        ResidualBlock(16, 16, stride=1, activation=ReLU()),
        ResidualBlock(16, 32, stride=2, activation=ReLU(), downsample=downsample),
        GlobalAvgPool2D(),
        Dense(32, 10), 
        Softmax()
    ]

    model = Model(layers=layers, loss=CategoricalCrossEntropy(), optimizer=Adam())
    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
    """
    Trains the model using mini-batch optimization with the Adam optimizer.

    Args:
        model (Model): The neural network model to train.
        x_train (np.ndarray): Training data.
        y_train (np.ndarray): One-hot encoded training labels.
        x_val (np.ndarray): Validation data.
        y_val (np.ndarray): One-hot encoded validation labels.
        epochs (int): Number of epochs to train.
        batch_size (int): Size of each mini-batch.

    Returns:
        Model: The trained model.
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

    return model

def load_batch(filename):
    """
    Loads a single batch from the CIFAR-10 dataset.

    Args:
        filename (str): Path to the batch file.

    Returns:
        tuple: Tuple of (data, labels) where data is np.ndarray and labels is list.
    """
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')  
    data = batch[b'data']      
    labels = batch[b'labels']  
    return data, labels

def load_cifar10(data_dir):
    """
    Loads all CIFAR-10 training and test data.

    Args:
        data_dir (str): Directory where CIFAR-10 data batches are stored.

    Returns:
        tuple: Tuple of (X_train, y_train, X_test, y_test) as NumPy arrays.
    """
    data_list = []
    labels_list = []
    for i in range(1, 6):
        data_batch, labels_batch = load_batch(f'{data_dir}/data_batch_{i}')
        data_list.append(data_batch)
        labels_list.append(labels_batch)
    
    X = np.concatenate(data_list)
    y = np.concatenate(labels_list)
    
    X_test, y_test = load_batch(f'{data_dir}/test_batch')
    
    return X, y, X_test, y_test

def preprocess(X):
    """
    Reshapes and normalizes CIFAR-10 image data.
    The input data is reshaped from (N, 3072), where each image is a 32x32x3 array flattened into a vector of size 3072.
    The array of images is reshaped to (N, 3, 32, 32) for processing by the neural network.
    The pixel values are also normalized to the range [0, 1] by dividing by 255.0.

    Args:
        X (np.ndarray): Input image data of shape (N, 3072).

    Returns:
        np.ndarray: Reshaped and normalized data of shape (N, 3, 32, 32).
    """
    X = X.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return X

def one_hot_encode(labels, num_classes=10):
    """
    Converts integer labels into one-hot encoded format.

    Args:
        labels (list or np.ndarray): Class labels.
        num_classes (int): Number of unique classes.

    Returns:
        np.ndarray: One-hot encoded label matrix.
    """
    encoded = np.zeros((len(labels), num_classes))
    encoded[np.arange(len(labels)), labels] = 1
    return encoded

def predict(model, x):
    """
    Generates predicted class labels from the model output.

    Args:
        model (Model): Trained neural network model.
        x (np.ndarray): Input data.

    Returns:
        np.ndarray: Predicted class indices.
    """
    return np.argmax(model.forward(x), axis=1)

def main():
    """
    Main execution routine: loads data, trains the model, evaluates accuracy on the test set.
    """
    model = create_model()
    data_dir = './data'

    X, y, X_test, y_test = load_cifar10(data_dir)

    X = preprocess(X)
    X_test = preprocess(X_test)

    y_ohe = one_hot_encode(y)

    X_train, X_val, y_train_ohe, y_val_ohe = train_test_split(X, y_ohe, test_size=0.2, random_state=42)
    
    model = train_model(model, X_train, y_train_ohe, X_val, y_val_ohe, epochs=10, batch_size=32)

    y_pred = predict(model, X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()