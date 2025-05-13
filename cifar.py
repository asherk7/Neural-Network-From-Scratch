def unpickle_cifar10(file):
    """
    Unpickle CIFAR-10 dataset.
    Args:
        file (str): Path to the CIFAR-10 file.
    Returns:
        tuple: Tuple containing the data and labels.
    """
    import pickle
    import numpy as np

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = dict[b'data']
    y = dict[b'labels']
    X = np.array(X)
    y = np.array(y)
    return X, y

