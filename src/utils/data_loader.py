"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

# Imports
import numpy as np
import keras.datasets # For dataloading
import sklearn # For train-val split and confusion matrix

def load_dataset(args):
    """
    Load and preprocess dataset.
    Args:
        dataset_name: 'mnist' or 'fashion_mnist'
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """

    # Load dataset
    data = keras.datasets.mnist.load_data() if args.dataset == "mnist" else keras.datasets.fashion_mnist.load_data() if args.dataset == "fashion_mnist" else NotImplementedError("Supports only 'mnist' or 'fashion_mnist' datasets. Implement custom dataset loading if using other datasets.")
    X_train, y_train = data[0]
    X_test, y_test = data[1]
    # Train-val split
    val_size = args.val_size if "val_size" in vars(args) else 0.1 # If val_size not specified, either use default or its inference mode. In latter case, irrelevant -> ignore
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size=val_size, random_state=42, shuffle=True)
    # Normalization
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    # Flatten to input for FCC
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    # One-hot encode output labels
    num_classes = args.output_size
    y_train = np.eye(num_classes)[y_train]
    y_val = np.eye(num_classes)[y_val]
    y_test = np.eye(num_classes)[y_test]
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    # Simple tests
    class Args:
        dataset = "mnist"
        val_size = 0.1
        output_size = 10
    args = Args()
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args)
    print("All tests successful")