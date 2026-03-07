"""
Misc. Utility functions.
"""

# Imports
import numpy as np

# Weight initialization

def initialize_weights(input_size, output_size, method):
    """
    Initialize weights and biases for a layer.
    
    Args:
        input_size: Number of input neurons
        output_size: Number of output neurons
        method: Weight initialization method ('random' or 'xavier')
    Returns:
        W: Initialized weights
        b: Initialized biases
    """
    # # Experiment: Zero initialization
    # W = np.zeros((input_size, output_size))
    # b = np.zeros((1, output_size))
    # return W, b

    if method == "random":
        W = np.random.randn(input_size, output_size) * 0.01 # Populate from standard normal * 0.01 (scale down variance)
        b = np.zeros((1, output_size)) # Weight initialization required for symmetry breaking. Bias unnecessarily adds noise. Simply set to zero.
    elif method == "xavier":
        W = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size)) # variance preservation 
        b = np.zeros((1, output_size))
    else:
        raise NotImplementedError("Supports only 'random' or 'xavier' weight initialization")
    return W, b

if __name__ == "__main__":
    # Simple tests
    np.random.seed(42)
    W_random, b_random = initialize_weights(4, 3, method='random')
    W_xavier, b_xavier = initialize_weights(4, 3, method='xavier')
    print("All tests successful")