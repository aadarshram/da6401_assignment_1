"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

# Imports
from utils.misc import initialize_weights
import numpy as np

class NeuralLayer:

    def __init__(self, input_size, output_size, weight_init):
        self.input_size= input_size
        self.output_size = output_size
        self.W = None
        self.b = None
        self.X = None # Stores input for backprop
        # Initialize weights and bias
        self.W, self.b = initialize_weights(input_size, output_size, method=weight_init)
        
    def forward(self, X):
        """
        Forward pass through the Layer
        Inputs:
            X : shape = (batch_size, input_size) - Input to the layer
        Returns:
            Z : shape = (batch_size, output_size) - Output of the layer
        """
        self.X = X # Store input for backprop
        Z = np.matmul(X, self.W) + self.b
        return Z
    
    def backward(self, dZ):
        """
        Backward pass through the Layer
        Inputs:
            dZ: shape = (batch_size, output_size) - Gradient of loss wrt layer output
        Returns:
            dX: shape = (batch_size, input_size) - Gradient of loss wrt layer input
        """
        self.grad_W = np.matmul(self.X.T, dZ) # shape: (input_size, batch_size) x (batch_size, output_size) = (input_size, output_size)
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) # shape: (1, output_size)
        # NOTE: grad_W, grad_b normalized by batch_size in the loss function.
        dX = np.matmul(dZ, self.W.T) # shape: (batch_size, ouptut_size) x (output_size, input_size) = (batch_size, input_size)
        return dX

if __name__ == "__main__":
    # Simple tests
    # Check if initialize, fwd and bwd methods run
    np.random.seed(42)
    layer = NeuralLayer(input_size=4, output_size=3, weight_init='xavier')
    X = np.random.randn(5, 4) # batch_size=5, input_size=4
    Z = layer.forward(X)
    dZ = np.random.randn(5, 3) # batch_size=5, output_size=3
    dX = layer.backward(dZ)
    print("All tests successful")