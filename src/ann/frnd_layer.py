"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

from .activations import relu, sigmoid, tanh, softmax, relu_derivative, sigmoid_derivative, tanh_derivative
import numpy as np

class NeuralLayer:
    """
    Represents a single layer in the neural network.
    Initializes weights and biases, performs forward pass, and computes gradients during backward pass.
    """

    def __init__(self, input_size, output_size, activation, weight_init):
        """
        Initialize weights, biases, and activation type. 
        Create placeholders for forward and backward pass variables.
        """
        if weight_init == "xavier":
            self.W = np.random.randn(input_size, output_size) * np.sqrt(1/input_size)
        elif weight_init == "random":
            self.W = np.random.randn(input_size, output_size) * 0.01
        elif weight_init == "zeros":
            self.W = np.zeros((input_size, output_size))
        
        self.b = np.zeros((1, output_size))
        
        self.activation_name = activation
        
        self.Z = None
        self.A = None
        self.input = None
        self.grad_b = None
        self.grad_W = None

    def forward(self, X):
        """
        Forward pass through the layer
        """
        self.input = X
        self.Z = X @ self.W + self.b
        self.A = self.activate(self.Z)
        return self.A

    def backward(self, dA):
        """
        Backward pass through the layer. Access to next layer's dA to compute gradients. 
        Return dX to pass to previous layer. Store grad_W and grad_b for optimizer step.
        """
        dZ = dA * self.activation_derivative(self.Z)
        self.grad_W = self.input.T @ dZ
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ self.W.T
        
        return dX, self.grad_W, self.grad_b

    def activate(self, Z):
        """
        Apply the specified activation function to Z.
        """
        if self.activation_name == "linear":
            return Z
        elif self.activation_name == "relu":
            return relu(Z)
        elif self.activation_name == "sigmoid":
            return sigmoid(Z)
        elif self.activation_name == "tanh":
            return tanh(Z)
        elif self.activation_name == "softmax":
            return softmax(Z)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")
    
    def activation_derivative(self, Z):
        """
        Compute the derivative of the activation function with respect to Z.
        Softmax is typically handled separately in the loss function, so we won't compute its derivative here.
        """
        if self.activation_name == "linear":
            return np.ones_like(Z)
        elif self.activation_name == "relu":
            return relu_derivative(Z)
        elif self.activation_name == "sigmoid":
            return sigmoid_derivative(Z)
        elif self.activation_name == "tanh":
            return tanh_derivative(Z)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")