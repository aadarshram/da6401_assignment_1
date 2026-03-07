"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

# Imports
import numpy as np

class ReLU:
    def forward(self, Z):
        """
        Apply ReLU activation to input.
        Args:
            Z: shape = (batch_size, input_size) - Input to activation
        Returns:
            Z: shape = (batch_size, input_size) = Output after activation
        """
        self.Z = Z # Store input for backprop
        self.output = np.maximum(0, Z) # Store output for backprop
        return self.output # z if z > 0 else 0
    def backward(self, dZ):
        """
        Compute gradient of ReLU activation.
        Args:
            dZ: shape = (batch_size, input_size) - Gradient from output layer
        Returns:
            dZ: shape = (batch_size, input_size) - Gradient before activation
        """
        return dZ * (self.Z > 0) # dg_z/dZ = 1 if z > 0 else 0; Assume derivative at 0 is 0

class Sigmoid:
    def forward(self, Z):
        """
        Apply sigmoid activation to input.
        Args:
            Z: shape = (batch_size, input_size) - Input to activation
        Returns:
            Z: shape = (batch_size, input_size) = Output after activation
        """
        self.Z = Z # Store input for backprop
        self.output = 1 / (1 + np.exp(-Z))
        return self.output
    def backward(self, dZ):
        """
        Compute gradient of sigmoid activation.
        Args:
            dZ: shape = (batch_size, input_size) - Gradient from output layer
        Returns:
            dZ: shape = (batch_size, input_size) - Gradient before activation
        """
        return dZ * (self.output * (1 - self.output)) # dg_z/dZ = (g_z * (1 - g_z))

class Tanh:
    def forward(self, Z):
        """
        Apply tanh activation to input.
        Args:
            Z: shape = (batch_size, input_size) - Input to activation
        Returns:
            Z: shape = (batch_size, input_size) = Output after activation
        """
        self.Z = Z # Store input for backprop
        self.output = np.tanh(Z)
        return self.output
    def backward(self, dZ):
        """
        Compute gradient of tanh activation.
        Args:
            dZ: shape = (batch_size, input_size) - Gradient from output layer
        Returns:
            dZ: shape = (batch_size, input_size) - Gradient before activation
        """
        return dZ * (1 - self.output**2) # dg_z/dZ = (1 - g_z^2)

class Softmax:
    def forward(self, Z):
        """
        Apply softmax activation to input.
        Args:
            Z: shape = (batch_size, input_size) - Input to activation
        Returns:
            Z: shape = (batch_size, input_size) = Output after activation
        """
        self.Z = Z # Store input for backprop
        # exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True)) # for numerical stability (e^100 ~ unstable); Subtracting c from all exponents does not change the output
        # self.output = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        self.output = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True) # shape: (batch_size, input_size)
        return self.output
    
    def backward(self, dZ):
        """
        Compute gradient of softmax activation.
        NOTE: Gradient is complex. Often preferred to club softmax with cross-entropy in computing loss and gradient for numerical stability and simplicity. If doing separately, then the gradient of softmax is computed as follows.
        Args:
            dZ: shape = (batch_size, input_size) - Gradient from output layer
        Returns:
            dZ: shape = (batch_size, input_size) - Gradient before activation
        """
        # Jacobian based calc after vector simplfication: dL/dz = softmax * (dL/dsoftmax - sum(dL/dsoftmax * softmax))
        sum_term = np.sum(dZ * self.output, axis=1, keepdims=True)
        return self.output * (dZ - sum_term)


if __name__ == "__main__":
    pass
    # # Simple tests
    # # Check if fwd and bwd methods run
    # np.random.seed(0)
    # batch_size, input_size = 2, 3
    # Z = np.random.randn(batch_size, input_size)
    # dZ = np.random.randn(batch_size, input_size)
    # # ReLU
    # relu = ReLU()
    # relu_out = relu.forward(Z)
    # relu_dZ = relu.backward(dZ)
    # # Sigmoid
    # sigmoid = Sigmoid()
    # sigmoid_out = sigmoid.forward(Z)
    # sigmoid_dZ = sigmoid.backward(dZ)
    # # Tanh
    # tanh = Tanh()
    # tanh_out = tanh.forward(Z)
    # tanh_dZ = tanh.backward(dZ)
    # # Softmax
    # softmax = Softmax()
    # softmax_out = softmax.forward(Z)
    # softmax_dZ = softmax.backward(dZ)

    # print ("All tests successful")
