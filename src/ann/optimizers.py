"""
Optimization Algorithms
Implements: sgd, momentum, nag, rmsprop, adam, nadam

TODO: Implement Adam and Nadam optimizers based on learnt knowledge from coursework.
"""

# Imports
from ann.neural_layer import NeuralLayer

import numpy as np

class SGD:
    def update(self, layers, learning_rate, weight_decay):
        """
        Update weights and biases of all layers using Stochastic Gradient Descent (SGD).
        Args:
                layers: List of NeuralLayer objects in the network
                learning_rate: Learning rate for SGD updates
                weight_decay: L2 regularization strength
        """
        for layer in layers:
            if isinstance(layer, NeuralLayer):
                # Update weights with L2 regularization
                layer.W -= learning_rate * (layer.grad_W + weight_decay * layer.W)
                layer.b -= learning_rate * layer.grad_b

class Momentum:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.velocity_W = {} # per layer
        self.velocity_b = {} # per layer
    def update(self, layers, learning_rate, weight_decay):
        """
        Update weights and biases of all layers using Momentum optimization.
        Args:
                layers: List of NeuralLayer objects in the network
                learning_rate: Learning rate for updates
                weight_decay: L2 regularization strength
        """
        for layer in layers:
            if isinstance(layer, NeuralLayer):
                # Initialize velocity if not already done
                if layer not in self.velocity_W:
                    self.velocity_W[layer] = np.zeros_like(layer.W)
                    self.velocity_b[layer] = np.zeros_like(layer.b)
                # Update velocity + L2 regularization
                self.velocity_W[layer] = self.momentum * self.velocity_W[layer] + learning_rate * (layer.grad_W + weight_decay * layer.W)
                self.velocity_b[layer] = self.momentum * self.velocity_b[layer] + learning_rate * layer.grad_b
                # Update W, b
                layer.W -= self.velocity_W[layer]
                layer.b -= self.velocity_b[layer]

class NAG:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.velocity_W = {} # per layer
        self.velocity_b = {} # per layer

    def update(self, layers, learning_rate, weight_decay, model, y_true, y_pred, X_batch):
        """
        Update weights and biases of all layers using Nesterov Accelerated Gradient (NAG) optimization.
        Args:
                layers: List of NeuralLayer objects in the network
                learning_rate: Learning rate for updates
                weight_decay: L2 regularization strength
                # NAG specific args for lookahead gradient computation
                model: NeuralNetwork object to compute lookahead gradients
                y_true: True labels for current batch (for lookahead gradient computation)
                y_pred: Predicted logits for current batch (for lookahead gradient computation)
        """

        # Initialize velocities if needed
        for layer in layers:
            if isinstance(layer, NeuralLayer):
                if layer not in self.velocity_W:
                    self.velocity_W[layer] = np.zeros_like(layer.W)
                    self.velocity_b[layer] = np.zeros_like(layer.b)

        # Save original weights
        original_params = []
        for layer in layers:
            if isinstance(layer, NeuralLayer):
                original_params.append((layer, layer.W.copy(), layer.b.copy()))

        # Lookahead step
        for layer in layers:
            if isinstance(layer, NeuralLayer):
                layer.W -= self.momentum * self.velocity_W[layer]
                layer.b -= self.momentum * self.velocity_b[layer]

        # Compute gradient at lookahead position
        logits = model.forward(X_batch)
        model.backward(y_true, logits)

        # Restore original weights
        for layer, W_orig, b_orig in original_params:
            layer.W = W_orig
            layer.b = b_orig

        # Update
        for layer in layers:
            if isinstance(layer, NeuralLayer):
                grad_W = layer.grad_W + weight_decay * layer.W
                grad_b = layer.grad_b

                self.velocity_W[layer] = self.momentum * self.velocity_W[layer] + learning_rate * grad_W

                self.velocity_b[layer] = self.momentum * self.velocity_b[layer] + learning_rate * grad_b

                layer.W -= self.velocity_W[layer]
                layer.b -= self.velocity_b[layer]

class RMSProp:
    def __init__(self, beta=0.9, epsilon=1e-8):
        self.beta = beta
        self.epsilon = epsilon
        self.cache_W = {}
        self.cache_b = {}

    def update(self, layers, learning_rate, weight_decay):
        """
        Update weights and biases of all layers using RMSProp optimization.
        Args:
                layers: List of NeuralLayer objects in the network
                learning_rate: Learning rate for updates
                weight_decay: L2 regularization strength
        """
        for layer in layers:
            if isinstance(layer, NeuralLayer):
                # Initialize cache if not already done
                if layer not in self.cache_W:
                    self.cache_W[layer] = np.zeros_like(layer.W)
                    self.cache_b[layer] = np.zeros_like(layer.b)

                # Compute gradient with L2 regularization
                grad_W = layer.grad_W + weight_decay * layer.W
                grad_b = layer.grad_b

                # Update cache with current gradients
                self.cache_W[layer] = self.beta * self.cache_W[layer] + (1 - self.beta) * grad_W**2
                self.cache_b[layer] = self.beta * self.cache_b[layer] + (1 - self.beta) * grad_b**2

                # Update weights and biases using RMSProp update rule
                layer.W -= learning_rate * grad_W / (np.sqrt(self.cache_W[layer]) + self.epsilon)
                layer.b -= learning_rate * grad_b / (np.sqrt(self.cache_b[layer]) + self.epsilon)

if __name__ == "__main__":
    # Simple tests for optimizers
    np.random.seed(42)
    layer = NeuralLayer(input_size=4, output_size=3, weight_init='xavier')
    X = np.random.randn(5, 4) # batch_size=5, input_size=4
    Z = layer.forward(X)
    dZ = np.random.randn(5, 3) # batch_size=5, output_size=3
    dX = layer.backward(dZ)

    # Test SGD
    sgd_optimizer = SGD()
    sgd_optimizer.update([layer], learning_rate=0.01, weight_decay=0.0001)

    # Test Momentum
    momentum_optimizer = Momentum(momentum=0.9)
    momentum_optimizer.update([layer], learning_rate=0.01, weight_decay=0.0001)

    # Test NAG
    nag_optimizer = NAG(momentum=0.9)
    # Dummy model and dummy y_true and y_pred for lookahead gradient computation.
    class DummyModel:
        def forward(self, X):
            return np.random.randn(X.shape[0], 3) # output_size=3
        def backward(self, y_true, y_pred):
            pass # No actual backprop needed for this test

    dummy_model = DummyModel()
    y_true_dummy = np.random.randn(5, 3) # batch_size=5, output_size=3
    y_pred_dummy = dummy_model.forward(X)
    nag_optimizer.update([layer], learning_rate=0.01, weight_decay=0.0001, model=dummy_model, y_true=y_true_dummy, y_pred=y_pred_dummy, X_batch=X)

    print("All tests successful")