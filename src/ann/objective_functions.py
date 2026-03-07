"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

# Imports
from ann.activations import Softmax
import numpy as np

class MSE:
    """
    Mean Squared Error Loss
    """
    def compute_loss(self, y_true, y_pred):
        """
        Compute Mean Squared Error Loss
        Args:
            y_true: shape = (batch_size, output_size) - True labels
            y_pred: shape = (batch_size, output_size) - Predicted outputs
        Returns:
            Loss: shape = (1,) - MSE Loss
        """
        batch_size = y_true.shape[0]
        loss = (1 / batch_size) * np.sum((y_true - y_pred) ** 2)
        return loss
    
    def gradient(self, y_true, y_pred):
        """
        Computes gradient of MSE Loss wrt output predictions
        Args:
            y_true: shape = (batch_size, output_size) - True labels
            y_pred: shape = (batch_size, output_size) - Predicted outputs
        Returns:
            dZ: shape = (batch_size, output_size) - Gradient of loss wrt output predictions
        """
        batch_size = y_true.shape[0]
        # Normalize by batch_size
        dZ = 2 * (y_pred - y_true) / batch_size
        return dZ

class CrossEntropy:
    """
    Cross-Entropy Loss for multi-class classification
    NOTE: Assumes model output is raw logits without softmax activation. Applies the softmax function to the output logits to get predicted probabilities before computing loss and gradient. This is a common practice for numerical stability when computing cross-entropy loss with softmax outputs. The gradients become simpler when doing so.
    """
    def compute_loss(self, y_true, y_pred):
        """
        Compute Cross-Entropy Loss
        Args:
            y_true: shape = (batch_size, output_size) - True labels (one-hot),
                    OR (batch_size, 1) / (batch_size,) - class indices
            y_pred: shape = (batch_size, output_size) - Predictions
        Returns:
            Loss: shape = (1,) - Cross-Entropy Loss
        """
        batch_size = y_pred.shape[0]
        output_size = y_pred.shape[1] if y_pred.ndim > 1 else y_pred.shape[0]

        # If model output is raw logits
        # Apply softmax
        softmax = Softmax()
        y_pred = softmax.forward(y_pred)
        self.output = y_pred # For backprop
        # Convert class indices to one-hot if necessary
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] != output_size):
            indices = y_true.flatten().astype(int)
            y_true_onehot = np.zeros((batch_size, output_size))
            y_true_onehot[np.arange(batch_size), indices] = 1.0
            y_true = y_true_onehot
        # Add small epsilon for numerical stability - to avoid log(0) and log(1)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = (-1 / batch_size) * np.sum(y_true * np.log(y_pred_clipped))
        return loss
    
    def gradient(self, y_true, y_pred):
        """
        Computes gradient of Cross-Entropy Loss wrt output predictions
        Args:
            y_true: shape = (batch_size, output_size) - True labels (one-hot),
                    OR (batch_size, 1) / (batch_size,) - class indices
            y_pred: shape = (batch_size, output_size) - Predictions (raw logits)
        Returns:
            dZ: shape = (batch_size, output_size) - Gradient of loss wrt output predictions
        """
        batch_size = y_pred.shape[0]
        output_size = y_pred.shape[1] if y_pred.ndim > 1 else y_pred.shape[0]
        # If model output is raw logits, apply softmax
        softmax = Softmax()
        y_pred_softmax = softmax.forward(y_pred)
        # Convert class indices to one-hot if necessary
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] != output_size):
            indices = y_true.flatten().astype(int)
            y_true_onehot = np.zeros((batch_size, output_size))
            y_true_onehot[np.arange(batch_size), indices] = 1.0
            y_true = y_true_onehot
        # Normalize by batch_size
        dZ = (y_pred_softmax - y_true) / batch_size
        return dZ

if __name__ == "__main__":
    pass
    # # Simple tests
    # np.random.seed(42)
    # y_true = np.array([[0, 1, 0], [1, 0, 0]]) # batch_size=2, output_size=3 (one-hot encoded)
    # y_pred_logits = np.array([[0.2, 0.5, 0.3], [0.6, 0.2, 0.2]]) # batch_size=2, output_size=3 (raw logits)
    # ce_loss = CrossEntropy()
    # loss = ce_loss.compute_loss(y_true, y_pred_logits)
    # dZ = ce_loss.gradient(y_true, y_pred_logits)
    # mse_loss = MSE()
    # loss_mse = mse_loss.compute_loss(y_true, y_pred_logits)
    # dZ_mse = mse_loss.gradient(y_true, y_pred_logits)
    # print("All tests successful")
