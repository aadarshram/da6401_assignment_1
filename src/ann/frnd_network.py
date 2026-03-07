"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

from .neural_layer import NeuralLayer
from .optimizers import SGD, Momentum, NAG, RMSProp
from .objective_functions import CrossEntropyLoss, MSELoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
import wandb

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """

        # initialize layers, epochs, loss function, batch size and optimizer based on cli_args
        self.layers = []
        self.epochs = cli_args.epochs
        self.loss_fn = CrossEntropyLoss() if cli_args.loss == "cross_entropy" else MSELoss()
        self.batch_size = cli_args.batch_size

        if cli_args.optimizer == "sgd":
            self.optimizer = SGD(cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == "momentum":
            self.optimizer = Momentum(cli_args.learning_rate, weight_decay=cli_args.weight_decay)      
        elif cli_args.optimizer == "nag":
            self.optimizer = NAG(cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == "rmsprop":
            self.optimizer = RMSProp(cli_args.learning_rate, weight_decay=cli_args.weight_decay)

        # create layers based on input size, hidden layer sizes, output size, activation functions, and weight initialization method
        layer_sizes = [784] + \
                      cli_args.hidden_size + \
                      [10]
        
        for i in range(len(layer_sizes) - 2):
            self.layers.append(
                NeuralLayer(
                    layer_sizes[i],
                    layer_sizes[i+1],
                    activation=cli_args.activation,
                    weight_init=cli_args.weight_init
                )
            )

        self.layers.append(
            NeuralLayer(
                layer_sizes[-2],
                layer_sizes[-1],
                activation="linear",
                weight_init=cli_args.weight_init
            )
        )
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """

        # Forward pass through each layer
        for layer in self.layers:
            X = layer.forward(X)

        return X
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        grad_W_list = []
        grad_b_list = []

        # Compute initial gradient from loss function
        dA = self.loss_fn.backward(y_true, y_pred)

        # Backward pass through each layer in reverse order
        for layer in reversed(self.layers):
            dA, grad_W, grad_b = layer.backward(dA)
            grad_W_list.append(grad_W)
            grad_b_list.append(grad_b)

        # Create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        # update weights for each layer using the optimizer and the computed gradients
        self.optimizer.step(self.layers)
    
    def train(self, X_train, y_train, args):
        """
        Train the network for specified epochs.
        """

        # Stratified train validation split (90% train, 10% val)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.1,      
            random_state=42,    
            stratify=y_train   
        )

        # Helper function to create batches
        def create_batches(X, y, batch_size, shuffle=True):
            n_samples = X.shape[0]
            indices = np.arange(n_samples)

            if shuffle:
                np.random.shuffle(indices)

            for start_idx in range(0, n_samples, args.batch_size):
                end_idx = start_idx + args.batch_size
                batch_indices = indices[start_idx:end_idx]

                yield X[batch_indices], y[batch_indices]

        # Placeholders for tracking metrics
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        train_precisions = []
        val_precisions = []
        train_recalls = []
        val_recalls = []    
        train_f1s = []  
        val_f1s = []
        all_train_preds = []
        all_train_labels = []

        # # Experiment 2.9 tracking variables
        # iteration = 0
        # LOG_NEURONS = 5       
        # LOG_LAYER   = 0      
        # MAX_ITER    = 50      

        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = 0
            correct_predictions = 0
            total_samples = 0

            for X_batch, y_batch in create_batches(X_train, y_train, args.batch_size):
                logits = self.forward(X_batch)
                loss = self.loss_fn.forward(y_batch, logits)

                epoch_loss += loss
                num_batches += 1

                predictions = np.argmax(logits, axis=1)
                correct_predictions += np.sum(predictions == y_batch)
                total_samples += y_batch.shape[0]

                all_train_preds.extend(predictions)
                all_train_labels.extend(y_batch)

                self.backward(y_batch, logits)
                self.update_weights()

                # commented out for now to avoid cluttering logs
                # if iteration < MAX_ITER:
                #     layer_grad = self.grad_W[-(LOG_LAYER + 1)]  # shape: (in, out)
                #     neuron_log = {"iteration": iteration}
                #     for n in range(LOG_NEURONS):
                #         # mean absolute gradient across all incoming weights for neuron n
                #         neuron_log[f"neuron_{n}_grad_layer{LOG_LAYER}"] = float(
                #             np.mean(np.abs(layer_grad[:, n]))
                #         )
                #     wandb.log(neuron_log)

                # iteration += 1

            train_loss = epoch_loss / num_batches
            train_acc = correct_predictions / total_samples
            train_precision = precision_score(all_train_labels, all_train_preds, average="macro")
            train_recall = recall_score(all_train_labels, all_train_preds, average="macro")
            train_f1 = f1_score(all_train_labels, all_train_preds, average="macro")

            val_loss, val_acc, val_precision, val_recall, val_f1 = self.evaluate(X_val, y_val)

            if epoch > 0:
                convergence_rate = (train_losses[-1] - train_loss) / train_losses[-1]
            else:
                convergence_rate = 0

            weight_norms = {}
            grad_norms = {}

            for i, layer in enumerate(self.layers):
                if hasattr(layer, "W"):
                    weight_norm = np.linalg.norm(layer.W)
                    weight_norms[f"layer_{i}_weight_norm"] = weight_norm

                if hasattr(layer, "grad_W"):
                    grad_norm = np.linalg.norm(layer.grad_W)
                    grad_norms[f"layer_{i}_grad_norm"] = grad_norm

            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "overfitting_gap": train_loss - val_loss,
                "convergence_rate": convergence_rate,
            }

            log_dict.update(weight_norms)
            log_dict.update(grad_norms)
                        
            # Log metrics to wandb
            wandb.log(log_dict)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            train_precisions.append(train_precision)
            val_precisions.append(val_precision)
            train_recalls.append(train_recall)
            val_recalls.append(val_recall)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)


            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}, "
                  f"Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.4f}")
        
        # Log final metrics after training completes
        final_metrics = {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_train_accuracy': train_accuracies[-1],
            'final_val_accuracy': val_accuracies[-1],
            'final_train_precision': train_precisions[-1],
            'final_val_precision': val_precisions[-1],
            'final_train_recall': train_recalls[-1],
            'final_val_recall': val_recalls[-1],
            'final_train_f1': train_f1s[-1],
            'final_val_f1': val_f1s[-1],
            'best_val_loss': min(val_losses),
            'best_val_accuracy': max(val_accuracies),
            'convergence_epoch': np.argmin(val_losses),
        }

        wandb.log(final_metrics)


    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        logits = self.forward(X)

        # Compute loss
        loss = self.loss_fn.forward(y, logits)

        # Compute predictions
        predictions = np.argmax(logits, axis=1)

        # Compute all metrics
        accuracy = np.mean(predictions == y)
        precision = precision_score(y, predictions, average="macro")
        recall = recall_score(y, predictions, average="macro")
        f1 = f1_score(y, predictions, average="macro")

        return loss, accuracy, precision, recall, f1
    
    def get_weights(self):

        # Return a dictionary of weights and biases for all layers, keyed by "W0", "b0", "W1", "b1", etc.
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):

        # Set weights and biases for all layers from a dictionary, keyed by "W0", "b0", "W1", "b1", etc.
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()