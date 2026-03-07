"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

# Imports
from ann.objective_functions import MSE, CrossEntropy
from ann.neural_layer import NeuralLayer
from ann.activations import ReLU, Sigmoid, Tanh, Softmax
from ann.optimizers import SGD, Momentum, NAG, RMSProp
from ann.objective_functions import MSE, CrossEntropy

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
        if cli_args.dataset == "mnist" or cli_args.dataset == "fashion_mnist":
            # Standard input and output sizes for MNIST and Fashion-MNIST - 28x28 image -> 10 classes
            self.input_size = 784
            self.output_size = 10
        elif cli_args.input_size and cli_args.output_size:
            self.input_size = cli_args.input_size
            self.output_size = cli_args.output_size
        else:
            raise ValueError("For datasets apart from {MNIST, Fashion MNIST}, please provide input_size and output_size as command-line arguments")
        self.batch_size = cli_args.batch_size
        self.hidden_size = cli_args.hidden_size
        self.num_layers = len(cli_args.hidden_size) + 1 # Ignore cli_arg: num_layers
        self.activation = cli_args.activation
        train_mode = "weight_init" in vars(cli_args) and "loss" in vars(cli_args) and "optimizer" in vars(cli_args) and "learning_rate" in vars(cli_args) and "weight_decay" in vars(cli_args) # Workaround for train/inference specific initialization.
        if cli_args.loss: # Workaround for use of loss functions in inference mode.
            self.loss = MSE() if cli_args.loss == "mean_squared_error" else CrossEntropy() if cli_args.loss == "cross_entropy" else NotImplementedError("Supports only 'mean_squared_error' or 'cross_entropy' loss functions")
        else:
            self.loss = None if not train_mode else NotImplementedError("Loss function must be specified for training. Supports only 'mean_squared_error' or 'cross_entropy' loss functions")
        self.weight_init = "random"
        self.optimizer = None
        if train_mode:
            self.weight_init = cli_args.weight_init
            self.loss = MSE() if cli_args.loss == "mean_squared_error" else CrossEntropy() if cli_args.loss == "cross_entropy" else NotImplementedError("Supports only 'mean_squared_error' or 'cross_entropy' loss functions")
            self.optimizer = SGD() if cli_args.optimizer == "sgd" else Momentum() if cli_args.optimizer == "momentum" else NAG() if cli_args.optimizer == "nag" else RMSProp() if cli_args.optimizer == "rmsprop" else NotImplementedError("Supports only 'sgd', 'momentum', 'nag', 'rmsprop' optimizers")
            self.learning_rate = cli_args.learning_rate
            self.weight_decay = cli_args.weight_decay

        # Initialize layers
        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(
                NeuralLayer(
                    input_size = self.input_size if i==0 else self.hidden_size[i-1],
                    output_size = self.output_size if i == self.num_layers-1 else self.hidden_size[i],
                    weight_init = self.weight_init ,
                )
            )
            if i != self.num_layers - 1: # No activation after output layer
                self.layers.append(
                    ReLU() if self.activation == "relu" else
                    Sigmoid() if self.activation == "sigmoid" else
                    Tanh() if self.activation == "tanh" else
                    Softmax() if self.activation == "softmax" else
                    NotImplementedError("Supports only 'relu', 'sigmoid', 'tanh', or 'softmax' activation functions")
                )
            
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: shape = (batch_size, input_size) or (input_size,) - Input data
            
        Returns:
            Output logits: shape = (batch_size, output_size) or (output_size,)
        """
        # Handle single sample input: reshape (input_size,) to (1, input_size)
        original_shape = X.shape
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
            
        # activations = []
        for layer in self.layers:
            X = layer.forward(X)
            # activations.append(X)
        
        # If input was single sample, return single sample output
        if single_sample:
            X = X.flatten()
            
        return X #, activations # return activation for dead neuron logging
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: shape = (batch_size, output_size) or (output_size,) - True labels
            y_pred: shape = (batch_size, output_size) or (output_size,) - Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        # Handle single sample: reshape (output_size,) to (1, output_size)
        if y_true.ndim == 1:
            y_true = y_true.reshape(1, -1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)
            
        # Initial gradient dZ
        dZ = self.loss.gradient(y_true, y_pred)

        # Backprop
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)

        grad_W_list = [layer.grad_W for layer in reversed(self.layers) if isinstance(layer, NeuralLayer)]
        grad_b_list = [layer.grad_b for layer in reversed(self.layers) if isinstance(layer, NeuralLayer)]
        # create explicit object arrays to avoid numpy trying to broadcast shapes
        grad_W = np.empty(len(grad_W_list), dtype=object)
        grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            grad_W[i] = gw
            grad_b[i] = gb
        return grad_W, grad_b
    
    def update_weights(self, y_true, y_pred, X):
        """
        Update weights using the optimizer.
        """
        if isinstance(self.optimizer, NAG):
            self.optimizer.update(self.layers, self.learning_rate, self.weight_decay, self, y_true, y_pred, X) # Pass the model instance, y_true and y_pred for optimizers like NAG that require lookahead gradient computation.
        else:
            self.optimizer.update(self.layers, self.learning_rate, self.weight_decay)
        
    
    def train(self, X_train, y_train, epochs, batch_size, **kwargs):
        """
        Train the network for specified epochs.
        """
        num_samples = X_train.shape[0]
        # Validation - if exists log val loss and metrics else ignore
        X_val = kwargs.get('X_val', None)
        y_val = kwargs.get('y_val', None)
        val_frequency = kwargs.get('val_frequency', 1 if X_val is not None else 0) # Log validation metrics every 'val_frequency' epochs. Default is 1 (every epoch).
        # Logs
        train_losses = []
        val_losses = []
        gradient_norms = [] # per epoch
        # first_layer_grad_norms = [] # per epoch - to observe vanishing/exploding gradients in the first layer separately.
        
        # # Track activations for each Tanh layer dynamically
        # num_tanh_layers = sum(1 for layer in self.layers if isinstance(layer, Tanh))
        # epoch_activations = [[] for _ in range(num_tanh_layers)]
        # neuron_grad_0_1 = [] # Log per iter for 50 iters
        # neuron_grad_1_1 = []
        # neuron_grad_2_1 = []
        # neuron_grad_3_1 = []
        # neuron_grad_4_1 = []
        # symm_iter = 0
            
        for epoch in range(epochs): # For every epoch
            batch_train_losses = []
            batch_gradient_norms = []
            # batch_first_layer_grad_norms = []
            # batch_activations = [[] for _ in range(num_tanh_layers)]
            for i in range(0, num_samples, batch_size):
                # Prepare batch
                X_train_batch = X_train[i:i+batch_size]
                y_train_batch = y_train[i:i+batch_size]
                # Forward pass
                y_pred = self.forward(X_train_batch) # activations, for dead neuron logging
                # Compute loss
                train_loss = self.loss.compute_loss(y_train_batch, y_pred)
                # Backward pass
                grad_W, grad_b = self.backward(y_train_batch, y_pred)

                # Update weights
                self.update_weights(y_train_batch, y_pred, X_train_batch) # For NAG, need to pass y and y_pred and X_train_batch for lookahead gradient computation.

                # Logging
                batch_train_losses.append(train_loss)
                grad_norm = np.sqrt(sum(np.sum(g**2) for g in grad_W) + sum(np.sum(g**2) for g in grad_b))
                batch_gradient_norms.append(grad_norm)
                # Optional - log first layer gradient norms separately to observe vanishing/exploding gradients
                # first_layer_grad_norm = np.sqrt(np.sum(grad_W[-1]**2) + np.sum(grad_b[-1]**2))
                # batch_first_layer_grad_norms.append(first_layer_grad_norm)
                
                # # Optional - log activations of each Tanh layer to observe dead neurons
                # tanh_idx = 0
                # for layer_idx, layer in enumerate(self.layers):
                #     if isinstance(layer, Tanh):
                #         # Get the Tanh output (activations[layer_idx])
                #         batch_activations[tanh_idx].append(activations[layer_idx])
                #         tanh_idx += 1
                
                # # Symmetry breaking Experiment:
                # # Log grad norms of 5 neurons in first hidden layer
                # neuron_grad_0_1.append(np.linalg.norm(grad_W[2][:, 0]))
                # neuron_grad_1_1.append(np.linalg.norm(grad_W[2][:, 1]))
                # neuron_grad_2_1.append(np.linalg.norm(grad_W[2][:, 2]))
                # neuron_grad_3_1.append(np.linalg.norm(grad_W[2][:, 3]))
                # neuron_grad_4_1.append(np.linalg.norm(grad_W[2][:, 4]))
                # symm_iter += 1
                
                # # Wandb log wrt iter - custom metric definition handles the x-axis
                # if symm_iter <= 50: # Log for first 50 iterations
                #     wandb.log({
                #         "iteration": symm_iter,
                #         "train/grad_neuron_0_layer_1": neuron_grad_0_1[-1],
                #         "train/grad_neuron_1_layer_1": neuron_grad_1_1[-1],
                #         "train/grad_neuron_2_layer_1": neuron_grad_2_1[-1],
                #         "train/grad_neuron_3_layer_1": neuron_grad_3_1[-1],
                #         "train/grad_neuron_4_layer_1": neuron_grad_4_1[-1],
                #     })
                    
            # Epoch logs
            train_losses.append(np.mean(batch_train_losses))
            gradient_norms.append(np.mean(batch_gradient_norms))
            # first_layer_grad_norms.append(np.mean(batch_first_layer_grad_norms))
            
            # # Concatenate all batch activations for the epoch
            # for i in range(num_tanh_layers):
            #     epoch_activations[i].append(np.concatenate(batch_activations[i], axis=0))

            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Gradient Norm: {gradient_norms[-1]:.4f}")
            wandb.log({"epoch": epoch+1, "train/train_loss": train_losses[-1], "train/gradient_norm": gradient_norms[-1]}) # "train/first_layer_grad_norm": first_layer_grad_norms[-1]
            
            # # Optional - log the activation distributions of each Tanh layer to observe dead neurons
            # dead_neuron_metrics = {}
            
            # for i in range(num_tanh_layers):
            #     layer_acts = epoch_activations[i][-1]  # shape: (num_samples, num_neurons)
            #     # Compute % of dead neurons (neurons that are 0 for ALL samples in this epoch)
            #     dead_neurons_all = np.mean(np.all(layer_acts == 0, axis=0)) * 100  # % neurons always 0
            #     dead_neuron_metrics[f"dead_neurons_layer_{i+1}"] = dead_neurons_all
            
            # wandb.log({**dead_neuron_metrics}, step=epoch+1)

            if (X_val is not None and y_val is not None) and (epoch + 1) % val_frequency == 0:
                _, val_loss, val_accuracy, val_precision, val_recall, val_f1_score = self.evaluate(X_val, y_val)
                val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {np.mean(val_precision):.4f}, Recall: {np.mean(val_recall):.4f}, F1-Score: {np.mean(val_f1_score):.4f}")
                wandb.log({"epoch": epoch+1, "val/val_loss": val_loss, "val/val_accuracy": val_accuracy, "val/val_precision": np.mean(val_precision), "val/val_recall": np.mean(val_recall), "val/val_f1_score": np.mean(val_f1_score)})
        
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        y_pred_logits = self.forward(X) # activations for dead neuron
        # Compute loss (loss function implicitly applies softmax if using cross-entropy. MSE works with raw logits.)
        loss = self.loss.compute_loss(y, y_pred_logits) if self.loss is not None else None
        
        # Compute probs
        if hasattr(self.loss, 'output') and self.loss.output is not None:
            y_pred = self.loss.output
        else:
            softmax = Softmax()
            y_pred = softmax.forward(y_pred_logits)
        # Metrics: Accuracy, Precision, Recall, F1-Score

        # Accuracy - Over all classes, (TP + TN) / (TP + TN + FP + FN)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred_labels == y_true_labels)

        # Precision - per class, TP / (TP + FP)
        # Recall - per class, TP / (TP + FN)
        # F1-Score - per class, 2 * (Precision * Recall) / (Precision + Recall)
        num_classes = self.output_size
        
        y_pred_one_hot = np.eye(num_classes)[y_pred_labels]
        y_true_one_hot = y
        
        tp = np.sum(y_pred_one_hot * y_true_one_hot, axis=0)
        fp = np.sum(y_pred_one_hot * (1 - y_true_one_hot), axis=0)
        fn = np.sum((1 - y_pred_one_hot) * y_true_one_hot, axis=0)
        
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0)
        f1_score = np.where(precision + recall > 0, 2 * (precision * recall) / (precision + recall), 0)

        return y_pred_logits, loss, accuracy, precision, recall, f1_score

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, NeuralLayer):
                d[f"W{i}"] = layer.W.copy()
                d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        errors = []
        weight_info = []
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, NeuralLayer):
                w_key = f"W{i}"
                b_key = f"b{i}"
                
                if w_key in weight_dict:
                    loaded_W = weight_dict[w_key].copy()
                    expected_shape = (layer.input_size, layer.output_size)
                    
                    # Log weight info
                    weight_info.append(f"  W{i}: loaded={loaded_W.shape}, expected={expected_shape} "
                                     f"[in={layer.input_size}, out={layer.output_size}]")
                    
                    # Check if dimensions match exactly
                    if loaded_W.shape == expected_shape:
                        layer.W = loaded_W
                    # Check if dimensions are transposed
                    elif loaded_W.shape == (layer.output_size, layer.input_size):
                        print(f"Warning: W{i} loaded with transposed shape {loaded_W.shape}, expected {expected_shape}. Auto-transposing.")
                        layer.W = loaded_W.T
                    else:
                        errors.append(f"W{i}: loaded shape {loaded_W.shape}, "
                                    f"expected {expected_shape} or transposed {(layer.output_size, layer.input_size)}")
                        # Don't load incompatible weights
                
                if b_key in weight_dict:
                    loaded_b = weight_dict[b_key].copy()
                    expected_b_shape = (1, layer.output_size)
                    
                    weight_info.append(f"  b{i}: loaded={loaded_b.shape}, expected={expected_b_shape}")
                    
                    # Handle different bias shapes
                    if loaded_b.shape == expected_b_shape:
                        layer.b = loaded_b
                    elif loaded_b.shape == (layer.output_size,):
                        layer.b = loaded_b.reshape(1, -1)
                    elif loaded_b.shape == (layer.output_size, 1):
                        layer.b = loaded_b.T
                    else:
                        errors.append(f"b{i}: loaded shape {loaded_b.shape}, expected {expected_b_shape}")
        
        # Print all weight information
        print("\n=== Weight Loading Details ===")
        for info in weight_info:
            print(info)
        print("==============================\n")
        
        # If there were errors, raise comprehensive error message
        if errors:
            error_msg = "Weight dimension mismatch(es) detected:\n" + "\n".join(errors)
            error_msg += f"\n\nModel architecture: input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size}"
            error_msg += f"\nTotal layers: {len([l for l in self.layers if isinstance(l, NeuralLayer)])}"
            raise ValueError(error_msg)

    def save_weights(self, model_save_path):
        """
        Save model weights to disk as .npy file.
        """
        weights = self.get_weights()
        # Ensure path has .npy extension
        if not model_save_path.endswith('.npy'):
            model_save_path = model_save_path.replace('.npz', '.npy')
        np.save(model_save_path, weights, allow_pickle=True)

if __name__ == "__main__":
    # Simple tests
    # Test if train, eval, save_weights and load_weights methods run
    np.random.seed(42)
    cli_args = lambda: None # Simple namespace for CLI args
    cli_args.dataset = "mnist"
    cli_args.batch_size = 32
    cli_args.hidden_size = [64, 32]
    cli_args.activation = "relu"
    cli_args.weight_init = "xavier"
    cli_args.loss = "cross_entropy"
    cli_args.optimizer = "sgd"
    cli_args.learning_rate = 0.01
    cli_args.weight_decay = 0.0001
    import wandb
    wandb.init(project="da6401_assignment_1", name="test_run_neural_network", reinit=True, mode="disabled") # "disabled" mode for testing without logging to wandb server.
    model = NeuralNetwork(cli_args)
    X = np.random.randn(10, 784) # batch_size=10, input_ize=784
    y = np.random.randint(0, 10, size=(10, 10)) # batch_size=10, output_size=10 (one-hot encoded)
    model.train(X, y, epochs=5, batch_size=2)
    y_pred_logits, loss, accuracy, precision, recall, f1_score = model.evaluate(X, y)
    model.save_weights("models/test_model_weights.npz")
    model.load_weights("models/test_model_weights.npz")
    wandb.finish()
    print("All tests successful")