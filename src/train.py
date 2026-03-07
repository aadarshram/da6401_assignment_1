"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt # For plotting training curves
import wandb # For experiment tracking
import json
import ast

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset

def parse_hidden_size_type(value):
    """
    Custom type parser for hidden_size argument.
    Handles both individual integers (e.g., 64 64 64) and string lists (e.g., "[64, 64, 64]").
    """
    # Check if it's a string representation of a list from wandb
    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        return ast.literal_eval(value)
    return int(value)

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    
    parser.add_argument('-d', '--dataset', type=str, default="fashion_mnist", help='Dataset for training (mnist or fashion_mnist)')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer')
    parser.add_argument('-o', '--optimizer', type=str, default="rmsprop", help='Optimizer to use (sgd, momentum, nag, rmsprop)')
    parser.add_argument('-sz', '--hidden_size', nargs='+', type=parse_hidden_size_type, default=[128, 128, 128], help='List of hidden layer sizes (e.g., "64 64 64" or "[64, 64, 64]"). Recommended < 128')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, help='Number of hidden layers. Preferred < 6.')
    parser.add_argument('-a', '--activation', type=str, default="relu", help='Activation function to use (sigmoid, relu, tanh)')
    parser.add_argument('-l', '--loss', type=str, default="cross_entropy", help='Loss function to use (MSE (on raw logits) or Cross-Entropy (with softmax)). Choose from ["mean_squared_error", "cross_entropy"]')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0001, help='L2 regularization strength (default: 0.0001)')
    parser.add_argument('-w_i', '--weight_init', type=str, default="xavier", help='Weight initialization method (random or xavier)')
    parser.add_argument('-w_p', '--wandb_project', type=str, default="da6401_assignment_1", help='W&B project name')
    parser.add_argument('-m_p', '--model_path', type=str, default="models/fmnist_best_model.npz", help='Path to save trained model (relative path)') # models/best_model.npz
    # OPTIONAL

    # Optional arguements for custom datasets. For MNIST and Fashion-MNIST, input and output sizes are fixed to 784 and 10 respectively.
    parser.add_argument('-i_s', '--input_size', type=int, default=784, help='Input size for custom dataset')
    parser.add_argument('-o_s', '--output_size', type=int, default=10, help='Output size for custom dataset')
    # NOTE: One can attempt to extract the sizes from the dataset itsef, but extracting class labels from the data assumes all classes are represented which may not always be the case. Hence, I ask users to specify especially for custom datasets.
    # Optional - val_size for train-val split. Default is 0.1 (10% of training data for validation)
    parser.add_argument('-v_s', '--val_size', type=float, default=0.1, help='Validation set size as a fraction of training data (default: 0.1)')
    # Optional - val_frequency for logging validation metrics every 'val_frequency' epochs. Default is 2
    parser.add_argument('-v_f', '--val_frequency', type=int, default=2, help='Frequency (in epochs) for logging validation metrics (default: 2)')
    
    args = parser.parse_args()
    
    # Post-process hidden_size: if it's [[64, 64, 64]], flatten to [64, 64, 64]
    if isinstance(args.hidden_size, list) and len(args.hidden_size) == 1 and isinstance(args.hidden_size[0], list):
        args.hidden_size = args.hidden_size[0]
    
    return args


def main():
    """
    Main training function.
    """
    args = parse_arguments()

    # Logging
    wandb.init(project=args.wandb_project, name='fmnist_best_model') # mode="disabled" ; for testing

    # Define custom metrics for separate x-axes
    # wandb.define_metric("iteration")
    wandb.define_metric("epoch")
    # wandb.define_metric("train/grad_neuron_*", step_metric="iteration")
    wandb.define_metric("train/train_loss", step_metric="epoch")
    wandb.define_metric("train/gradient_norm", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

    # Apply wandb config if exists (for sweeps)
    if wandb.config:
        for key, value in wandb.config.items():
            # Workaround for list parsing in wandb sweep agent)
            if key == 'hidden_size':
                if isinstance(value, str):
                    # Handle string representation like "[128, 128, 128]"
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list):
                        value = parsed
            setattr(args, key, value)
    wandb.config.update(vars(args), allow_val_change=True)
    
    # Initialize model
    model = NeuralNetwork(args)
    # Load data
    X_train, y_train, X_val, y_val, _, _ = load_dataset(args)
    
    # Train model
    model.train(X_train, y_train, args.epochs, args.batch_size, X_val=X_val, y_val=y_val, val_frequency=args.val_frequency) # val args are kwargs to allow optional validation during training

    # Finish
    model.save_weights(args.model_path)
    # Save config
    with open(args.model_path.replace("_model.npz", "_config.json"), "w") as f: # best_config,json
        json.dump(vars(args), f, indent=4)
    wandb.finish()
    print("Training complete!")


if __name__ == '__main__':
    main()
    # Simple tests
    # Update mode='diabled' in wandb.init() in main() to run without logging to wandb server for testing purposes.
    # run in cli with: python src/train.py --epochs 1 --wandb_project "test_run_da6401_1" --model_path "models/test_model.npz"
    # print("All tests successful")

