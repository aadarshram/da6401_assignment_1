"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
import json
import wandb

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset
from ann.activations import Softmax

def parse_arguments():
    """
    Parse command-line arguments for inference.
    """    
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('-m_p', '--model_path', type=str, default="models/best_model.npz", help='Path to saved model weights (relative path)')
    parser.add_argument('-d', '--dataset', type=str, default="mnist", help='Dataset to evaluate on. Supported: mnist, fashion_mnist. For custom datasets, implement loading logic in main().')

    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('-h_s', '--hidden_size', nargs='+', type=int, default=[128, 128], help='List of hidden layer sizes. Preferred <= 128')
    parser.add_argument('-n_l', '--num_layers', type=int, default=4, help='Number of hidden layers. Preferred <= 6.')
    parser.add_argument('-a', '--activation', type=str, default='relu', help='Activation function (relu, sigmoid, tanh)')
    # Optional 
    parser.add_argument('-i_s', '--input_size', type=int, default=784, help='Input size for custom dataset')
    parser.add_argument('-o_s', '--output_size', type=int, default=10, help='Output size for custom dataset')
    parser.add_argument('-l', '--loss', type=str, default="cross_entropy", help='Loss function to use for eval. (MSE (on raw logits) or Cross-Entropy (with softmax))')
    # Ignore other arguements; Sometimes same cli args are used for training and inference, but some args are only relevant for training. For simplicity, ignore unrecognized args here.
    args, _ = parser.parse_known_args()
    return args


def load_model(model_path):
    """
    Load weights from saved model
    """
    data = np.load(model_path, allow_pickle=True)
    return dict(data.items())

def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    Returns:
        Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    logits, loss, accuracy, precision, recall, f1_Score = model.evaluate(X_test, y_test)
    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_Score
    }

def main():
    """
    Main inference function.

    Returns:
        Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    # Initialize
    wandb.init(project="da6401_assignment_1", name='inference_results') # mode="disabled" ; for testing
    args = parse_arguments()
    
    # Load config from saved model to ensure architecture matches
    config_path = args.model_path.replace("_model.npz", "_config.json")
    try:
        with open(config_path, "r") as f:
            saved_config = json.load(f)
            # Override architecture-related args with saved config
            args.hidden_size = saved_config.get('hidden_size', args.hidden_size)
            args.activation = saved_config.get('activation', args.activation)
            args.input_size = saved_config.get('input_size', args.input_size)
            args.output_size = saved_config.get('output_size', args.output_size)
            print(f"Loaded config from {config_path}: hidden_size={args.hidden_size}, activation={args.activation}")
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using command-line args.")
    
    model = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    # Test data
    _, _, X_val, y_val, _, _ = load_dataset(args)
    # Evaluate
    results = evaluate_model(model, X_val, y_val)
    print(f"Loss: {results['loss']:.4f}, Accuracy: {results['accuracy']:.4f}, Precision: {results['precision']}, Recall: {results['recall']}, F1-Score: {results['f1_score']}")
    print("Evaluation complete!")
    # # Save results based on f1 score
    # weights = model.get_weights()
    # np.save(f"models/test/model_f1_{results['f1_score']:.4f}.npz", weights)
    # with open(f"models/test/config_f1_{results['f1_score']:.4f}.json", "w") as f:
    #     json.dump(vars(args), f, indent=4)

    # Experiment: Error analysis
    logits = results['logits']
    # Probabilities after softmax
    probs = Softmax().forward(logits)
    # Convert one-hot encoded labels to class indices
    y_true_labels = np.argmax(y_val, axis=1) if y_val.ndim > 1 else y_val
    # Plot confusion matrix in wandb
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=probs, y_true=y_true_labels, class_names=[str(i) for i in range(args.output_size)])})
    # Log some misclassified examples in wandb
    misclassified_indices = np.where(np.argmax(probs, axis=1) != y_true_labels)[0]
    misclassified_images = []

    for idx in misclassified_indices[:10]:
        img = X_val[idx].reshape(28,28)
        true_label = y_true_labels[idx]
        pred_label = np.argmax(probs[idx])

        misclassified_images.append(
            wandb.Image(img, caption=f"True:{true_label} Pred:{pred_label}")
        )

    wandb.log({"misclassified_examples": misclassified_images})
    # Log low confidence examples from that correctly predicted (where max prob is below a threshold)
    low_confidence_indices = np.where((np.max(probs, axis=1) < 0.6) & (np.argmax(probs, axis=1) == y_true_labels))[0]
    low_conf_images = []

    for idx in low_confidence_indices[:10]:
        img = X_val[idx].reshape(28,28)
        true_label = y_true_labels[idx]
        pred_label = np.argmax(probs[idx])
        conf = np.max(probs[idx])

        low_conf_images.append(
            wandb.Image(img, caption=f"True:{true_label} Pred:{pred_label} Conf:{conf:.2f}")
        )
    wandb.log({"low_confidence_examples": low_conf_images})

    return results

if __name__ == '__main__':
    main()
    # Simple tests
    # run with: python src/inference.py --model_path models/test_model.npz --dataset mnist
