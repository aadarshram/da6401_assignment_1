# DA6401 Assignment 1: Multi-Layer Perceptron (MLP) for Image Classification

## Overview

This repository contains an implementation of a configurable, modular Multi-Layer Perceptron (MLP) using only NumPy. It contains the complete training pipeline from forward propagation to various optimization strategies to classify the MNIST and Fashion-MNIST datasets.

**Links**
- [GitHub Repository](https://github.com/aadarshram/da6401_assignment_1)
- [WandB Report](https://wandb.ai/ramachandranaadarsh-indian-institute-of-technology-madras/da6401_assignment_1/reports/DA6401-Assignment-1-Report--VmlldzoxNjEyMTg1Ng?accessToken=v7zjwto3ql0mj9famz978npvm7xhqtybcaks74kyetpww038sgm3e6wvx2ugoaux)
  
## Requirements

This project uses Python 3.x and the Python dependencies listed in `requirements.txt`.

Install requirements (recommended inside a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already have the packages installed, ensure versions are compatible by checking `requirements.txt`.

## Install

Clone the repo and install dependencies (if not already done):

```bash
git clone <this-repo-url>
cd da6401_assignment_1
# create & activate virtual env, then:
pip install -r requirements.txt
```

## Quick start

### Train a model (with full parameters)

```bash
python src/train.py \
  -d mnist \
  -e 10 \
  -b 64 \
  -lr 0.001 \
  -o rmsprop \
  -nhl 3 \
  -sz 128 128 128 \
  -a relu \
  -l cross_entropy \
  -w_i xavier \
  -wd 0.0 \
  -w_p da6401_assignment_1 \
  -w_x experiment_1 \
  -msp best_model.npy
```

| Argument | Description                                                  |
| -------- | ------------------------------------------------------------ |
| `-d`     | Dataset to use (e.g., `mnist`)                               |
| `-e`     | Number of training epochs                                    |
| `-b`     | Batch size for training                                      |
| `-lr`    | Learning rate                                                |
| `-o`     | Optimizer: `sgd`, `momentum`, `nag`, `rmsprop`               |
| `-nhl`   | Number of hidden layers                                      |
| `-sz`    | Hidden layer sizes (space-separated). Example: `128 128 128` |
| `-a`     | Activation functions per layer. Example: `relu`              |
| `-l`     | Loss function. Example: `cross_entropy`                      |
| `-w_i`   | Weight initialization method. Example: `xavier`              |
| `-wd`    | Weight decay (regularization)                                |
| `-w_p`   | WandB project name for logging                               |
| `-msp`   | Path to save the trained model (use `.npy`)                  |


### Run inference 

```bash
python src/inference.py \
  --model_path best_model.npy \
  --dataset mnist
```

The inference script evaluates the saved model on the test dataset and reports:

- Loss
- Accuracy
- Precision
- Recall
- F1-score

This helps you measure the model’s generalization on unseen data.

## Project structure

```
da6401_assignment_1/
├── models/                 # (empty placeholder / outputs)
├── notebooks/              # Jupyter notebooks for experiments & WandB demos
├── src/
│   ├── ann/                # core MLP implementation
│   │   ├── activations.py
│   │   ├── neural_layer.py
│   │   ├── neural_network.py
│   │   ├── objective_functions.py
│   │   └── optimizers.py
│   ├── plots/              # (plot utilities / saved figures)
│   ├── utils/
│   │   └── data_loader.py  # dataset loading utilities
│   ├── train.py            # training entrypoint
│   ├── inference.py        # inference / evaluation script
│   ├── best_config.json    # example best hyperparameter config
│   ├── best_model.npy      # example saved model weights (NumPy)
│   └── best_model.npz      # example model checkpoint
└── requirements.txt
```

Brief file descriptions
- `src/ann/activations.py`: activation functions and derivatives (ReLU, sigmoid, softmax, etc.)
- `src/ann/neural_layer.py`: an implementation of a dense/fully-connected layer
- `src/ann/neural_network.py`: builds and orchestrates layers into a network; forward/backward passes
- `src/ann/objective_functions.py`: loss functions (cross-entropy, MSE, etc.)
- `src/ann/optimizers.py`: optimizer implementations (SGD, Momentum, Adam, etc.)
- `src/utils/data_loader.py`: routines to load MNIST/Fashion-MNIST and prepare batches
- `src/train.py`: training loop, logging, saving models/configs
- `src/inference.py`: load a saved model and run predictions / evaluation

Model artifacts
- `src/best_model.npz`, `src/best_model.npy`: example saved model weights/checkpoints
- `src/best_config.json`: hyperparameters used for the saved best model

## Visualizing / Notebooks

- `notebooks/wandb_demo.ipynb` and `notebooks/wandb_report.ipynb` contain experiment demos and reports. Open them in JupyterLab / Jupyter Notebook.
- `src/sweep.yaml` is a Weights & Biases sweep configuration for hyperparameter tuning.
