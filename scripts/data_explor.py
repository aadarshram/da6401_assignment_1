"""
Script to log W&B Table for data exploration and class distribution
"""

# Imports
import numpy as np
import wandb
import keras.datasets

# Load dataset
(X_train, y_train), _ = keras.datasets.mnist.load_data()
# Uses only train data

# Initialize W&B run
wandb.init(project="da6401_assignment_1", name="data_exploration")

# Log 5 sample image from each class to W&B Table
table = wandb.Table(columns=["image", "label"])
for class_label in np.unique(y_train):
    class_indices = np.where(y_train == class_label)[0]
    sample_indices = np.random.choice(class_indices, size=5, replace=False)
    for idx in sample_indices:
        table.add_data(wandb.Image(X_train[idx]), int(y_train[idx]))
wandb.log({"data_exploration/class_samples": table})

