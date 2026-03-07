"""
Script to convert old weight indexing (W0, W2, W4) to new indexing (W0, W1, W2)
Old format used position in layers list (including activations)
New format uses sequential numbering of only weight layers
"""

import numpy as np
import os
import glob

def convert_weight_dict(old_dict):
    """Convert weight dict from old indexing to new indexing."""
    new_dict = {}
    
    # Get all weight keys and sort them
    w_keys = sorted([k for k in old_dict.keys() if k.startswith('W')])
    b_keys = sorted([k for k in old_dict.keys() if k.startswith('b')])
    
    print(f"Old keys: {w_keys + b_keys}")
    
    # Renumber sequentially
    for new_idx, old_key in enumerate(w_keys):
        new_w_key = f"W{new_idx}"
        new_dict[new_w_key] = old_dict[old_key]
        print(f"  {old_key} → {new_w_key}: shape {old_dict[old_key].shape}")
    
    for new_idx, old_key in enumerate(b_keys):
        new_b_key = f"b{new_idx}"
        new_dict[new_b_key] = old_dict[old_key]
        print(f"  {old_key} → {new_b_key}: shape {old_dict[old_key].shape}")
    
    return new_dict

def convert_model_file(filepath):
    """Convert a single model file."""
    print(f"\nConverting: {filepath}")
    
    # Load old format
    if filepath.endswith('.npy'):
        old_weights = np.load(filepath, allow_pickle=True).item()
    elif filepath.endswith('.npz'):
        old_weights = dict(np.load(filepath))
    else:
        print(f"  Skipping unknown format: {filepath}")
        return
    
    # Convert to new format
    new_weights = convert_weight_dict(old_weights)
    
    # Save with new format (backup old file first)
    backup_path = filepath + '.backup'
    if not os.path.exists(backup_path):
        print(f"  Creating backup: {backup_path}")
        if filepath.endswith('.npy'):
            np.save(backup_path, old_weights, allow_pickle=True)
        else:
            np.savez(backup_path, **old_weights)
    
    # Save converted file
    output_path = filepath.replace('.npz', '.npy') if filepath.endswith('.npz') else filepath
    np.save(output_path, new_weights, allow_pickle=True)
    print(f"  Saved: {output_path}")

def main():
    # Find all model files
    model_dir = "models"
    patterns = ["*.npy", "*.npz"]
    
    model_files = []
    for pattern in patterns:
        model_files.extend(glob.glob(os.path.join(model_dir, pattern)))
    
    # Exclude backup files
    model_files = [f for f in model_files if not f.endswith('.backup')]
    
    print(f"Found {len(model_files)} model files to convert:")
    for f in model_files:
        print(f"  - {f}")
    
    # Convert each file
    for filepath in model_files:
        convert_model_file(filepath)
    
    print("\n✓ Conversion complete!")
    print("Original files backed up with .backup extension")

if __name__ == "__main__":
    main()
