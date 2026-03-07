"""
Convert existing .npz model files to .npy format
"""

import numpy as np
import os
import glob

def convert_npz_to_npy(npz_path):
    """Convert a single npz file to npy format"""
    # Load npz file
    data = np.load(npz_path, allow_pickle=True)
    weight_dict = dict(data.items())
    
    # Create npy filename
    npy_path = npz_path.replace('.npz', '.npy')
    
    # Save as npy
    np.save(npy_path, weight_dict, allow_pickle=True)
    print(f"Converted: {npz_path} -> {npy_path}")
    
    # Verify the conversion
    loaded = np.load(npy_path, allow_pickle=True).item()
    print(f"  Keys in converted file: {list(loaded.keys())}")
    return npy_path

if __name__ == "__main__":
    # Find all npz files in models directory
    models_dir = "models"
    npz_files = glob.glob(os.path.join(models_dir, "*.npz"))
    
    print(f"Found {len(npz_files)} .npz files to convert\n")
    
    for npz_file in npz_files:
        try:
            convert_npz_to_npy(npz_file)
        except Exception as e:
            print(f"Error converting {npz_file}: {e}")
    
    print("\nConversion complete!")
    print("\nNote: Original .npz files are kept. You can delete them if needed.")
