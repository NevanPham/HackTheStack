"""Script to update existing K-Means model with PCA and normalization ranges"""
import sys
from pathlib import Path
import joblib
import json
import numpy as np

# Paths
pca_src = Path("preprocessed/kmeans/tfidf_1000/pca_2d.pkl")
pca_dst = Path("models/kmeans/k3/tfidf_1000/pca_2d.pkl")
meta_path = Path("models/kmeans/k3/tfidf_1000/feature_metadata.json")
train_2d_path = Path("preprocessed/kmeans/tfidf_1000/X_train_2d.npy")

print("Updating K-Means model with PCA and normalization ranges...")

# Copy PCA model
if pca_src.exists():
    pca_dst.parent.mkdir(parents=True, exist_ok=True)
    pca_model = joblib.load(pca_src)
    joblib.dump(pca_model, pca_dst)
    print(f"✓ PCA model copied to {pca_dst}")
else:
    print(f"✗ PCA source file not found at {pca_src}")
    sys.exit(1)

# Update metadata with normalization ranges
if train_2d_path.exists() and meta_path.exists():
    X_train_2d = np.load(train_2d_path)
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    meta["pca_2d_normalization"] = {
        "x_min": float(np.min(X_train_2d[:, 0])),
        "x_max": float(np.max(X_train_2d[:, 0])),
        "y_min": float(np.min(X_train_2d[:, 1])),
        "y_max": float(np.max(X_train_2d[:, 1]))
    }
    
    print(f"Normalization ranges:")
    print(f"  X: [{meta['pca_2d_normalization']['x_min']:.2f}, {meta['pca_2d_normalization']['x_max']:.2f}]")
    print(f"  Y: [{meta['pca_2d_normalization']['y_min']:.2f}, {meta['pca_2d_normalization']['y_max']:.2f}]")
    
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"✓ Metadata updated at {meta_path}")
else:
    print(f"✗ Missing files: train_2d={train_2d_path.exists()}, meta={meta_path.exists()}")

print("\nDone! The K-Means model is now ready for 2D visualization.")

