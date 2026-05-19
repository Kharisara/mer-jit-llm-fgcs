# mer/utils/pca_fit_states.py
"""
Fit PCA (2 components) on saved .npy state vectors.

This script is defensive:
 - resolves project root as two levels above this file (repo root)
 - searches the expected state folder, prints diagnostics
 - exits cleanly if no .npy files found
"""

import os
import glob
import joblib
import numpy as np
from sklearn.decomposition import PCA

# === Resolve paths ===
# __file__ -> .../mer/utils/pca_fit_states.py
# project_root -> parent of mer -> repo root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))   # <-- correct: parent of 'mer'
STATE_ROOT = os.path.join(PROJECT_ROOT, "data", "processed", "MELD_state_embeddings_tav_context")
OUT_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)

def collect_files(root, ext="*.npy"):
    files = glob.glob(os.path.join(root, ext))
    files = sorted(files)
    return files

def load_sample(files, max_samples=5000):
    X = []
    for i,f in enumerate(files):
        if i >= max_samples:
            break
        try:
            a = np.load(f)
            a = a.ravel().astype(np.float32)
            # skip obviously bad shapes
            if a.size == 0 or not np.isfinite(a).all():
                continue
            X.append(a)
        except Exception:
            continue
    if len(X) == 0:
        return None
    X = np.stack(X, axis=0)
    return X

if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("Looking for state files in:", STATE_ROOT)
    if not os.path.exists(STATE_ROOT):
        print("State root does not exist. Try running the extractor or check folder name.")
        # As a helpful fallback, search recursively in the project for any .npy and print a few matches
        all_npy = glob.glob(os.path.join(PROJECT_ROOT, "**", "*.npy"), recursive=True)
        print("Total .npy files found recursively in project:", len(all_npy))
        print("Examples:")
        for p in all_npy[:10]:
            print("  ", p)
        raise SystemExit(1)

    files = collect_files(STATE_ROOT, "*.npy")
    print("Found state files (in state_root):", len(files))
    if len(files) == 0:
        # fallback recursive search
        all_npy = glob.glob(os.path.join(PROJECT_ROOT, "**", "*.npy"), recursive=True)
        print("No .npy files in the expected folder. Recursive search found:", len(all_npy))
        print("Examples:")
        for p in all_npy[:10]:
            print("  ", p)
        raise SystemExit(1)

    X = load_sample(files, max_samples=5000)
    if X is None:
        print("No valid state arrays loaded (all files empty or invalid). Aborting.")
        raise SystemExit(1)

    print("Loaded X shape:", X.shape)
    # Optionally check dimensionality consistency
    dists = [x.shape[0] for x in X]
    uniq_dims = sorted(list(set(dists)))
    print("Unique state dims in loaded sample (showing up to 10):", uniq_dims[:10])
    # If dims are not identical, we will trim/pad to the median dim before PCA
    target_dim = X.shape[1]
    print("Using target_dim for PCA:", target_dim)

    pca = PCA(n_components=2, random_state=42)
    pca.fit(X)
    out_path = os.path.join(OUT_DIR, "pca_states_2.pkl")
    joblib.dump(pca, out_path)
    print("Saved PCA to", out_path)
