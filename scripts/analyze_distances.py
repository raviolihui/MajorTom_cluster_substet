"""
Analyze the distance distribution in the embedding space.
This helps determine what threshold values make sense for Birch clustering.
"""

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import os

# Load embeddings
embeddings_path = "/data/databases/MajorTom5T/outputs/embeddings.npz"
print(f"Loading embeddings from {embeddings_path} …")
data = np.load(embeddings_path, allow_pickle=True)
embeddings = data["embeddings"].astype(np.float32)

print(f"Embeddings shape: {embeddings.shape}")

# PCA reduction (same as step2)
print("\nRunning PCA: 1024 → 280 dims …")
pca = PCA(n_components=280, random_state=42, svd_solver="randomized")
X = pca.fit_transform(embeddings)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum() * 100:.1f}%")


print(f"Final X shape: {X.shape}")

# Sample for efficiency
sample_size = min(10000, len(X))
print(f"\nSampling {sample_size} points for distance analysis …")
sample_idx = np.random.choice(len(X), sample_size, replace=False)
X_sample = X[sample_idx]

# Compute pairwise distances on sample
print("Computing pairwise distances (this may take a minute) …")
dists = pairwise_distances(X_sample, metric='euclidean')

# Extract non-zero distances
nonzero_dists = dists[dists > 0]

print("\n" + "="*60)
print("DISTANCE DISTRIBUTION STATISTICS")
print("="*60)
print(f"Min distance:        {nonzero_dists.min():.8f}")
print(f"5th percentile:      {np.percentile(nonzero_dists, 5):.8f}")
print(f"25th percentile:     {np.percentile(nonzero_dists, 25):.8f}")
print(f"Median (50th):       {np.percentile(nonzero_dists, 50):.8f}")
print(f"75th percentile:     {np.percentile(nonzero_dists, 75):.8f}")
print(f"95th percentile:     {np.percentile(nonzero_dists, 95):.8f}")
print(f"Max distance:        {nonzero_dists.max():.8f}")
print("="*60)

# Recommendations
print("\nRECOMMENDED BIRCH THRESHOLD VALUES:")
print(f"  - For ~1000 clusters:  ~{np.percentile(nonzero_dists, 99):.4f}")
print(f"  - For ~10k clusters:   ~{np.percentile(nonzero_dists, 95):.4f}")
print(f"  - For ~100k clusters:  ~{np.percentile(nonzero_dists, 85):.4f}")
print(f"  - For ~1M clusters:    ~{np.percentile(nonzero_dists, 50):.4f}")
print(f"\nYour current thresholds:")
print(f"  - 0.0001: likely TOO LOW (below min {nonzero_dists.min():.8f})")
print(f"  - 0.001:  likely TOO LOW (below min {nonzero_dists.min():.8f})")
print("="*60)



# Loading embeddings from /data/databases/MajorTom5T/outputs/embeddings.npz …
# Embeddings shape: (2245886, 1024)

# Running PCA: 1024 → 280 dims …
# PCA explained variance: 90.8%
# L2-normalizing …
# Final X shape: (2245886, 280)

# Sampling 10000 points for distance analysis …
# Computing pairwise distances (this may take a minute) …

# ============================================================
# DISTANCE DISTRIBUTION STATISTICS
# ============================================================
# Min distance:        0.00000002
# 5th percentile:      1.10475087
# 25th percentile:     1.33073938
# Median (50th):       1.43918622
# 75th percentile:     1.51368809
# 95th percentile:     1.58624959
# Max distance:        1.73962986
# ============================================================

# RECOMMENDED BIRCH THRESHOLD VALUES:
#   - For ~1000 clusters:  ~1.6242
#   - For ~10k clusters:   ~1.5862
#   - For ~100k clusters:  ~1.5444
#   - For ~1M clusters:    ~1.4392

# Your current thresholds:
#   - 0.0001: likely TOO LOW (below min 0.00000002)
#   - 0.001:  likely TOO LOW (below min 0.00000002)
# ============================================================