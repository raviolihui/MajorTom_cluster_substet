from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

# After PCA normalization (from step2_kmeans_balance.py)
# Sample a subset of distances
sample_size = 10000
sample_idx = np.random.choice(len(X), sample_size, replace=False)
X_sample = X[sample_idx]

# Compute pairwise distances on small sample
dists = pairwise_distances(X_sample, metric='euclidean')
print(f"Distance stats (sample of {sample_size} points):")
print(f"  Min: {dists[dists > 0].min():.6f}")
print(f"  25th percentile: {np.percentile(dists[dists > 0], 25):.6f}")
print(f"  Median: {np.median(dists[dists > 0]):.6f}")
print(f"  75th percentile: {np.percentile(dists[dists > 0], 75):.6f}")
print(f"  Max: {dists.max():.6f}")