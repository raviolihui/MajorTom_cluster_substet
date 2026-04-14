import numpy as np

centroids_path = "/data/databases/MajorTom5T/Core-S2L2A-subset/extras/centroids_faiss.npz"
data = np.load(centroids_path)
centroids = data["centroids"].astype(np.float32)

# Normalize centroids
norms = np.linalg.norm(centroids, axis=1, keepdims=True)
centroids = centroids / (norms + 1e-8)

target_cluster = 13507
target_centroid = centroids[target_cluster]

# Compute cosine similarity (dot product of normalized vectors)
similarities = np.dot(centroids, target_centroid)

# Count how many have similarity > 0.9, 0.95, 0.99
print(f"Total clusters: {len(centroids)}")
print(f"Clusters with similarity > 0.8: {np.sum(similarities > 0.8)}")
print(f"Clusters with similarity > 0.9: {np.sum(similarities > 0.9)}")
print(f"Clusters with similarity > 0.95: {np.sum(similarities > 0.95)}")
print(f"Clusters with similarity > 0.99: {np.sum(similarities > 0.99)}")
