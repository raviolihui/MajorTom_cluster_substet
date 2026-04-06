import numpy as np
import faiss
from sklearn.preprocessing import normalize
import os
import pyarrow.parquet as pq
import pandas as pd
from PIL import Image
import io

print("Loading embeddings...")
data = np.load("/data/databases/MajorTom5T/outputs/embeddings.npz", allow_pickle=True)
embeddings = data["embeddings"].astype(np.float32)

print("Loading metadata...")
base_dir = "/data/databases/Core-S2L2A"
manifest = pd.read_parquet(os.path.join(base_dir, "metadata.parquet"), columns=["grid_cell", "parquet_url", "parquet_row"])

def bytes_to_array(b):
    if b is None: return np.zeros((256, 256))
    if isinstance(b, dict): b = b['bytes']
    return np.array(Image.open(io.BytesIO(b))).astype(float)

print("Scanning for a completely black image...")
black_idx = None
# Shuffle a small chunk to find one quickly (1.3% of images are pure black)
np.random.seed(42)
test_indices = np.random.choice(len(manifest), 5000, replace=False)

for idx in test_indices:
    row = manifest.iloc[idx]
    parquet_file = os.path.join(base_dir, "images", row['parquet_url'].split("/")[-1])
    row_group = int(row['parquet_row'])
    try:
        dataset = pq.ParquetFile(parquet_file)
        table = dataset.read_row_group(row_group, columns=['B04', 'B03', 'B02'])
        R = bytes_to_array(table.column('B04')[0].as_py())
        G = bytes_to_array(table.column('B03')[0].as_py())
        B = bytes_to_array(table.column('B02')[0].as_py())
        if R.max() == 0 and G.max() == 0 and B.max() == 0:
            black_idx = idx
            print(f"Found purely black image at index {idx}!")
            break
    except Exception:
        continue

if black_idx is None:
    print("Could not find a pure black image!")
    exit(1)

black_vector = embeddings[black_idx]
print(f"Black vector norm: {np.linalg.norm(black_vector)}")

print("Loading centroids...")
X = normalize(embeddings, norm="l2")
centroids_data = np.load("/data/databases/MajorTom5T/outputs/centroids_faiss.npz")
centroids = centroids_data["centroids"].astype(np.float32)

print("Computing distances of centroids to the black vector...")
black_vector_norm = normalize(black_vector.reshape(1, -1), norm="l2")

# Inner product of centroids and black vector
similarities = np.dot(centroids, black_vector_norm.T).flatten()

thresholds = [0.999, 0.99, 0.95, 0.90]
for t in thresholds:
    count = np.sum(similarities > t)
    print(f"Centroids with similarity > {t} to pure black image: {count} out of {len(centroids)}")

# What index does the black image itself belong to?
index_cpu = faiss.IndexFlatL2(centroids.shape[1])
index_cpu.add(centroids)
D, I = index_cpu.search(black_vector_norm, 1)
print(f"The pure black image is assigned to cluster centroid {I[0][0]} with squared L2 distance {D[0][0]}")

