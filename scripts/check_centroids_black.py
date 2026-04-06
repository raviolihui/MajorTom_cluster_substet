import numpy as np
import os

centroids_data = np.load("/data/databases/MajorTom5T/outputs/centroids_faiss.npz")
print("Centroids keys:", centroids_data.files)
centroids = centroids_data["centroids"]
print(f"Num centroids: {len(centroids)}")
