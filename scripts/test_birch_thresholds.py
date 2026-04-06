import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import Birch
import time

def find_birch_sweetspot():
    embeddings_path = "/data/databases/MajorTom5T/outputs/embeddings.npz"
    print(f"Loading embeddings from {embeddings_path}...")
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)

    # 1. PCA & Normalization
    print("Running PCA 1024 -> 280...")
    pca = PCA(n_components=280, random_state=42, svd_solver="randomized")
    X = pca.fit_transform(embeddings)
    
    # print("L2 Normalizing...")
    # X = normalize(X, norm="l2")

    # 2. Sample exactly 10,000 points
    sample_size = 10000
    print(f"\nSampling {sample_size} points to test thresholds...")
    np.random.seed(42)
    sample_idx = np.random.choice(len(X), sample_size, replace=False)
    X_test = X[sample_idx]

    # Target: 20% of the sample = 2,000 subclusters
    target_clusters = 2000

    # 3. Sweep Thresholds
    # We know 1.0 gives 1 cluster, and 0.1 gives 2.2M clusters
    thresholds_to_test = np.linspace(11, 14, 45) # Test 45 values between 0.1 and 1.0
    print(f"\nTesting {len(thresholds_to_test)} thresholds...")
    print(f"{'Threshold':<15} | {'Subclusters':<15} | {'Time (s)':<10}")
    print("-" * 45)

    for thresh in thresholds_to_test:
        t0 = time.time()
        # Note: branching_factor=50 is default
        br = Birch(n_clusters=None, threshold=thresh, branching_factor=50, copy=False)
        br.fit(X_test)
        
        # Predict to get accurate subcluster count
        labels = br.predict(X_test)
        n_clusters = int(labels.max() + 1)
        t_elapsed = time.time() - t0
        
        print(f"{thresh:<15.4f} | {n_clusters:<15} | {t_elapsed:<10.2f}")

        # If it collapses to 1, no point in going larger
        if n_clusters == 1:
            break

if __name__ == "__main__":
    find_birch_sweetspot()
