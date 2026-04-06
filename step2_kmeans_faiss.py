"""
Step 2 (Alternative) – FAISS Spherical k-Means Balancing
=========================================================
Clusters the feature embeddings produced by Step 1 into 10,000 groups
using FAISS spherical k-means. Then performs proportional sampling,
selecting the closest 20% of tiles from *each* cluster to form the 
final dataset (which results in a 20% overall subset of the data).

This gracefully handles L2 normalized PCA reduced datasets, ensuring
we capture representative images from all semantic clusters.

Algorithm
---------
1.  Load embeddings.npz  (N_all, D)
2.  PCA → reduce dimensions (e.g., to 280).
3.  L2 Normalization (projects to spherical surface).
4.  FAISS Spherical k-Means → assign each tile to one of 10,000 clusters.
5.  For each cluster k:
        candidates_k = all point indices in cluster k
        n_take = 20% of len(candidates_k)
        Sample closest n_take points to the cluster's centroid
6.  Concatenate and save manifest + stats.
"""

import argparse
import logging
import os
import time
from typing import List

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

try:
    import faiss
except ImportError:
    raise ImportError("FAISS is not installed. Please install via: conda install -c pytorch faiss-cpu or faiss-gpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _run_pca(X: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    """Fit PCA on X and return the reduced array."""
    log.info("Running PCA: %d → %d dims …", X.shape[1], n_components)
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized")
    X_r = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    log.info("PCA explained variance: %.1f%%", explained * 100)
    return X_r


def run(cfg: dict) -> str:
    """Run FAISS spherical k-means and proportional sampling."""
    data_cfg = cfg["data"]
    km_cfg   = cfg.get("kmeans", {})
    output_dir = data_cfg["output_dir"]
    
    embeddings_fname = cfg["feature_extraction"].get("embeddings_file", "embeddings.npz")
    embeddings_path  = os.path.join(output_dir, embeddings_fname)
    
    # Custom output names for the FAISS runs
    subset_path = os.path.join(output_dir, "subset_manifest_faiss.parquet")
    stats_path  = os.path.join(output_dir, "faiss_kmeans_stats.csv")
    centroids_path = os.path.join(output_dir, "centroids_faiss.npz")

    # Hardcoded or config-driven values for this specific approach
    n_clusters = 450000
    # pca_components = int(km_cfg.get("pca_components", 280))
    seed = int(km_cfg.get("seed", 42))

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load embeddings + manifest
    # ------------------------------------------------------------------
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    log.info("Loading embeddings from %s …", embeddings_path)
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    
    base_dir = "/data/databases/Core-S2L2A"
    meta_path = os.path.join(base_dir, "metadata.parquet")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing root metadata file: {meta_path}")

    log.info("Loading full manifest from %s …", meta_path)
    manifest = pd.read_parquet(meta_path, columns=["grid_cell", "parquet_url", "parquet_row"])
    
    def _to_local_path(url: str) -> str:
        fname = url.split("/")[-1]
        return os.path.join(base_dir, "images", fname)

    manifest["parquet_file"] = manifest["parquet_url"].apply(_to_local_path)
    manifest["row_group"]    = manifest["parquet_row"].astype(int)
    manifest["row_in_rg"]    = 0
    
    if "indices" in data:
        log.info("Filtering manifest using valid indices from embeddings...")
        valid_indices = data["indices"]
        manifest = manifest.iloc[valid_indices].copy()
    
    # Optional subsetting for testing
    subset_size = cfg.get("test_subset_size", None)
    if subset_size is not None and subset_size < len(embeddings):
        log.warning("TESTING MODE: Truncating data to %d samples", subset_size)
        embeddings = embeddings[:subset_size]
        manifest = manifest.iloc[:subset_size].copy()

    n_full = len(manifest)
    n_sample = len(embeddings)
    log.info("Total tiles loaded: %d", n_sample)

    if n_sample != n_full:
        raise ValueError("Embeddings and manifest row counts differ. Mismatch!")

    # ------------------------------------------------------------------
    # 2. Extract full embeddings & L2 Normalize
    # ------------------------------------------------------------------
    X = embeddings.copy()
    
    log.info("L2-normalizing full %d-dimensional vectors before regular K-Means (no PCA) …", X.shape[1])
    X = normalize(X, norm="l2")

    # ------------------------------------------------------------------
    # 3. FAISS Regular K-Means
    # ------------------------------------------------------------------
    d = X.shape[1]
    log.info("Training FAISS Regular K-Means (K=%d, d=%d) …", n_clusters, d)
    
    # Regular K-Means uses Euclidean distance automatically
    kmeans = faiss.Kmeans(
        d=d, 
        k=n_clusters, 
        niter=6, 
        verbose=True, 
        spherical=False, 
        seed=seed,
        gpu=True,  # Enables GPU acceleration
        max_points_per_centroid=100,  # Limits how many points FAISS thinks it needs per cluster for training
        min_points_per_centroid=1     # Tells FAISS it's perfectly fine to have just 1-5 points per cluster
    )
    
    t0 = time.time()
    kmeans.train(X)
    t1 = time.time()
    log.info("FAISS training completed in %.1f seconds", t1 - t0)

    # ------------------------------------------------------------------
    # 4. Assign points to clusters and evaluate distances
    # ------------------------------------------------------------------
    log.info("Assigning points to clusters and computing distances …")
    # For regular K-Means, use L2 distance (IndexFlatL2)
    index = faiss.IndexFlatL2(d)
    index.add(kmeans.centroids)
    
    # D: distances (lower is closer), I: cluster indices
    distances, labels = index.search(X, 1)
    
    distances = distances.flatten()
    labels = labels.flatten()

    # ------------------------------------------------------------------
    # 5. Selecting 1 central tile per cluster
    # ------------------------------------------------------------------
    log.info("Sampling the 1 closest tile to the centroid for each of the %d clusters …", n_clusters)
    
    chosen_idx = []
    stats_rows = []

    # Bincount to find sizes of all clusters
    counts = np.bincount(labels, minlength=n_clusters)

    for k in range(n_clusters):
        pts = np.where(labels == k)[0]
        n_cands = len(pts)
        
        # We only want to take ONE item per sub-cluster (the centroid representation)
        n_take = 1 if n_cands > 0 else 0
        
        if n_take > 0:
            # Sort points by distance (ascending order, lowest distance first)
            cluster_dists = distances[pts]
            closest_indices = np.argsort(cluster_dists)[:n_take]
            
            sampled_pts = pts[closest_indices]
            chosen_idx.append(sampled_pts)
        
        stats_rows.append({
            "cluster_id": k,
            "n_candidates": n_cands,
            "n_sampled": n_take,
        })

    all_chosen_idx = np.concatenate(chosen_idx) if chosen_idx else np.array([], dtype=int)
    log.info("Sampling complete! Selected %d tiles (%.2f%% of total)", 
             len(all_chosen_idx), 100 * len(all_chosen_idx) / n_sample)

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    np.savez_compressed(centroids_path, centroids=kmeans.centroids, counts=counts)
    log.info("Centroids saved to %s", centroids_path)

    # Build manifest
    subset_manifest_final = manifest.iloc[all_chosen_idx].copy()
    subset_manifest_final["cluster_id"] = labels[all_chosen_idx]

    if "parquet_url" in subset_manifest_final.columns:
        subset_manifest_final = subset_manifest_final.drop(columns=["parquet_url", "parquet_row"])

    subset_manifest_final = subset_manifest_final.reset_index(drop=True)
    subset_manifest_final.to_parquet(subset_path, index=False)
    log.info("Subset manifest saved to %s", subset_path)

    # Stats
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(stats_path, index=False)
    log.info("Cluster stats saved to %s", stats_path)

    # TB calculation
    total_bytes = len(subset_manifest_final) * 1068 * 1068 * 12 * 2
    log.info("Subset summary: %d tiles ≈ %.1f TB (uint16, 12 bands)", 
             len(subset_manifest_final), total_bytes / 1e12)

    return subset_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
        
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
        
    run(cfg)
