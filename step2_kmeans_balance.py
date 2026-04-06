"""
Step 2 – k-Means Balancing
===========================
Clusters the feature embeddings produced by Step 1 into ``n_clusters`` groups,
then performs *balanced* sampling across clusters so that the final subset
contains ``target_n_tiles`` tiles with equal representation from every
semantic cluster (deserts, oceans, cities, agriculture, etc.).

Because Step 1 embeds **all** tiles (sample_fraction=1.0), every tile in the
manifest has an exact cluster assignment — no extrapolation is needed.

Algorithm
---------
1.  Load embeddings.npz  (N_all, D)  where N_all == len(full manifest)
2.  (Optional) PCA → reduce to ``pca_components`` dimensions.
3.  k-Means → assign each tile to one of ``n_clusters``.
4.  For each cluster k:
        candidates_k = all tiles whose embedding falls in cluster k.
        sample_k     = randomly draw ``tiles_per_cluster`` from candidates_k.
5.  Concatenate all samples → final subset manifest.
6.  Save to ``<output_dir>/subset_manifest.parquet``.

Output
------
``<output_dir>/subset_manifest.parquet``
  Same columns as the cloud-filtered manifest, plus ``cluster_id``.

``<output_dir>/kmeans_stats.csv``
  Per-cluster statistics (size, sampled, centroid norm, …).
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def _run_pca(X: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    """Fit PCA on X and return the reduced array."""
    log.info("Running PCA: %d → %d dims …", X.shape[1], n_components)
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized")
    X_r = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    log.info("PCA explained variance: %.1f%%", explained * 100)
    return X_r


def _run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    max_iter: int,
    seed: int,
) -> MiniBatchKMeans:
    """Fit MiniBatchKMeans on X and return the fitted estimator."""
    log.info(
        "Fitting MiniBatchKMeans: %d clusters on %d samples …",
        n_clusters,
        len(X),
    )
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        batch_size=4096,
        random_state=seed,
        n_init=1,
        verbose=0,
        compute_labels=True,
    )
    km.fit(X)
    inertia = km.inertia_
    log.info("k-Means done. Inertia=%.3e", inertia)
    return km


# ---------------------------------------------------------------------------
def _run_birch(
    X: np.ndarray,
    target_clusters: int,
    threshold: float,
    min_threshold: float,
    decay: float,
    max_attempts: int,
    seed: int,
) -> Tuple[Birch, int, float]:
    """Fit Birch, reducing the threshold until we hit the requested bucket count."""
    log.info(
        "Starting Birch with threshold %.5f to reach %d clusters on %d samples …",
        threshold,
        target_clusters,
        len(X),
    )
    current_threshold = threshold
    attempts = 0
    while True:
        attempts += 1
        br = Birch(
            n_clusters=None,
            threshold=current_threshold,
            branching_factor=50,
            copy=True,
        )
        br.fit(X)
        # Don't use n_subclusters_ — it's unreliable until after predict()
        # Instead, call predict() to get the actual cluster assignments
        labels = br.predict(X)
        subclusters = int(labels.max() + 1)
        log.info(
            "Birch pass %d: threshold %.5f produced %d subclusters",
            attempts,
            current_threshold,
            subclusters,
        )

        if subclusters >= target_clusters or current_threshold <= min_threshold or attempts >= max_attempts:
            break

        current_threshold = max(current_threshold * decay, min_threshold)
        log.info(
            "Reducing Birch threshold to %.5f and retrying (attempt %d/%d)",
            current_threshold,
            attempts + 1,
            max_attempts,
        )

    return br, subclusters, current_threshold


def _run_birch_threshold_only(
    X: np.ndarray,
    threshold: float,
) -> Birch:
    """Fit Birch with a fixed threshold, returning the fitted estimator.
    
    Note: We don't return n_subclusters here because the attribute isn't reliable
    until after predict() is called. The caller should compute it from predict() output.
    """
    log.info(
        "Fitting Birch with threshold %.5f (threshold-only mode, no n_clusters target) on %d samples …",
        threshold,
        len(X),
    )
    br = Birch(
        n_clusters=None,
        threshold=threshold,
        branching_factor=50,
        copy=False,
    )
    br.fit(X)
    return br


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: dict) -> str:
    """Run k-means balancing and return the path to the subset manifest."""
    data_cfg  = cfg["data"]
    km_cfg    = cfg["kmeans"]

    output_dir       = data_cfg["output_dir"]
    manifest_fname   = cfg.get("cloud_filter", {}).get("manifest_file", "cloud_filtered_manifest.parquet")
    embeddings_fname = cfg["feature_extraction"].get("embeddings_file", "embeddings.npz")
    subset_fname     = km_cfg.get("subset_manifest_file", "subset_manifest.parquet")

    manifest_path    = os.path.join(output_dir, manifest_fname)
    embeddings_path  = os.path.join(output_dir, embeddings_fname)
    subset_path      = os.path.join(output_dir, subset_fname)
    stats_path       = os.path.join(output_dir, "kmeans_stats.csv")

    n_clusters      = int(km_cfg.get("n_clusters", 1000))
    max_iter        = int(km_cfg.get("max_iter", 300))
    pca_components  = int(km_cfg.get("pca_components", 64))
    seed            = int(km_cfg.get("seed", 42))
    target_n_tiles  = int(km_cfg.get("target_n_tiles", 530_000))
    clustering_method = km_cfg.get("clustering_method", "kmeans")
    birch_threshold = float(km_cfg.get("birch_threshold", 0.25))
    birch_min_threshold = float(km_cfg.get("birch_min_threshold", 1e-6))
    birch_decay = float(km_cfg.get("birch_threshold_decay", 0.5))
    birch_attempts = int(km_cfg.get("birch_threshold_attempts", 8))

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load embeddings + manifest
    # ------------------------------------------------------------------
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"Embeddings not found: {embeddings_path}\nRun step2_extract_features.py first."
        )

    log.info("Loading embeddings from %s …", embeddings_path)
    data       = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)   # (N_sample, D)
    indices    = data["indices"].astype(np.int64)         # (N_sample,) → full manifest rows
    
    # Use the same manifest path logic as step 1
    base_dir = "/data/databases/Core-S2L2A"
    meta_path = os.path.join(base_dir, "metadata.parquet")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing root metadata file: {meta_path}")

    log.info("Loading full manifest from %s …", meta_path)
    manifest = pd.read_parquet(meta_path, columns=["grid_cell", "parquet_url", "parquet_row"])
    
    # Reconstruct columns to match the imagery structure expected by any downstream code
    def _to_local_path(url: str) -> str:
        fname = url.split("/")[-1]
        return os.path.join(base_dir, "images", fname)

    manifest["parquet_file"] = manifest["parquet_url"].apply(_to_local_path)
    manifest["row_group"]    = manifest["parquet_row"].astype(int)
    manifest["row_in_rg"]    = 0
    
    log.info("Full manifest: %d tiles.", len(manifest))

    n_full   = len(manifest)
    n_sample = len(embeddings)

    if n_sample != n_full:
        raise ValueError(
            f"Embeddings ({n_sample}) and manifest ({n_full}) row counts differ. "
            "Re-run step1_extract_features.py with sample_fraction=1.0 to embed all tiles."
        )

    # ------------------------------------------------------------------
    # 2. (Optional) PCA
    # ------------------------------------------------------------------
    X = embeddings.copy()
    if pca_components > 0 and pca_components < X.shape[1]:
        X = _run_pca(X, pca_components, seed)

    # L2-normalise before k-means (cosine similarity via Euclidean on unit sphere)
    X = normalize(X, norm="l2")

    # ------------------------------------------------------------------
    # 3. Clustering on all tiles (kmeans or hierarchical/Birch)
    # ------------------------------------------------------------------
    method = clustering_method.lower()
    is_threshold_only = (method == "birch-threshold-only")

    if method in ("kmeans", "mini-batch-kmeans"):
        km = _run_kmeans(X, n_clusters=n_clusters, max_iter=max_iter, seed=seed)
        full_labels = km.labels_.astype(np.int32)
        n_clusters_actual = n_clusters
        centroids = km.cluster_centers_

    elif is_threshold_only:
        log.info("Running Birch in threshold-only mode with threshold %.5f …", birch_threshold)
        br = _run_birch_threshold_only(X, threshold=birch_threshold)
        full_labels = br.predict(X).astype(np.int32)
        n_clusters_actual = int(full_labels.max() + 1)
        if n_clusters_actual == 0:
            raise ValueError("Birch returned zero clusters")
        log.info("Birch (threshold-only) produced %d subclusters naturally", n_clusters_actual)
        centroids = None  # Computed below

    elif method in ("hierarchical", "birch"):
        log.info("Running hierarchical clustering (Birch) to obtain %d clusters …", target_n_tiles)
        br, _, final_threshold = _run_birch(
            X,
            target_clusters=target_n_tiles,
            threshold=birch_threshold,
            min_threshold=birch_min_threshold,
            decay=birch_decay,
            max_attempts=birch_attempts,
            seed=seed,
        )
        full_labels = br.predict(X).astype(np.int32)
        n_clusters_actual = int(full_labels.max() + 1)
        if n_clusters_actual == 0:
            raise ValueError("Birch returned zero clusters")
        log.info(
            "Birch produced %d subclusters (requested %d) at threshold %.6f",
            n_clusters_actual, target_n_tiles, final_threshold,
        )
        centroids = None  # Computed below

    else:
        raise ValueError(f"Unknown clustering_method: {clustering_method}")

    # Compute centroids and cluster sizes for Birch/hierarchical methods
    counts = np.bincount(full_labels, minlength=n_clusters_actual)
    if centroids is None:
        centroids = np.zeros((n_clusters_actual, X.shape[1]), dtype=np.float32)
        for i in range(n_clusters_actual):
            idx = np.where(full_labels == i)[0]
            if len(idx) > 0:
                centroids[i] = X[idx].mean(axis=0)

    # Save centroid embeddings
    centroids_path = os.path.join(output_dir, "centroids.npz")
    np.savez_compressed(centroids_path, centroids=centroids, counts=counts)
    log.info("Centroids saved to %s", centroids_path)

    # ------------------------------------------------------------------
    # 4. Balanced sampling
    # ------------------------------------------------------------------
    chosen: List[np.ndarray] = []
    stats_rows: List[dict] = []

    if is_threshold_only:
        log.info("Threshold-only mode: using centroids only (no additional tile selection)")
        # In threshold-only mode, we just represent the centroids directly
        all_chosen_idx = np.arange(n_clusters_actual, dtype=np.int64)
        full_labels = np.arange(n_clusters_actual, dtype=np.int32)
        
        for i in range(n_clusters_actual):
            stats_rows.append({
                "cluster_id": i,
                "n_candidates": counts[i],
                "n_sampled": 1,
                "cluster_rank": i,
            })
            
    else:
        tiles_per_cluster = max(1, target_n_tiles // n_clusters_actual)
        remainder = target_n_tiles % n_clusters_actual

        log.info(
            "Balanced sampling: %d tiles/cluster (+ %d remainder) → target %d tiles total",
            tiles_per_cluster, remainder, target_n_tiles,
        )

        # Sort clusters by size descending to cleanly distribute the remainder
        sorted_clusters = np.argsort(-counts)
        for rank, k in enumerate(sorted_clusters):
            candidates = np.where(full_labels == k)[0]
            n_cands = len(candidates)
            n_take = tiles_per_cluster + (1 if rank < remainder else 0)
            n_take = min(n_take, n_cands)

            if n_cands == 0:
                sampled_idx = np.array([], dtype=np.int64)
            elif n_take >= n_cands:
                sampled_idx = candidates
            else:
                centroid = centroids[k]
                cand_X = X[candidates]
                dists = ((cand_X - centroid) ** 2).sum(axis=1)
                sampled_idx = candidates[np.argsort(dists)[:n_take]]

            chosen.append(sampled_idx)
            stats_rows.append({
                "cluster_id": k,
                "n_candidates": n_cands,
                "n_sampled": len(sampled_idx),
                "cluster_rank": rank,
            })

        all_chosen_idx = np.concatenate(chosen)
        log.info("Total tiles in subset: %d", len(all_chosen_idx))

    # ------------------------------------------------------------------
    # 5. Build and save the subset manifest
    # ------------------------------------------------------------------
    if is_threshold_only:
        # Threshold-only mode: just save the centroid info
        subset_manifest_final = pd.DataFrame({"cluster_id": np.arange(n_clusters_actual)})
        log.info("Threshold-only mode: subset manifest contains %d centroids (not tile references)", len(subset_manifest_final))
    else:
        # k-means and hierarchical modes: select actual tiles from the manifest
        subset_manifest_final = manifest.iloc[all_chosen_idx].copy()
        subset_manifest_final["cluster_id"] = full_labels[all_chosen_idx]

        # Drop intermediate mapping columns before saving
        if "parquet_url" in subset_manifest_final.columns:
            subset_manifest_final = subset_manifest_final.drop(columns=["parquet_url", "parquet_row"])

        subset_manifest_final = subset_manifest_final.reset_index(drop=True)

    subset_manifest_final.to_parquet(subset_path, index=False)
    log.info("Subset manifest saved to %s", subset_path)

    # Save per-cluster stats
    stats_df = pd.DataFrame(stats_rows).sort_values("cluster_id")
    stats_df.to_csv(stats_path, index=False)
    log.info("Cluster stats saved to %s", stats_path)

    # Summary – size in TB (uint16 = 2 bytes per value)
    if is_threshold_only:
        log.info(
            "Subset summary (threshold-only): %d centroids (embeddings in feature space, not tiles)",
            len(subset_manifest_final),
        )
    else:
        total_bytes = len(subset_manifest_final) * 1068 * 1068 * 12 * 2
        log.info(
            "Subset summary: %d tiles  ≈ %.1f TB (uint16, 12 bands)",
            len(subset_manifest_final),
            total_bytes / 1e12,
        )
    return subset_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 2: k-Means balanced sampling for Core-S2L2A subset selection."
    )
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    run(cfg)
