# Make_MajorTom_sbst – Core-S2L2A Subset Selection Pipeline

Creates a semantically balanced **~5 TB** subset of the
[Major-TOM Core-S2L2A](https://huggingface.co/datasets/Major-TOM/Core-S2L2A)
dataset (full dataset: ~23 TB parquet / 4 491 772 tiles / >2.56 T pixels).

---

## Pipeline overview

```
/data/databases/Core-S2L2A/images/*.parquet   (~4 492 files, ~4.49 M tiles)
         ▼ Step 1 – Feature Extraction
         │  Embed ALL tiles with DINOv3-ViT-L/16 (RGB, bicubic downscaling to 224×224).
         │  No sub-sampling — every tile gets an exact semantic fingerprint.
         │  Save CLS-token embeddings (dim = 1024).
         │  ~4.49 M tiles @ ~80 img/s on 1 GPU ≈ 15–16 h (ViT-L).
         │  ▶  outputs/embeddings.npz
         │
         ▼ Step 2 – k-Means Balancing
            PCA: 1024-D → 250-D (>90% explained variance).
            k-Means: Same number of clusters as images we want to keep: 978260. Number decided due to target subset size: 5 TB compressed (23TB / 4.49M tiles)
            ▶  outputs/subset_manifest.parquet
            ▶  outputs/kmeans_stats.csv
When `kmeans.clustering_method: birch` is set, Birch builds tuned leaf clusters and balanced sampling picks tiles per subcluster; the centroids are saved to `outputs/centroids.npz` so you can inspect or reuse the semantic anchors.
```

---

## File structure

```
Make_MajorTom_sbst/
├── config.yaml                 ← all knobs / paths in one place
├── run_pipeline.py             ← master orchestrator (run this)
├── step1_extract_features.py   ← Step 1: DINOv3-ViT-L/16 feature extraction
├── step2_kmeans_balance.py     ← Step 2: k-means balanced sampling
├── sbatch_subset.sh            ← SLURM launcher
└── outputs/                    ← created automatically
    ├── embeddings.npz
    ├── subset_manifest.parquet
    ├── kmeans_stats.csv
    └── centroids.npz           ← produced only when `kmeans.clustering_method: birch` (centroid embeddings + counts)
```

---

## Quick start

```bash
conda activate envr
cd /home/carmenoliver/my_projects/Make_MajorTom_sbst

# Full pipeline
python run_pipeline.py

# Or via SLURM
sbatch sbatch_subset.sh
```

---

## Configuration (`config.yaml`)

| Key | Value | Description |
|-----|-------|-------------|
| `feature_extraction.sample_fraction` | `1.0` | Embed **all** tiles (no sub-sampling) |
| `feature_extraction.model_arch` | `vitl16` | ViT-L/16 SAT (1024-D, best semantic quality) |
| `feature_extraction.weights_path` | local `.pth` | Pre-trained DINOv3 weights |
| `feature_extraction.batch_size` | `64` | Inference batch size |
| `kmeans.n_clusters` | `978260` | Number of semantic clusters |
| `kmeans.clustering_method` | `kmeans` | `kmeans` (default), `birch` (tuning threshold until target count), or `birch-threshold-only` (fixed threshold, natural cluster count) |
| `kmeans.birch_threshold` | `0.3` | Distance threshold fed to Birch. Lower values → smaller subcluster radius and more subclusters. Used by both `birch` and `birch-threshold-only` modes. |
| `kmeans.birch_threshold_decay` | `0.5` | Reduce factor applied when Birch returns too few subclusters. |
| `kmeans.birch_threshold_attempts` | `8` | Maximum retries before accepting the current threshold. |
| `kmeans.birch_min_threshold` | `1e-6` | Threshold floor; Birch will stop shrinking below this even if the target cluster count isn’t reached. |

## Clustering modes

### `kmeans` (default)
MiniBatchKMeans with fixed `n_clusters`. Fast, reliable, scales well.

### `birch` 
Birch with **adaptive threshold tuning** that shrinks the distance threshold until it reaches `target_n_tiles` clusters (or hits the attempt/threshold floor). Good for semantic hierarchies but can be slow if threshold tuning takes many passes.

### `birch-threshold-only` (recommended for centroid-only datasets)
Birch with a **fixed threshold** (`birch_threshold`), producing **centroids only** (no additional tile selection).

**Key difference:** Instead of clustering tiles and then sampling a balanced subset of tiles from each cluster, this mode:
1. Runs Birch with the fixed threshold
2. Computes the centroid embedding for each subcluster
3. **Uses only these centroids as the final dataset** (no tile selection)
4. Saves centroids to `outputs/centroids.npz` (centroid embeddings + cluster sizes)

This is useful when you want a small, semantically diverse dataset of **representative embeddings** rather than actual satellite imagery tiles. The output manifest will contain only cluster IDs (not tile references).

**Example:** Set `kmeans.clustering_method: birch-threshold-only` and `kmeans.birch_threshold: 0.3` to get ~1000 semantic centroids representing the full dataset's diversity.


Use `visualize_embeddings.py` to peek at the Step 1 CLS embeddings in 3D. The script runs a PCA to four dimensions, plots the first three axes in a 3D scatter, and colors points by their fourth component.

```sh
cd /home/carmenoliver/my_projects/Make_MajorTom_sbst
python visualize_embeddings.py --embeddings outputs/embeddings.npz --max-samples 50000 --output outputs/embeddings_pca3d.png
```

Adjust `--max-samples` and `--seed` for reproducibility; the script saves a PNG so you can share or open it locally.

---

## Output: `subset_manifest.parquet`

Each row identifies a tile to include in the training set:

| Column | Description |
|--------|-------------|
| `product_id` | Sentinel-2 product identifier |
| `grid_cell` | Major-TOM grid cell (e.g. `10S_DG`) |
| `parquet_file` | Absolute path to the source parquet shard |
| `row_group` | Row-group index within the parquet file |
| `row_in_rg` | Row index within the row group |
| `cloud_fraction` | Measured cloud fraction (≤ 0.10) |
| `cluster_id` | k-Means cluster assignment (0 – 978260) |

---

## Output: `centroids.npz` (Birch mode)

Generated only when `kmeans.clustering_method` is `birch`, this compressed NumPy archive contains the tuned centroids that we use during balanced sampling.

| Array | Shape | Description |
|-------|-------|-------------|
| `centroids` | `(n_clusters, D)` | Mean embedding of each Birch leaf cluster after threshold tuning. |
| `counts` | `(n_clusters,)` | Number of tiles assigned to each cluster. |

You can reuse the centroids for semantic analysis or as references for selecting representative tiles without needing the full subset manifest.

## Size maths

```
23TB -- 4.5M img
5T   -- 978260 img

```
