# Make_MajorTom_sbst – Core-S2L2A Subset Selection Pipeline

Creates a semantically balanced **~5 TB** subset of the
[Major-TOM Core-S2L2A](https://huggingface.co/datasets/Major-TOM/Core-S2L2A)
dataset (full dataset: ~23 TB parquet /2,245,886 tiles).

---

 <img width="2896" height="1574" alt="image" src="https://github.com/user-attachments/assets/75a6b4a2-786a-4fe1-b35a-019554c21021" />


## Pipeline overview

```
/data/databases/Core-S2L2A/images/*.parquet   (~4 492 files, ~2.24 M tiles)
         ▼ Step 1 – Feature Extraction
         |  Filter all tiles to keep only data with <45% zeros, divide by 10000 and clamp to [0,1] then       normalize with their band statistics to follow what DINOv3 expects. 
         |. Make a new metadata file to record the new filtered index
         │  Embed ALL tiles with DINOv3-ViT-L/16 (RGB, bicubic downscaling to 224×224).
         │  No sub-sampling — every tile gets an exact semantic fingerprint.
         │  Save CLS-token embeddings (dim = 1024).
         │  ~2.23 M tiles @ ~80 img/s on 1 GPU ≈ 15–16 h (ViT-L).
         │  ▶  Core-S2L2a-subset/embeddings.npz
         │  ▶. Core-S2L2a-subset/full_dataset_manifest.parquet
         ▼ Step 2 – FAISS k-Means 
            k-Means: Same number of clusters as images we want to keep: 450000. Number decided due to target subset size: 4.3 TB (23TB / 2.24M tiles)
            ▶  Core-S2L2a-subset/manifest.parquet
            ▶  Core-S2L2a-subset/kmeans_faiss_stats.csv

```

---

## File structure

```
Make_MajorTom_sbst/
├── config.yaml                 ← all knobs / paths in one place
├── run_pipeline.py             ← master orchestrator
├── step1_extract_features.py   ← Step 1: DINOv3-ViT-L/16 feature extraction
├── step2_kmeans_faiss.py       ← Step 2: FAISS k-Means clustering and sampling
├── sbatch_subset.sh            ← SLURM launcher
├── inspect_cluster.py          ← Peek at images inside specific semantic clusters
├── find_large_clusters.py      ← Compute size distributions / histograms of clusters
└── /data/databases/MajorTom5T/Core-S2L2A-subset/  ← output directory (example)
    ├── embeddings_filtered.npz
    ├── full_dataset_manifest.parquet
    ├── subset_manifest_faiss.parquet
    ├── centroids_faiss.npz
    └── faiss_kmeans_stats.csv
```

---

## Quick start

```bash
conda activate envr #(create your own)
cd /home/carmenoliver/my_projects/Make_MajorTom_sbst

# Full pipeline
python run_pipeline.py --config config.yaml

# Or via SLURM
sbatch sbatch_subset.sh
```

---

## Configuration (`config.yaml`)

| Key | Value | Description |
|-----|-------|-------------|
| `feature_extraction.nodata_threshold`| `0.45` | Drops tiles with > 45% zero/nodata values. |
| `feature_extraction.sample_fraction` | `1.0` | Embed **all** valid tiles (no sub-sampling). |
| `feature_extraction.model_arch` | `vitl16` | ViT-L/16 (1024-D, best semantic quality). |
| `feature_extraction.weights_path` | local `.pth` | Pre-trained DINOv3 weights. |
| `feature_extraction.batch_size` | `64` | Inference batch size. |
| `kmeans.clustering_method` | `faiss` | Scalable k-Means using FAISS on GPU/CPU. |
| `kmeans.n_clusters` | `450000` | Number of semantic clusters (equal to target subset size). |
| `kmeans.max_iter` | `300` | Maximum FAISS k-Means iterations. |

---

## Output: `subset_manifest_faiss.parquet`

Each row identifies a tile to include in the training set. Since we want 450,000 images and request 450,000 clusters, the script extracts one representative tile (usually closest to the centroid) per cluster.

| Column | Description |
|--------|-------------|
| `grid_cell` | Major-TOM grid cell |
| `parquet_url` / `parquet_file`| Path to the source parquet shard |
| `parquet_row` | Absolute row index within the source parquet |
| `row_group` | Row-group index within the parquet file |
| `row_in_rg` | Row index within the row group |
| `cluster_id` | FAISS k-Means cluster assignment (0 to 449,999) |
| `distance_to_centroid` | L2 distance from this tile's embedding to its cluster centroid |

---

## Tooling scripts

### Checking Cluster Distribution
```bash
python find_large_clusters.py --stats_csv /data/databases/MajorTom5T/Core-S2L2A-subset/extras/faiss_kmeans_stats.csv
```
Generates a histogram and line plot (`cluster_distribution_plot.png`) of how many images ended up in each cluster.

### Inspecting Specific Clusters
```bash
python inspect_cluster.py
```
Edit `TARGET_CLUSTERS` inside this script to pull the actual image patches matching the exact semantic clusters. It normalizes embeddings and uses FAISS to find the closest parquet items, then reconstructs the RGB images into a PNG grid.

## Size maths

```
23TB -- 2.24M img
4.3TB -- 450,000 img
```
