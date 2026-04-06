import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import io
import os
import matplotlib.pyplot as plt
from PIL import Image
import faiss
from sklearn.preprocessing import normalize

def bytes_to_array(b):
    if b is None:
        return np.zeros((256, 256))
    if isinstance(b, dict):
        b = b['bytes']
    return np.array(Image.open(io.BytesIO(b))).astype(float)

def main():
    print("Loading embeddings...")
    data = np.load("/data/databases/MajorTom5T/outputs/embeddings.npz", allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)

    print("Loading metadata...")
    base_dir = "/data/databases/Core-S2L2A"
    manifest = pd.read_parquet(os.path.join(base_dir, "metadata.parquet"), columns=["grid_cell", "parquet_url", "parquet_row"])

    print("Identifying pure black representation (Index 1231300)...")
    black_idx = 1231300
    black_vector = embeddings[black_idx]
    black_norm = normalize(black_vector.reshape(1, -1), norm="l2")

    print("Loading centroids...")
    centroids_data = np.load("/data/databases/MajorTom5T/outputs/centroids_faiss.npz")
    centroids = centroids_data["centroids"].astype(np.float32)

    print("Identifying garbage clusters (>0.90 similarity to pure black)...")
    sims = np.dot(centroids, black_norm.T).flatten()
    garbage_clusters = set(np.where(sims > 0.90)[0])
    print(f"Found {len(garbage_clusters)} garbage clusters.")

    # Sample indices randomly to make execution fast
    np.random.seed(42)
    sample_size = 5000
    print(f"Sampling {sample_size} random images for the plot...")
    sample_indices = np.random.choice(len(manifest), sample_size, replace=False)

    print("Assigning clusters to sampled images...")
    X_sample = normalize(embeddings[sample_indices], norm="l2")
    index_cpu = faiss.IndexFlatL2(centroids.shape[1])
    index_cpu.add(centroids)
    _, labels = index_cpu.search(X_sample, 1)
    labels = labels.flatten()

    good_pcts = []
    garbage_pcts = []

    print("Reading parquet files to calculate % black pixels per image...")
    
    # Group by parquet file to optimize I/O
    df_samples = pd.DataFrame({
        'idx': sample_indices,
        'cluster': labels,
        'url': manifest.iloc[sample_indices]['parquet_url'].values,
        'row_group': manifest.iloc[sample_indices]['parquet_row'].values
    })
    df_samples['parquet_file'] = df_samples['url'].apply(lambda x: os.path.join(base_dir, "images", x.split("/")[-1]))

    grouped = df_samples.groupby('parquet_file')
    
    processed = 0
    for parquet_file, group in grouped:
        try:
            dataset = pq.ParquetFile(parquet_file)
            for _, s_row in group.iterrows():
                rg = int(s_row['row_group'])
                table = dataset.read_row_group(rg, columns=['B04', 'B03', 'B02'])
                
                R = bytes_to_array(table.column('B04')[0].as_py())
                G = bytes_to_array(table.column('B03')[0].as_py())
                B = bytes_to_array(table.column('B02')[0].as_py())
                
                # Compute percentage of pixels that are exactly zero across RGB
                rgb = np.stack([R, G, B], axis=-1)
                zeros_pct = (rgb == 0).mean() * 100
                
                if s_row['cluster'] in garbage_clusters:
                    garbage_pcts.append(zeros_pct)
                else:
                    good_pcts.append(zeros_pct)
                
                processed += 1
                if processed % 500 == 0:
                    print(f"Processed {processed}/{sample_size} images...")
        except Exception as e:
            pass

    print(f"\nFinal tally: {len(good_pcts)} in Good Clusters, {len(garbage_pcts)} in Garbage Clusters.")
    
    print("Plotting results...")
    plt.figure(figsize=(12, 6))
    
    # Try both overlapping histograms
    plt.hist(good_pcts, bins=50, alpha=0.5, label='Good Clusters', color='green', edgecolor='black')
    plt.hist(garbage_pcts, bins=50, alpha=0.5, label='Garbage Clusters', color='red', edgecolor='black')
    
    # Enhance plot readability
    plt.yscale('log')
    plt.xlabel('Percentage of Black Pixels (%)')
    plt.ylabel('Number of Images (Log Scale)')
    plt.title(f'Distribution of Missing Data in Good vs Garbage Clusters\n(Sample size: {processed} images)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = '/home/carmenoliver/my_projects/Make_MajorTom_sbst/garbage_vs_good_clusters.png'
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved plot successfully to: {out_path}")

if __name__ == '__main__':
    main()