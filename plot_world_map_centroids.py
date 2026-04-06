import pandas as pd
import matplotlib.pyplot as plt
import os

print("Loading subset manifest (cluster representatives)...")
subset = pd.read_parquet("/data/databases/MajorTom5T/outputs/subset_manifest_faiss.parquet")

print("Loading full metadata for lat/lon...")
meta = pd.read_parquet("/data/databases/Core-S2L2A/metadata.parquet", columns=["grid_cell", "centre_lat", "centre_lon"])

# Merge to get lat/lon for the chosen subset images
print("Merging lat/lon data...")
# We can just drop duplicates on grid_cell to make merge faster, or just merge directly
meta_unique = meta.drop_duplicates(subset=['grid_cell'])
merged = subset.merge(meta_unique, on="grid_cell", how="left")

print("Plotting world map...")
plt.figure(figsize=(16, 8))

# Basic styling
plt.style.use('dark_background')
plt.scatter(
    merged['centre_lon'], 
    merged['centre_lat'], 
    s=0.5, # Point size
    c='cyan', 
    alpha=0.3,
    marker='.'
)

plt.title(f"Global Distribution of {len(merged)} Cluster Representative Images (Centroids)", fontsize=18)
plt.xlabel("Longitude", fontsize=14)
plt.ylabel("Latitude", fontsize=14)
plt.grid(True, alpha=0.1)

# Quick background map roughly outlining continents by just keeping standard axes limits
plt.xlim(-180, 180)
plt.ylim(-90, 90)

output_file = "/home/carmenoliver/my_projects/Make_MajorTom_sbst/centroid_world_map.png"
plt.savefig(output_file, bbox_inches='tight', dpi=300)
print(f"Saved world map to {output_file}")
