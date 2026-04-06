import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Find clusters with more than a specified number of points.")
    parser.add_argument("--stats_csv", type=str, default="/data/databases/MajorTom5T/outputs/faiss_kmeans_stats.csv",
                        help="Path to the faiss_kmeans_stats.csv file")
    parser.add_argument("--min_points", type=int, default=15,
                        help="Minimum number of points in the cluster")
    args = parser.parse_args()

    if not os.path.exists(args.stats_csv):
        print(f"Error: Could not find {args.stats_csv}")
        return

    # Read the CSV
    df = pd.read_csv(args.stats_csv)

    # ==========================================
    # PART 1.5: Print Table of Cluster Tiers
    # ==========================================
    bins = [-1, 0, 9, 99, 999, 9999, 99999, float('inf')]
    labels = ['0', '1-9', '10-99', '100-999', '1,000-9,999', '10,000-99,999', '100,000+']
    df['bucket'] = pd.cut(df['n_candidates'], bins=bins, labels=labels)
    
    # We must properly aggregate the categories in the exact order we want
    counts = df['bucket'].value_counts()
    
    print("\n" + "="*45)
    print(f"{'Images in Cluster':<25} | {'Number of Clusters':<15}")
    print("-" * 45)
    for label in labels:
        count = counts.get(label, 0)
        print(f"{label:<25} | {count:<15,}")
    print("="*45 + "\n")

    # ==========================================
    # PART 2: Visualizations
    # ==========================================
    print("\nGenerating cluster size distribution plots...")
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # To avoid zero-skewing, we look only at clusters that have at least 1 image
    # (Though if you want to include zeros, just remove this filter)
    non_empty_clusters = df[df['n_candidates'] > 0]['n_candidates']
    sizes = non_empty_clusters.sort_values(ascending=False).values
    
    # Plot 1: Sorted sizes
    ax1.plot(sizes, color='blue', linewidth=2)
    ax1.set_xlabel("Number of Clusters (Ranked by Size)")
    ax1.set_ylabel("Number of Images in Cluster (Log Scale)")
    ax1.set_yscale('log') # Log scale fixes the L-shape issue for highly skewed data
    ax1.set_title("Cluster Size Curve (Log Scale)")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Histogram
    # We use a log scale on the y-axis here too, otherwise the vast number of tiny clusters hides the rest
    ax2.hist(non_empty_clusters, bins=50, color='orange', alpha=0.7, log=True)
    ax2.set_xlabel("Number of Images in Cluster")
    ax2.set_ylabel("Frequency (Log Scale)")
    ax2.set_title("Histogram of Non-Empty Cluster Sizes")
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_file = os.path.join(os.path.dirname(__file__), "cluster_distribution_plot.png")
    plt.savefig(plot_file, dpi=150)
    print(f"✅ Saved cluster size visualization to {plot_file}")

if __name__ == "__main__":
    main()
