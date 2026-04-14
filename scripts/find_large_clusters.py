import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Find clusters with more than a specified number of points.")
    parser.add_argument("--stats_csv", type=str, default="/data/databases/MajorTom5T/outputs_filtered/faiss_kmeans_stats.csv",
                        help="Path to the faiss_kmeans_stats.csv file")
    parser.add_argument("--min_points", type=int, default=99,
                        help="Minimum number of points in the cluster")
    args = parser.parse_args()

    if not os.path.exists(args.stats_csv):
        print(f"Error: Could not find {args.stats_csv}")
        return

    # Read the CSV
    df = pd.read_csv(args.stats_csv)

    # Filter for clusters that have strictly more than min_points
    filtered_df = df[df['n_candidates'] > args.min_points]
    
    # Get the cluster IDs as a list
    cluster_ids = filtered_df['cluster_id'].tolist()

    print(f"Found {len(cluster_ids)} clusters with more than {args.min_points} points.")
    
    if len(cluster_ids) > 0:
        print("Here are some of those cluster IDs (up to 50):")
        print(cluster_ids[:50])

    # Optionally, you can save these to a file
    output_file = "large_clusters.txt"
    with open(output_file, "w") as f:
        for cid in cluster_ids:
            f.write(f"{cid}\n")
    print(f"\nAll cluster IDs have been saved to {output_file}")

if __name__ == "__main__":
    main()
