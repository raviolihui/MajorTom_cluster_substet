import pandas as pd
import os

def enrich_metadata():
    subset_manifest_path = "/data/databases/MajorTom5T/outputs_filtered/images/metadata.parquet"
    original_manifest_path = "/data/databases/Core-S2L2A/metadata.parquet"
    output_path = "/data/databases/MajorTom5T/outputs_filtered/images/metadata_full.parquet"
    
    print(f"Loading bare subset manifest from {subset_manifest_path}...")
    subset_df = pd.read_parquet(subset_manifest_path)
    
    print(f"Loading rich original manifest from {original_manifest_path}...")
    original_df = pd.read_parquet(original_manifest_path)
    
    print(f"Subset columns: {list(subset_df.columns)}")
    print(f"Original columns: {list(original_df.columns)}")
    
    print("\nMerging metadata...")
    # The subset metadata has basically a file name for `parquet_url`, but original has a full URL.
    # We'll create a temporary 'merge_url' column to match on, without touching/deleting ANY original data columns!
    subset_df['merge_url'] = subset_df['parquet_url'].apply(lambda x: str(x).split('/')[-1])
    original_df['merge_url'] = original_df['parquet_url'].apply(lambda x: str(x).split('/')[-1])
    
    # We perform a left merge. Everything in subset_df stays, and it grabs matching data from original_df.
    # Since we are not touching the original `parquet_url`, pandas will keep both if there is a conflict.
    enriched_df = pd.merge(
        subset_df, 
        original_df.drop_duplicates(subset=['merge_url', 'grid_cell']), 
        on=['merge_url', 'grid_cell'], 
        how='left',
        suffixes=('_subset', '_original')
    )
    
    # Drop the temporary merge column, keeping all the real data columns perfectly intact
    enriched_df = enriched_df.drop(columns=['merge_url'])
    
    print(f"\nMerge complete!")
    print(f"New Enriched columns: {list(enriched_df.columns)}")
    print(f"Total rows in enriched dataset: {len(enriched_df)}")
    
    if len(enriched_df) != len(subset_df):
        print("WARNING: Row count changed after merge. Check for duplicate keys.")
    
    print(f"\nSaving enriched metadata to {output_path}...")
    enriched_df.to_parquet(output_path, index=False)
    print("Done! Your subset manifest now contains all the original metadata while keeping mapping references.")
    
if __name__ == "__main__":
    enrich_metadata()