import pandas as pd

def inspect_metadata():
    file_path = "/data/databases/MajorTom5T/outputs_filtered/images/metadata_full.parquet"
    print(f"Loading metadata from {file_path}...\n")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("--- General Info ---")
    print(f"Total Rows (Images): {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}\n")

    print("--- First 3 Rows ---")
    # Displaying transposed for easier reading of columns
    print(df.head(3).T)
    print("\n")

    print("--- Missing Values (NaNs) ---")
    print(df.isna().sum())
    print("\n")

    print("--- Quick Statistics ---")
    # Just selecting a few interesting numerical columns if they exist
    cols_to_stat = ['cloud_cover', 'nodata', 'centre_lat', 'centre_lon']
    existing_cols = [col for col in cols_to_stat if col in df.columns]
    
    if existing_cols:
        print(df[existing_cols].describe())
    else:
        print("Numerical columns for statistics not found.")
        
    print("\n--- Value Counts for 'cluster_id' (Top 5) ---")
    if 'cluster_id' in df.columns:
        print(df['cluster_id'].value_counts().head(5))

if __name__ == "__main__":
    inspect_metadata()
