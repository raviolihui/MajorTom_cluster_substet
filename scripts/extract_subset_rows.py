import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def process_single_file(args):
    """Worker function to process a single parquet file."""
    src_file, group_df, dest_dir = args
    src_path = Path(src_file)
    
    if not src_path.exists():
        return {"success": False, "file": src_file, "error": "File not found", "rows": [], "count": 0}
        
    row_groups_to_keep = group_df['row_group'].sort_values().tolist()
    new_rows = []
    
    try:
        # Surgically extract only the required row groups from disk
        pf = pq.ParquetFile(src_file)
        table = pf.read_row_groups(row_groups_to_keep)
        
        # Save the shrunk table to the new directory
        out_file = Path(dest_dir) / src_path.name
        pq.write_table(table, out_file)
        
        # Map old row_group -> new row_group sequentially
        rg_mapping = {old_rg: new_rg for new_rg, old_rg in enumerate(row_groups_to_keep)}
        
        # We must return the dictionary format of the new rows so the main thread can build the manifest
        for _, row in group_df.iterrows():
            new_row_dict = row.to_dict()
            new_row_dict['parquet_file'] = str(out_file)
            new_row_dict['row_group'] = rg_mapping[row['row_group']]
            new_rows.append(new_row_dict)
            
        return {"success": True, "file": src_file, "rows": new_rows, "count": len(row_groups_to_keep)}
        
    except Exception as e:
        return {"success": False, "file": src_file, "error": str(e), "rows": [], "count": 0}

def main():
    parser = argparse.ArgumentParser(description="Extract only the selected rows from the full Parquet files.")
    parser.add_argument("--manifest", type=str, default="/data/databases/MajorTom5T/outputs/subset_manifest_faiss.parquet",
                        help="Path to the subset manifest")
    parser.add_argument("--dest_dir", type=str, required=True,
                        help="Where to save the newly shrunk Parquet files")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of CPU cores to use. Default is 8.")
    
    args = parser.parse_args()
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading manifest from {args.manifest}...")
    manifest = pd.read_parquet(args.manifest)
    
    grouped = list(manifest.groupby('parquet_file'))
    print(f"Found {len(grouped)} unique parquet files to process.")
    print(f"Spinning up {args.workers} CPU workers for parallel extraction...")
    
    # Prepare arguments for the worker pool
    tasks = [(src_file, group_df, args.dest_dir) for src_file, group_df in grouped]
    
    kept_rows = 0
    skipped_files = 0
    new_manifest_rows = []

    # Run in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_single_file, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            result = future.result()
            
            if result["success"]:
                new_manifest_rows.extend(result["rows"])
                kept_rows += result["count"]
            else:
                print(f"\nFailed on {result['file']}: {result['error']}")
                skipped_files += 1

    # Save the updated manifest inside the new subset directory
    new_manifest_path = dest_dir / "metadata.parquet"
    new_manifest_df = pd.DataFrame(new_manifest_rows)
    # We only keep the file name in parquet_url to match expected formats
    if 'parquet_url' not in new_manifest_df.columns:
        new_manifest_df['parquet_url'] = new_manifest_df['parquet_file'].apply(lambda x: Path(x).name)
        new_manifest_df['parquet_row'] = new_manifest_df['row_group']
        
    new_manifest_df.to_parquet(new_manifest_path)
    
    print("\n" + "="*50)
    print("SUBSET EXTRACTION COMPLETE!")
    print(f"Total shrunk parquet files created: {len(grouped) - skipped_files}")
    print(f"Total rows (images) safely extracted: {kept_rows}")
    print(f"New Manifest saved to: {new_manifest_path}")
    print("="*50)

if __name__ == '__main__':
    main()
