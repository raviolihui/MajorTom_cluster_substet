import os
import shutil
import pandas as pd
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Move/Copy subset physical files based on manifest.")
    parser.add_argument("--manifest", type=str, default="/data/databases/MajorTom5T/outputs_filtered/images/subset_manifest_faiss.parquet",
                        help="Path to subset manifest")
    parser.add_argument("--dest_dir", type=str, required=True,
                        help="Destination directory for the subset files")
    parser.add_argument("--mode", choices=["copy", "symlink"], default="symlink",
                        help="Whether to make physical copies or just symbolic links (saves disk space)")
    
    args = parser.parse_args()
    
    # 1. Load the manifest
    print(f"Loading manifest from {args.manifest}...")
    manifest = pd.read_parquet(args.manifest)
    
    # 2. Extract unique parquet files (since multiple rows might come from the same physical file)
    unique_files = manifest["parquet_file"].unique()
    print(f"Manifest contains {len(manifest)} selected images coming from {len(unique_files)} unique parquet files.")
    
    # 3. Create destination directory
    os.makedirs(args.dest_dir, exist_ok=True)
    print(f"Transferring files to {args.dest_dir} using mode: {args.mode.upper()}")
    
    # 4. Transfer the files
    success_count = 0
    missing_count = 0
    
    for src_path in tqdm(unique_files, desc="Processing files"):
        if not os.path.exists(src_path):
            missing_count += 1
            continue
            
        file_name = os.path.basename(src_path)
        dest_path = os.path.join(args.dest_dir, file_name)
        
        # Skip if already exists
        if os.path.exists(dest_path):
            success_count += 1
            continue
            
        try:
            if args.mode == "symlink":
                os.symlink(src_path, dest_path)
            elif args.mode == "copy":
                shutil.copy2(src_path, dest_path)
            success_count += 1
        except Exception as e:
            print(f"\nError transferring {src_path}: {e}")
            
    # 5. Bring over the modified manifest to the new folder so ML loaders can find it
    manifest_dest = os.path.join(args.dest_dir, "metadata.parquet")
    
    # Update paths in the manifest to point strictly to filenames in the new folder
    def update_path(old_path):
        return os.path.basename(old_path)
        
    manifest["parquet_url"] = manifest["parquet_file"].apply(update_path)
    manifest.to_parquet(manifest_dest)
    
    print("\n" + "="*50)
    print("TRANSFER COMPLETE!")
    print(f"Successfully transferred: {success_count} files.")
    if missing_count > 0:
        print(f"WARNING: {missing_count} source files were not found on disk.")
    print(f"New manifest saved to: {manifest_dest}")
    print("="*50)

if __name__ == "__main__":
    main()
