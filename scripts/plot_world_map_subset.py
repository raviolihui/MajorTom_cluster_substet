import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_world_map():
    metadata_path = '/data/databases/MajorTom5T/outputs_filtered/images/metadata_full.parquet'
    output_path = 'pics/subset_world_map.png'
    
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_parquet(metadata_path, columns=['centre_lat', 'centre_lon'])
    
    # Drop NaNs just in case
    df = df.dropna(subset=['centre_lat', 'centre_lon'])
    
    print(f"Plotting {len(df)} images on the world map...")
    
    plt.figure(figsize=(16, 8), facecolor='white')
    
    # Plot formatting to make it look like a beautiful world map
    plt.style.use('default') # Use default white background
    
    # Scatter plot of all image coordinates
    # We use a very small marker size (s=0.05) and low alpha (transparency=0.05)
    # so dense areas look greener, while sparse areas are visible
    plt.scatter(
        x=df['centre_lon'], 
        y=df['centre_lat'], 
        s=0.05, 
        alpha=0.1, 
        color='green', 
        marker='.'
    )
    
    plt.title('Global Distribution of the 450,000 MajorTom Subset Images', fontsize=18, color='black', pad=20)
    plt.xlabel('Longitude', fontsize=14, color='black')
    plt.ylabel('Latitude', fontsize=14, color='black')
    
    # Set the strict limits of the Earth's coordinates
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    
    # Draw simple gridlines to indicate the equator and prime meridian
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Tight layout to remove excess border
    plt.tight_layout()
    
    os.makedirs('pics', exist_ok=True)
    print(f"Saving high-resolution map to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Done! You can view the image.")

if __name__ == "__main__":
    plot_world_map()