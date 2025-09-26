
"""
Convert BigWig tracks to Zarr format for efficient genomic data storage.

This script reads BigWig files containing genomic tracks (e.g., ATAC-seq signals),
processes them per chromosome using pre-computed bin boundaries, and saves the
results as compressed Zarr arrays for downstream analysis.

Notes
-----
- Processes autosomes 1-22
- Requires bin boundaries from 2_tokenize_by_motif_freq.py
- Applies log1p transformation to signal values
- Handles NaN values by converting to zeros
- Outputs Zarr files with chunked storage for efficient access
- Currently configured for "Ruochi.bigWig" files specifically
"""

import numpy as np
import zarr
import time
from tqdm import tqdm
import os
import pandas as pd
import pyBigWig as pbw

# Configuration paths - replace with your actual directories
data_dir = "PATH_TO_MOTIF_TOKENIZED_GENOME_DIR/"
output_directory = "PATH_TO_GENOMIC_FEATURES_DIR/"
binned_tracks_dir = "PATH_TO_NORMALIZED_ZARR_TRACKS_DIR/"
def process_chromosome(chr_num, data_dir, output_directory, binned_tracks_dir):
    """Process a single chromosome: load bin boundaries, read BigWig tracks, save as Zarr."""
    print(f"Processing chromosome {chr_num}")
    
    # Load bin boundaries for this chromosome
    bin_boundaries_path = f"{data_dir}/chr{chr_num}_bin_boundaries.npy"
    if not os.path.exists(bin_boundaries_path):
        print(f"Warning: Bin boundaries file not found: {bin_boundaries_path}")
        return
    
    bin_boundaries = np.load(bin_boundaries_path)
    tracks = []

    # Process all BigWig files in the output directory
    if not os.path.exists(output_directory):
        print(f"Warning: Output directory not found: {output_directory}")
        return
        
    for filename in sorted(os.listdir(output_directory)):
        if filename.endswith("Ruochi.bigWig"):  # Filter for specific file pattern
            print(f"Processing {filename}")
            filepath = os.path.join(output_directory, filename)
            
            try:
                # Read per-base values between first and last bin edge
                with pbw.open(filepath) as bw_file:
                    signals = np.array(bw_file.values(f"chr{chr_num}", 
                                                    bin_boundaries[0], 
                                                    bin_boundaries[-1]), 
                                     dtype=np.float32)
                
                # Handle NaN values and apply log transformation
                signals = np.nan_to_num(signals)
                signals = np.log1p(signals)
                tracks.append(signals)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    if not tracks:
        print(f"No tracks processed for chromosome {chr_num}")
        return
    
    # Stack all tracks and save as Zarr
    y = np.stack(tracks, axis=1)
    
    # Ensure output directory exists
    os.makedirs(binned_tracks_dir, exist_ok=True)
    
    # Chunk along genomic axis for efficient streaming
    z = zarr.array(y, chunks=(10000, None))
    output_path = f'{binned_tracks_dir}/chr{chr_num}_normalized_bp_Nx1_RuochiATAC.zarr'
    zarr.save(output_path, z)
    
    print(f"Saved Zarr file for chr {chr_num} with shape {y.shape}")
    return y.shape

def main():
    """Main function to process all chromosomes."""
    print("Starting BigWig to Zarr conversion...")
    
    for chr_num in np.arange(1, 23):
        try:
            process_chromosome(chr_num, data_dir, output_directory, binned_tracks_dir)
        except Exception as e:
            print(f"Error processing chromosome {chr_num}: {e}")
            continue
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()

