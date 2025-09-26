"""
Compute per-bin averages over clustered features and write Zarr arrays.

Given a per-base NxM Zarr matrix per chromosome and a vector of cluster labels
for the M features, this script averages features within each cluster for each
position using chunked multiprocessing and saves the result as Zarr.
"""

import numpy as np
import zarr
import os
from multiprocessing import Pool


def process_chunk(args):
    """Worker function to compute cluster averages for a chunk."""
    chunk_index, X, cluster_labels = args
    chunk = np.array(X[chunk_index[0]:chunk_index[1]]).astype(np.float32).reshape(chunk_index[1]-chunk_index[0], -1)
  
    C = len(np.unique(cluster_labels))
    chunk_averages = np.zeros((chunk.shape[0], C), dtype=np.float32)
    for i, cluster_idx in enumerate(sorted(np.unique(cluster_labels))):
        cluster_inds = np.where(cluster_labels==cluster_idx)[0]
        cluster_features = chunk[:, cluster_inds]
        if cluster_features.size > 0:
            chunk_averages[:, i] = cluster_features.mean(axis=1)
    return chunk_averages

def compute_cluster_averages_multiprocessing(zarr_file, cluster_labels):
    """
    Compute the average of features with the same cluster assignment using multiprocessing.
    
    Parameters:
        zarr_file (str): Path to the Zarr file containing the dataset X (NxM).
        cluster_labels (array-like): Array of length M containing cluster labels for the M features.
        
    Returns:
        np.ndarray: A 2D array of shape NxC, where C is the number of unique clusters.
    """
    # Load the Zarr dataset
    X = zarr.open(zarr_file, mode='r')
    N, _, _ = X.shape

    # Find unique clusters and create a mapping from cluster to columns
    C = len(np.unique(cluster_labels))

    # Prepare the output array
    cluster_averages = np.zeros((N, C), dtype=np.float32)
    chr_length = len(X)
    m = 100000
    chunk_indices = [(i, min(i + m, chr_length)) for i in range(0, chr_length, m)]

    # Prepare arguments for each chunk
    tasks = [(chunk_index, X, cluster_labels) for chunk_index in chunk_indices]

    # Use multiprocessing Pool to parallelize the computation
    with Pool(processes=8) as pool:
        results = pool.map(process_chunk, tasks)

    # Combine results into the final output array
    for i, chunk_idxs in enumerate(chunk_indices):
        cluster_averages[chunk_idxs[0]:chunk_idxs[1], :] = results[i]

    return cluster_averages

def main():
    """Main function to compute cluster-averaged features for chromosomes."""
    # Configuration paths - replace with your actual directories
    data_dir = "PATH_TO_MOTIF_TOKENIZED_GENOME_DIR/"
    new_data_dir = "PATH_TO_CLUSTERED_FEATURES_OUTPUT_DIR/"
    cluster_labels_path = "PATH_TO_CLUSTER_LABELS_FILE"
    num_clusters = 862
    
    # Ensure output directory exists
    os.makedirs(new_data_dir, exist_ok=True)
    
    # Load cluster labels
    if not os.path.exists(cluster_labels_path):
        print(f"Error: Cluster labels file not found: {cluster_labels_path}")
        return
    
    cluster_labels = np.load(cluster_labels_path)
    print(f"Loaded cluster labels for {len(cluster_labels)} features, {num_clusters} clusters")
    
    # Process chromosomes 10-22
    for chr_num in range(10, 23):
        print(f"Processing chromosome {chr_num}")
        
        zarr_file = os.path.join(data_dir, f'chr{chr_num}_Nx842x2_1000chunksize.zarr')
        
        if not os.path.exists(zarr_file):
            print(f"Warning: Zarr file not found: {zarr_file}")
            continue
        
        try:
            result = compute_cluster_averages_multiprocessing(zarr_file, cluster_labels)
            
            # Save clustered features as Zarr
            z = zarr.array(result, chunks=(10000, None))
            output_path = os.path.join(new_data_dir, f'chr{chr_num}_Nx{num_clusters}_10000chunksize.zarr')
            zarr.save(output_path, z)
            
            print(f"Saved clustered features for chr {chr_num} with shape {result.shape}")
            
        except Exception as e:
            print(f"Error processing chromosome {chr_num}: {e}")
            continue
    
    print("Cluster averaging complete!")

if __name__ == "__main__":
    main()
