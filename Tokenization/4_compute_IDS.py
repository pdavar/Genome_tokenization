"""
Compute genome-wide IDS (Information Decomposition Statistics) summaries.

Utilities to:
- compute means of transformed features across the genome
- compute per-bin-length mean/std augmentation
- compute cross-order IDS correlation matrices using chunked multiprocessing

The script's main function computes IDS statistics using multiprocessing.
"""

import zarr
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool, Manager


# Configuration paths - replace with your actual directories
input_data_dir = "PATH_TO_MOTIF_TOKENIZED_GENOME_DIR/"
ids_output_dir = "PATH_TO_IDS_OUTPUT_DIR/"
       
def get_mean(n):
    """Compute mean of order-n transformed features over all autosomes."""
  
    feat_sums = np.zeros((842,2))
    chr_lens = 0
    for chr_num in np.arange(1,23):

        path = input_data_dir+f'chr{chr_num}_Nx842x2_1000chunksize.zarr'
        data = np.array(zarr.open(path, mode = 'r'))
        print("chr ", chr_num)

        feats = transform_feats(data.astype(np.float32), n)
        feat_sums += np.sum(feats, axis = 0)
        chr_lens += data.shape[0]
     
    np.save(f"{ids_output_dir}/genome_wide_transformed_mean_n{n}.npy", feat_sums/chr_lens)
    print(f"saved mean for transformed features {n}")
    return 
def get_global_mean_IDS():
    """Compute per-order transformed means using chunked multiprocessing."""
    # Define chunk size
    m = 100000  # length of each chunk to be processed
    num_features = 842*2
    # Parallel processing
    for n in range(6):
        print("n = ", n)
        feat_sums = np.zeros(num_features)
        chr_lens = 0
        results = []
        for chr_num in tqdm(np.arange(1, 23)):
            print("Processing chromosome:", chr_num)
            path = input_data_dir + f'chr{chr_num}_Nx842x2_1000chunksize.zarr'
            zarr_file = zarr.open(path, mode='r')
            chr_length = len(zarr_file)
            chunk_indices = [(i, min(i + m, chr_length)) for i in range(0, chr_length, m)]

            # Prepare arguments for each chunk
            tasks = [(chunk_index, path, num_features, n) for chunk_index in chunk_indices]
            with Pool(processes=8) as pool:

                # Process all chunks for this chromosome in parallel
                chr_results = pool.map(process_chunk_mean_IDS, tasks)
                results.extend(chr_results)

        # Aggregate results
        print("aggregating results...")
        for chunk_sum, sample_size in results:
            feat_sums += chunk_sum
            chr_lens += sample_size

        # Compute global correlation
        np.save(f"{ids_output_dir}/genome_wide_transformed_mean_n{n}.npy", feat_sums/chr_lens)
        print(f"saved mean for transformed features {n}")

def process_chunk_mean_IDS(args):
    """Process a single chunk of data."""
    chunk_index, path, num_features, n = args
    zarr_file = zarr.open(path, mode='r')
    chunk = np.array(zarr_file[chunk_index[0]:chunk_index[1]]).astype(np.float32).reshape(-1, num_features)
    print(len(chunk))
    chunk = transform_feats(chunk, n)
    chunk_sum = np.sum(chunk, axis = 0)
    sample_size = chunk.shape[0]
    return chunk_sum, sample_size

def get_bin_length_mean():
    """Return genome-wide average bin length from saved boundaries."""
    bin_length_sum = 0
    chr_lens = 0
    for chr_num in np.arange(1,23):
        print(chr_num)
        bin_lens = np.diff(np.load(f"{input_data_dir}/chr{chr_num}_bin_boundaries.npy"))
        bin_length_sum += np.sum(bin_lens)
        chr_lens += bin_lens.shape[0]
    print("genome wide bin length average: ", bin_length_sum.astype(np.float32)/chr_lens)
    return bin_length_sum.astype(np.float32)/chr_lens






def transform_feats(feats, n):
    """Apply order-n RBF-like transformation used by IDS."""
    return np.exp(-(feats**2)/2)*(feats**n)

def process_chunk(args):
    """Process a single chunk of data."""
    chunk_index, path, global_mean, num_features = args
    zarr_file = zarr.open(path, mode='r')
    chunk = np.array(zarr_file[chunk_index[0]:chunk_index[1]]).astype(np.float32).reshape(-1, num_features)
    chunk_centered = chunk - global_mean
    feat_cov = np.dot(chunk_centered.T, chunk_centered)
    feat_var = np.sum(chunk_centered ** 2, axis=0)
    sample_size = chunk.shape[0]
    return feat_cov, feat_var, sample_size

def process_chunk_IDS(args):
    """Process a single chunk of data."""
    chunk_index, path, n1, n2, num_features = args
    zarr_file = zarr.open(path, mode='r')
    chunk = np.array(zarr_file[chunk_index[0]:chunk_index[1]]).astype(np.float32).reshape(-1, num_features)
    chunk1 = transform_feats(chunk, n1)
    chunk2 = transform_feats(chunk, n2)
    global_mean1 = np.load(f"{ids_output_dir}/genome_wide_transformed_mean_n{n1}.npy").reshape(-1)
    global_mean2 = np.load(f"{ids_output_dir}/genome_wide_transformed_mean_n{n2}.npy").reshape(-1)
    
    
    chunk1_centered = chunk1 - global_mean1
    chunk2_centered = chunk2 - global_mean2
    feat_cov = np.dot(chunk1_centered.T, chunk2_centered)
    feat1_var = np.sum(chunk1_centered ** 2, axis=0)
    feat2_var = np.sum(chunk2_centered ** 2, axis=0)
    sample_size = chunk.shape[0]
    return feat_cov, feat1_var, feat2_var, sample_size

def get_global_IDS():
    """Compute IDS correlation between orders n1 and n2 across the genome."""
    num_features = 1684
    for n1 in range(4,5):
        for n2 in range(n1+1):
            print(f"n1 : {n1}, n2: {n2}")
            running_feat_cov = np.zeros((num_features, num_features), dtype=np.float32)
            running_feat1_var = np.zeros(num_features, dtype=np.float32)
            running_feat2_var = np.zeros(num_features, dtype=np.float32)
            running_sample_size = 0

            # Shared data structures to manage outputs from parallel workers
            results = []

            # Define chunk size
            n = 100000  # length of each chunk to be processed

            # Parallel processing
            with Pool(processes=8) as pool:
                for chr_num in tqdm(np.arange(1, 23)):
                    print("Processing chromosome:", chr_num)
                    path = input_data_dir + f'chr{chr_num}_Nx842x2_1000chunksize.zarr'
                    zarr_file = zarr.open(path, mode='r')
                    chr_length = len(zarr_file)
                    chunk_indices = [(i, min(i + n, chr_length)) for i in range(0, chr_length, n)]

                    # Prepare arguments for each chunk
                    tasks = [(chunk_index, path, n1, n2, num_features) for chunk_index in chunk_indices]

                    # Process all chunks for this chromosome in parallel
                    chr_results = pool.map(process_chunk_IDS, tasks)
                    results.extend(chr_results)

            # Aggregate results
            print("aggregating results...")
            for feat_cov, feat1_var, feat2_var, sample_size in results:
                running_feat_cov += feat_cov
                running_feat1_var += feat1_var
                running_feat2_var += feat2_var
                running_sample_size += sample_size

            # Compute global correlation
            std_devs1 = np.sqrt(running_feat1_var / running_sample_size)
            std_devs2 = np.sqrt(running_feat2_var / running_sample_size)
            running_global_corr = running_feat_cov / (std_devs1[:, None] * std_devs2[None, :] * running_sample_size)

            # Save result
            np.save(f"{ids_output_dir}/genome_wide_IDS_{n1}_{n2}.npy", running_global_corr)
            print(f"Saved genome-wide IDS for {n1} and {n2}.")


def main():
    """Main function to compute IDS statistics."""
    print("Starting IDS computation...")
    
    # Ensure output directory exists
    os.makedirs(ids_output_dir, exist_ok=True)
    
    try:
        # First compute global means for different transformation orders
        print("Computing global means for transformed features...")
        get_global_mean_IDS()
        
        # Then compute IDS correlations
        print("Computing IDS correlations...")
        get_global_IDS()
        
        print("IDS computation complete!")
        
    except Exception as e:
        print(f"Error during IDS computation: {e}")
        raise

if __name__ == "__main__":
    main()