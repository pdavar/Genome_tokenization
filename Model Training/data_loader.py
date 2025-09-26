import sys 
import os
import random
import pickle
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import pyBigWig as pbw
import torch
import multiprocessing as mp
from itertools import repeat
from multiprocessing import Pool
import zarr
from time import time
from collections import defaultdict
import pysam

from torch.utils.data import Dataset
np.random.seed(0)

def get_chromosome_dataset(input_encoding, **kwargs):
    if input_encoding=='ours':
        return ChromosomeDataset(**kwargs) 
    elif input_encoding=='onehot':
        return ChromosomeDatasetOneHot(**kwargs) 
    else:
        raise ValueError

# Create a combined dataset class
class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_ours, dataset_onehot):
        self.dataset_ours = dataset_ours
        self.dataset_onehot = dataset_onehot
        assert len(dataset_ours) == len(dataset_onehot), "Datasets must be the same length"
        
    def __len__(self):
        return len(self.dataset_ours)
    
    def __getitem__(self, idx):
        x_ours, y_ours, bb_ours = self.dataset_ours[idx]
        x_onehot, y_onehot, bb_onehot = self.dataset_onehot[idx]
        # Both y_ours and y_onehot should be the same since they're the same targets
        # assert torch.allclose(y_ours, y_onehot), "Targets from both datasets should be identical"
        return x_ours, x_onehot, bb_ours, bb_onehot, y_onehot
        
def check_interval_binned(args):
    """Standalone function for multiprocessing"""
    interval, output_track_binned, num_tracks, peak_val = args
    bin_start, bin_end = interval

    if np.max(np.array(output_track_binned[bin_start:bin_end, :num_tracks])) > peak_val:
        return interval  # Keep the interval
    return None  # Otherwise, return None
 

def check_interval_bp(args):
    """Standalone function for multiprocessing"""
    interval, output_track_bp, num_tracks, peak_val, bb, bin_res = args
    bin_start, bin_end = interval
    bp_start, bp_end = bb[bin_start],bb[bin_end]
    if (bp_end-bp_start)%bin_res!=0: bp_end-=(bp_end-bp_start)%bin_res
    track_binned = output_track_bp[bp_start:bp_end,:num_tracks].reshape(-1, bin_res, num_tracks).mean(axis=1) 
    if np.max(track_binned) > peak_val:
        return interval  # Keep the interval
    return None  # Otherwise, return None


class GenomeDataset(Dataset):
    '''
    Load all chromosomes
    '''
    def __init__(self, 
                 params, #dictionary containing all required parameters
                 logger,
                 mode = 'train', 
                 data_augmentation=True,
                 keep_only_peaks = False,
                 testing_mode = False,
                 custom_intervals_dict=None):
        self.params = params
        self.logger = logger
        self.data_augmentation = data_augmentation
        self.keep_only_peaks = keep_only_peaks
        self.testing_mode = testing_mode
        self.custom_intervals_dict = custom_intervals_dict
       
        if mode == 'train':
            self.chr_nums = [str(i) for i in range(1,23)]
            self.chr_nums.remove('10')
            self.chr_nums.remove('15')
            logger.log(f"Training on chromosomes {self.chr_nums}" )
        elif mode == 'val':
            self.chr_nums = ['15']
            logger.log(f"Validating on chromosomes {self.chr_nums}" )
        elif mode == 'test':
            self.chr_nums = ['10']
            logger.log(f"Testing on chromosomes {self.chr_nums}" )
        else:
            raise Exception(f'Unknown mode {mode}')
        
        self.chr_dataset_dict = self.load_chromosome_intervals()
        self.chr_dataset_lens = [len(dataset) for dataset in self.chr_dataset_dict.values()]
        
        logger.log(f"Length of {mode} dataset: {self.__len__()}")
        if mode=="train": logger.log(f"Output tracks:  {self.params['track_names'][:self.params['num_tracks']]}")
        if self.data_augmentation: print("using data augmentation")
    
    def load_chromosome_intervals(self):
        chr_dataset_list = []
        for chr_num in self.chr_nums:
            # Get custom intervals for this chromosome if they exist
            custom_intervals = self.custom_intervals_dict.get(chr_num) if self.custom_intervals_dict else None
            
            chr_dataset = get_chromosome_dataset(
                self.params['input_encoding'],
                chr_num=chr_num,
                params=self.params,
                logger=self.logger,
                data_augmentation=self.data_augmentation,
                keep_only_peaks=self.keep_only_peaks,
                testing_mode=self.testing_mode,
                custom_intervals=custom_intervals
            )
            chr_dataset_list.append(chr_dataset)
        
        return {c.chr_num: c for c in chr_dataset_list}
    
    def __len__(self):
        return sum(self.chr_dataset_lens)
    
    def __getitem__(self, idx): #idx is the interval index
        #find the chromosome number that corresponds
        assert idx < self.__len__()
        cumsum = np.cumsum([0]+self.chr_dataset_lens)
        chr_idx = np.argmax(cumsum>idx)-1
        chr_interval = idx - cumsum[chr_idx]
        return self.chr_dataset_dict[self.chr_nums[chr_idx]][chr_interval]
    

class ChromosomeDataset(Dataset):
    
    def __init__(self, 
                 chr_num,
                 params, #dictionary containing all required parameters
                 logger,
                 data_augmentation=True,
                 keep_only_peaks = False,
                 testing_mode = False,
                 custom_intervals = None #used for when we want to only have certain custom regions such as peaks
                ): 
        
        self.chr_num = chr_num
        
        self.block_size = params['block_size'] #in bins (not base pairs)
        self.stride = params['stride'] #bins
        self.input_data_dir = params['input_data_dir']
        self.zarr_tracks_dir = params['zarr_tracks_dir']
        self.data_augmentation=data_augmentation
        self.track_names = params['track_names'][:params['num_tracks']]
        
        self.logger = logger
        self.dtype = np.float32 #datatype used for saving the tokenized sequence as zarr
        
        self.num_tracks = params['num_tracks']
        self.custom_intervals=custom_intervals
        self.bin_boundaries = np.load(f"{self.input_data_dir}/chr{chr_num}_bin_boundaries.npy") #can be used to get the base pairs of each bin
        if testing_mode: 
            assert custom_intervals is None, print("Customm intervals can only be assigned in full mode!")
            self.bin_start = 700000
            self.bin_end = self.bin_start+300000
            # self.bin_boundaries = self.bin_boundaries[self.bin_start:self.bin_end+1]
            self.logger.log(f"testing mode: using only {len(self.bin_boundaries[self.bin_start:self.bin_end+1])} bins in chromosome {self.chr_num}")
        else:
            self.bin_start = 0
            self.bin_end = len(self.bin_boundaries)
        
        self.bin_lens = np.diff(self.bin_boundaries)
        self.bin_lens_scaled = self.bin_lens/np.max(self.bin_lens)
        self.bin_midpoints = (self.bin_boundaries[:-1] + self.bin_boundaries[1:]) / 2
        self.bin_midpoints_scaled = self.bin_midpoints/np.max(self.bin_midpoints)
        self.num_bins = len(self.bin_lens)
        
        # self.feature_sparsities = np.load("/orcd/home/002/parmida/TF_project/genomewide_IDSfeature_sparsities.npy")
        # self.feat_inds = np.where(self.feature_sparsities<params['feature_sparsity_threshold'])[0]

        
        
        self.path = self.input_data_dir+f'chr{self.chr_num}_Nx871_10000chunksize.zarr'
        self.logger.log("using IDS-reduced features")
        # self.track_path = self.zarr_tracks_dir + f'chr{self.chr_num}_normalized_binned_Nx1_RuochiATAC.zarr'
        self.track_path = self.zarr_tracks_dir + f'chr{self.chr_num}_normalized_binned_Nx10.zarr'
        self.output_track_binned = zarr.open(self.track_path, mode = 'r')[:,:params['num_tracks']]
        (output_length, _) = self.output_track_binned.shape
        input_length = zarr.open(self.path, mode = 'r').shape[0]
        assert input_length==output_length, f"input shape {input_length}  and output shape {output_length} don't match"
        assert input_length==self.num_bins, f"input shape {input_length} don't match number of bins {self.num_bins}"
     
        


        #custom intervals must be assigned in the original bin_boundaries space (testing_mode==False)
        if self.custom_intervals is not None: 
            assert not testing_mode, print("Customm intervals can only be assigned in full mode!")
            self.bin_intervals = self.custom_intervals
            self.logger.log(f"using {len(self.bin_intervals)} custom intervals")
        elif keep_only_peaks: #used for visualization: need to read the whole track in the RAM
            self.logger.log("keeping only the intervals with at least one peak in at least one track")
            bin_intervals = self.get_bin_intervals() 
            self.bin_intervals = self.remove_intervals_without_peak(bin_intervals)
        
        else:
            #read the zarr file and only convert to numpy during training (__getitem__())
            self.bin_intervals = self.get_bin_intervals()
        
        
        
           
            
    def add_binsize_binmid(self, chr_features, bin_lens, bin_midpoints):
        
        # bin_lens = np.tile(np.expand_dims(bin_lens, axis = (1,2)), (1,2)).astype(self.dtype)
        bin_lens = np.tile(np.expand_dims(bin_lens, axis = (1)), (1)).astype(self.dtype)
        bin_midpoints = np.tile(np.expand_dims(bin_midpoints, axis = (1)), (1)).astype(self.dtype)
        # bin_lens = bin_lens /self.gw_std[0,0]
        return np.concatenate([bin_midpoints, bin_lens, chr_features], axis = 1)
        
    def add_binsize(self, chr_features, bin_lens):
        
        bin_lens = np.tile(np.expand_dims(bin_lens, axis = (1)), (1)).astype(self.dtype)
        return np.concatenate([bin_lens, chr_features], axis = 1)
 

    

    def remove_intervals_without_peak(self, intervals, peak_val=2):
        self.logger.log(f"{len(intervals)} original intervals.")
    
        # Prepare arguments as tuples
        args = [(interval, self.output_track_binned, self.num_tracks, peak_val) for interval in intervals]
    
        # Parallel execution using multiprocessing
        with mp.Pool(processes=min(4,mp.cpu_count())) as pool:
            results = pool.map(check_interval_binned, args)
    
        # Filter out None values
        new_intervals = [interval for interval in results if interval is not None]
        assert len(new_intervals)>0, print("no intervals with peaks!")
        self.logger.log(f"Using {len(new_intervals)} intervals with peaks.")
    
        return new_intervals
            
    # def remove_intervals_without_peak(self, intervals, peak_val = 2):
        
    #     new_intervals = []
    #     if self.rank==0: self.logger.log(f"{len(intervals)} original intervals.")
    #     for interval in intervals:
    #         bin_start, bin_end = interval
    #         if np.max(self.output_track_binned[bin_start:bin_end,:self.num_tracks])>peak_val:
    #             new_intervals.append(interval)
    #     if self.rank==0: self.logger.log(f"using {len(new_intervals)} intervals with peaks")
    #     return new_intervals
        
        
    def __len__(self):
        return len(self.bin_intervals)
    
    def unif_log_normalize(self):
        """ Given that our input data is highly zero-inflated, this function will map 
        the zeros to a uniform distribution from -0.9 to 0.1, and then take the log1p.
        This results in a data distribution that is more Gaussian-like """
        y = x.copy()
        y[y==0]=(np.random.uniform(size=len(np.where(x==0)[0]))-0.9)
        return np.log1p(y)
    
    # helper function for  __getitem__
    # 852 features: 
    # 10 genomic features: 1 (token length) + 9 (genomic features)
    # 842 motif features: 1 (cpg) + 841 (motifs)
   
    def __getitem__(self, interval_id):
        assert interval_id < len(self.bin_intervals)
        bin_start, bin_end = self.bin_intervals[interval_id] #[bin_start, bin_end)
        
        if self.data_augmentation:
            shift = random.randint(-self.stride//2, self.stride//2+1)
            if bin_start+shift<self.bin_start or bin_end+shift>self.bin_end:
                shift=0 
            bin_start += shift
            bin_end += shift
        
        x = np.array(zarr.open(self.path, mode = 'r')[bin_start:bin_end])#.reshape(bin_end-bin_start, -1)
        # x[x>0]=1
        # x = np.log1p(x)
        x-=0.03
        x*=3

        
        # x = x[:,self.feat_inds]
    
        
        y = np.array(self.output_track_binned[bin_start:bin_end,:self.num_tracks])#i shouldn't subtract this: think about it later
        y/=4.0

        PE_bin_boundaries = self.bin_boundaries[bin_start:bin_end]-self.bin_boundaries[bin_start] #used for positional encoding
        
        return x.astype(np.float32), y.astype(np.float32), PE_bin_boundaries.astype(np.float32), self.bin_boundaries[bin_start], self.bin_boundaries[bin_end]
    
            
    def get_bin_intervals(self):
        '''
        Get intervals for sample data: [[bin_start, bin_end]]
        bin_start and bin_end are in absolute bin space (the entire length of the chromosome)
        '''

        ends = np.arange(self.bin_start+self.block_size, self.bin_end-self.stride, self.stride).reshape(-1, 1) 
        bin_intervals = np.append(ends - self.block_size, ends , axis=1)

        return bin_intervals.astype(int)
    
    #ensuring that each feature is 0 mean and unit variance
    def scale_data(self, data):
        data = data.astype(np.float32)
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            assert False

       

        # Normalize the data
        normalized_data = (data - self.gw_mean) / self.gw_std
        if np.any(np.isnan(normalized_data)) or np.any(np.isinf(normalized_data)):
            logger.log("nan or inf in normalized data: ")
            logger.log(np.where(np.isnan(normalized_data)), np.where(np.isinf(normalized_data)))
            assert False

        # Verifying the normalization (should be approximately 0 mean and unit variance for each feature)
        # mean_check = np.mean(normalized_data)
        # std_check = np.std(normalized_data)

        # print("Mean of normalized data (should be close to 0):", mean_check)
        # print("Standard deviation of normalized data (should be close to 1):", std_check)
        return normalized_data


    
    
class ChromosomeDatasetOneHot(ChromosomeDataset):
    def __init__(self, 
                 chr_num,
                 params, #dictionary containing all required parameters
                 logger,
                 data_augmentation=True,
                 keep_only_peaks = False,
                 testing_mode = False,
                 custom_intervals = None #used for when we want to only have certain custom regions such as peaks
                 ): 
        self.chr_num = chr_num
        self.dtype = np.float32 #datatype used for saving the tokenized sequence as zarr
        self.stride = params['stride'] #bins
        self.num_tracks = params['num_tracks']
        self.block_size = params['block_size']
        self.output_bin_res = params['output_bin_res']
        self.input_data_dir = params['input_data_dir']
        self.fasta_path = params['fasta_dir']+f"chr{chr_num}.fa"
        self.zarr_tracks_dir = params['zarr_tracks_dir']
        self.data_augmentation= data_augmentation
        self.logger = logger
        self.custom_intervals = custom_intervals
        self.track_names = params['track_names'][:params['num_tracks']]
        
        self.bin_boundaries = np.load(f"{self.input_data_dir}/chr{chr_num}_bin_boundaries.npy") #can be used to get the base pairs of each bin
        if testing_mode: 
            self.bin_start = 700000
            self.bin_end = self.bin_start+300000
            self.logger.log(f"testing mode: using only {len(self.bin_boundaries[self.bin_start:self.bin_end+1])} bins in chromosome {self.chr_num}")
        else:
            self.bin_start = 0
            self.bin_end = len(self.bin_boundaries)
        
        self.bp_start = self.bin_boundaries[0]
        self.bp_end = self.bin_boundaries[-1]
        self.bin_lens = np.diff(self.bin_boundaries)
        self.num_bins = len(self.bin_lens)

        self.track_path = self.zarr_tracks_dir + f'chr{chr_num}_normalized_bp_Nx10.zarr'
        self.output_tracks = zarr.open(self.track_path, mode = 'r')[:, :self.num_tracks]


   
         #custom intervals must be assigned in the original bin_boundaries space (testing_mode==False)
        if self.custom_intervals is not None: 
            assert not testing_mode, print("Customm intervals can only be assigned in full mode!")
            self.bin_intervals = self.custom_intervals
            self.logger.log(f"using {len(self.bin_intervals)} custom intervals")
        elif keep_only_peaks: #used for visualization: need to read the whole track in the RAM
            self.logger.log("keeping only the intervals with at least one peak in at least one track")
            bin_intervals = self.get_bin_intervals() 
            self.bin_intervals = self.remove_intervals_without_peak(bin_intervals)
        else:
            self.bin_intervals = self.get_bin_intervals()

        
  
        
        if self.chr_num==1: 
            self.logger.log(f"using one hot encoding of the sequence with output bin size of {self.output_bin_res}")


    def remove_intervals_without_peak(self, intervals, peak_val=2):
        self.logger.log(f"{len(intervals)} original intervals.")
    
        # Prepare arguments as tuples
        args = [(interval, self.output_tracks, self.num_tracks, peak_val, self.bin_boundaries, self.output_bin_res) for interval in intervals]
    
        # Parallel execution using multiprocessing
        with mp.Pool(processes=min(4,mp.cpu_count())) as pool:
            results = pool.map(check_interval_bp, args)
    
        # Filter out None values
        new_intervals = [interval for interval in results if interval is not None]
        self.logger.log(f"Using {len(new_intervals)} intervals with peaks.")
    
        return new_intervals


    def get_bin_intervals(self):
        '''
        Get intervals for sample data: [[bin_start, bin_end]]
        '''

        ends = np.arange(self.bin_start+self.block_size, self.bin_end-self.stride, self.stride).reshape(-1, 1) 
        bin_intervals = np.append(ends - self.block_size, ends , axis=1)

        return bin_intervals.astype(int)

    #making sure that the length of the sequence in bps is exactly 16 times more than the block size
    def adjust_bp_to_blocksize(self, bp_start, bp_end, bin_block_size):
        input_size = bin_block_size * self.output_bin_res
        
        middle_idx = int(np.mean([bp_start, bp_end]))
        new_bp_start = max(0, middle_idx - input_size // 2)
        new_bp_end = new_bp_start + input_size
        return new_bp_start, new_bp_end
      
    def get_bp_boundaries(self, interval_id):
        bin_start, bin_end = self.bin_intervals[interval_id] #[bin_start, bin_end)
        
        if self.data_augmentation:
            shift = random.randint(-self.stride//2, self.stride//2+1)
            if bin_start+shift<self.bin_start or bin_end+shift>self.bin_end:
                shift=0 
            bin_start += shift
            bin_end += shift
        
        
        bp_start, bp_end = self.bin_boundaries[bin_start],self.bin_boundaries[bin_end]
        bp_start, bp_end = self.adjust_bp_to_blocksize(bp_start, bp_end, bin_end-bin_start)
        return bp_start, bp_end
        
    def __getitem__(self, interval_id):
        bp_start, bp_end = self.get_bp_boundaries(interval_id)
           
        x = self.one_hot_encode(self.read_fasta_file_indexed(bp_start,bp_end)) #Nx4
        # padding = np.random.binomial(n=1, p=0.05, size=(x.shape[0], 400-4))
        # x = np.concatenate([x, padding], axis=1)  # Now Nx800

        # y = self.output_tracks[bp_start:bp_end,:self.num_tracks]

        y = self.output_tracks[bp_start:bp_end,:self.num_tracks]
        y = self.bin_track(y)
        
        y/=4.0
        # y = np.exp(y)-1

        PE_bin_boundaries = np.arange(0, self.block_size*self.output_bin_res, self.output_bin_res)
        assert (x.shape[0]//self.output_bin_res)==y.shape[0], print(y.shape, x.shape, bp_end-bp_start)
        
        return x.astype(np.float32), y.astype(np.float32), PE_bin_boundaries.astype(np.float32), bp_start, bp_end
    
    def one_hot_encode(self, sequence):
        """
        Converts a DNA sequence into its one-hot encoded representation.
        :param sequence: A string representing a DNA sequence (e.g., 'ATCG').
        :return: A numpy array of shape (len(sequence), 4) representing the one-hot encoded sequence.
        """
        # Mapping of nucleotides to one-hot encoding
        encoding = {'A': [1, 0, 0, 0], 
                    'T': [0, 1, 0, 0], 
                    'C': [0, 0, 1, 0], 
                    'G': [0, 0, 0, 1],
                    'a': [1, 0, 0, 0], 
                    't': [0, 1, 0, 0], 
                    'c': [0, 0, 1, 0], 
                    'g': [0, 0, 0, 1],
                    'n': [0, 0, 0, 0],
                    'N': [0, 0, 0, 0]}
        
        one_hot_matrix = np.array([encoding[nucleotide] for nucleotide in sequence])
        return one_hot_matrix
    
    def read_fasta_file(self):
        f=open(self.fasta_path,'r')
        lines=f.readlines()
        seq = ''.join(''.join(lines[1:]).splitlines())
        return seq[self.bp_start:self.bp_end]
   
    def read_fasta_file_indexed(self, start_bp, end_bp):
        """
        Given a start and end bp (1 -indexed), read from the fasta_path only the 
        relevant sequence without loading in the entire
        chromosome into the memory.

        NOTE: THIS METHOD TAKES 1-INDEXED INPUTS AND CONVERTS TO 0-INDEXED FOR YOU.
        """
        fasta = pysam.FastaFile(self.fasta_path)
        try:
            sequence = fasta.fetch(f"chr{self.chr_num}", start_bp, end_bp)  # Convert to 0-based indexing
            return sequence
        except ValueError:
            print(self.chr_num, start_bp, end_bp)
        fasta.close()
        
    
   
    def bin_track(self, y):
        assert y.shape[0]%self.output_bin_res==0
        # pad_size = (self.output_bin_res - y.shape[0] % self.output_bin_res) % self.output_bin_res  # This gives the total padding needed

        # pad_left = pad_size // 2  # Integer division for left padding
        # pad_right = pad_size - pad_left  # Remaining padding for the right side

        # # Step 3: Pad the array symmetrically along the first dimension
        # y_padded = np.pad(y, ((pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
        y_binned = y.reshape(-1, self.output_bin_res, y.shape[1]).mean(axis=1) 
        
        
        return y_binned

        
