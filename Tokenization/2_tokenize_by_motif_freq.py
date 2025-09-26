"""
Tokenize the genome by aggregating motif occurrences into variable-length bins.

This script constructs bin boundaries per chromosome using smoothed motif hit
density, then converts feature placeholders (motifs and CpG annotations) into
per-bin features. Outputs compressed arrays under the motif tokenized genome
directory.

Notes
-----
- Processes autosomes 1-22 by default
- Uses motif placeholders from JASPAR-derived TSV files
- Bin boundaries are derived from peak detection on smoothed motif density
- Features are tokenized as histograms of motif occurrences per bin
- Output format: Nx2 arrays (positive and negative strand features)
- Requires placeholder files from 1_create_annotation_placeholders.py
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import colors as mcolors
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import scipy
from scipy import signal
from time import time
import multiprocessing as mp
from scipy.ndimage import gaussian_filter




MOTIF_DIR = "PATH_TO_MOTIF_TSV_DIR/"

#dic containing the mapppings  matrix_id (in JASPAR):TF_name
def get_motif_name_dict(motif_dir):
    """Map motif matrix IDs (file stems) to TF names from TSV headers."""
    motif_names_dict = {}
    # print(os.listdir(motif_dir)) #THIS PRINTS CORRECTLY
    for file_name in os.listdir(motif_dir):
        # print(file_name)   #THIS NEVER PRINTS!!! (BUT IT DOES PRINT IN LLSUB)
        if file_name.endswith(".tsv"):
            motif_df = pd.read_csv(motif_dir+"/"+file_name, header = None, sep = '\t', nrows = 5)
            motif_names_dict[file_name[:-4]] = motif_df.iloc[0,3]
            # print(motif_names_dict)
    print("got motif names")
    return motif_names_dict


def get_motif_len(mid):
    """Return motif length by reading the header rows of its TSV file."""
    dirr = "PATH_TO_MOTIF_TSV_DIR/"
    filename = [f for f in os.listdir(dirr) if mid in f][0]
    motif_df = pd.read_csv( dirr + filename, sep = '\t', header = None, nrows = 5)
    return  motif_df.iloc[0,2] - motif_df.iloc[0,1]

class Tokenizer():
    """Compute per-bin features from base-level placeholders for a chromosome."""
    def __init__(self, chr_num):
        self.chr_num = chr_num
        self.chr_len = self.get_chr_len()
        self.motif_dir_string = "all_motifs"
       
        print(f"motif directory: {self.motif_dir_string}")
        bin_boundary_file_name = f"PATH_TO_OUTPUT_DIR/motif_tokenized_genome/{self.motif_dir_string}/chr{self.chr_num}_bin_boundaries.npy"
        # if not os.path.isfile(bin_boundary_file_name):
        print("making bin boundaries from scratch")
        self.bin_boundaries = self.get_bin_boundaries()
        self.cpg_placeholder = self.get_cpg_ph()
        
        print("saved bin boundaries")
        
    def get_motif_sums(self):
        """Sum motif placeholders across all motifs to estimate density curve."""
        print("getting bin boundaries from motif sums")
        motif_curve = np.zeros(self.chr_len)
        motif_ph_dir = "PATH_TO_PLACEHOLDER_DIR/chr_annotation_place_holders/all_motifs/"
        motif_MID_to_name = get_motif_name_dict(MOTIF_DIR)
        
        
        for i, mid in tqdm(enumerate(list(motif_MID_to_name.keys()))):
            motif_curve += np.load(motif_ph_dir+f"chr{self.chr_num}_feat_{mid.split('.')[0]}_neg_strand.npz")['arr_0']
            motif_curve += np.load(motif_ph_dir+f"chr{self.chr_num}_feat_{mid.split('.')[0]}_pos_strand.npz")['arr_0']

        return motif_curve

        
        # np.savez_compressed("/home/gridsan/pdavarmanesh/TF_project/annotation_data/hierarchically_resolved_overlaps/chr{}_motifs".format(self.chr_num), self.motif_sums)
        
   

    def get_chr_len(self):
        """Read chromosome lengths from the sizes file and return this chr."""
        chr_lens = {}
        with open("PATH_TO_CHROM_SIZES_FILE", "r+") as file1:
            for line in file1.readlines()[:24]:
                chr_num = line.split('\t')[0][3:]
                chr_len = int(line.split('\t')[1])
                chr_lens[chr_num] = chr_len
        return chr_lens[str(self.chr_num)]
    
    def get_bin_boundaries(self):
        """Derive bin boundaries using peaks of smoothed motif density.

        Post-process to split bins larger than 50 bases with 25 bp step.
        Ensures coverage from 0 to chromosome length inclusive.
        """
        motif_sums = self.get_motif_sums()#################
        print(f"{np.mean(motif_sums>0):.3f} of bases are covered by at least one motif")
        np.save(f"{self.chr_num}_motif_sums.npy", motif_sums)
        smoothed_curve = gaussian_filter(motif_sums, sigma = 3)

        peaks, props = scipy.signal.find_peaks(smoothed_curve, width = 5)
        bin_boundaries = np.sort(np.concatenate([props['right_ips'],props['left_ips']]).astype(int))


        if bin_boundaries[0] < 0:
            bin_boundaries = np.append(0, bin_boundaries[1:])
        elif bin_boundaries[0] > 0:
            bin_boundaries = np.append(0, bin_boundaries)

        if bin_boundaries[-1] > self.chr_len:
            bin_boundaries = np.append(bin_boundaries[:-1], self.chr_len)
        elif bin_boundaries[-1] < self.chr_len:
            bin_boundaries = np.append(bin_boundaries, self.chr_len)
       
        bin_boundaries = np.unique(bin_boundaries)
        assert np.all(bin_boundaries[1:]-bin_boundaries[:-1]>0)
        
        #post-processing: breaking large bins down (to minimize information loss (when we assign
        #rare tokens to <UNK>) and to increase resolution)
        bin_lens = np.diff(bin_boundaries)
        print(f"{np.mean(bin_lens>50):.3f} of bins are larger than 50")
        
        for i in range(1, len(bin_boundaries)):
            if bin_boundaries[i] - bin_boundaries[i-1] > 50:
                bin_boundaries = np.concatenate([bin_boundaries, np.arange(bin_boundaries[i-1], bin_boundaries[i], step = 25)], axis = 0)
  

        return bin_boundaries

    # def calculate_mean(self, i): 
    #     digitized = np.digitize(np.arange(0, self.chr_len), self.bin_boundaries)
    #     return self.feat_array[digitized == i].mean()
    
    def get_cpg_ph(self):
        """Load CpG positions and build a 0/1 placeholder of length chr_len."""
        print("getting CpG annotations")
        with open(f'PATH_TO_CPG_DIR/cpg_chr{self.chr_num}.txt', 'r') as file: #opens the text file in read mode 
            lines = file.readlines()
            cpg_pos = [int(line.split("\n")[0].split()[1]) for line in lines[1:]]
        cpg_place_holder = np.zeros(self.chr_len)
        for pos in cpg_pos:  #length of cpg feature is 2
            cpg_place_holder[pos] = 1
            cpg_place_holder[pos+1] = 1
        assert 2*len(cpg_pos) == np.sum(cpg_place_holder)
        return cpg_place_holder
 
    # in this encoding, each feature value corresponds to the proportion of that feature that falls within that bin (as opposed to proportion of bin that is covered by the feature)

    
    def get_feature_from_placeholder_motif(self, motif_len, ph):
        """Histogram ones into bins and normalize by motif length (per strand)."""

        return np.histogram(np.where(ph)[0], bins = self.bin_boundaries)[0]/motif_len  
        

        
        
    def tokenize_for_features(self, features): #each value is the proportion of the token that is occupied by a feature
        print(f"tokenizing genome for chr {self.chr_num} and features {features}")
        
        for feat in tqdm(features):
            if feat=="cpg": #CpG features are loaded differently
                 # For CpG, use simple histogram without weights since we don't have the weights function
                 tokenized = np.histogram(np.where(self.cpg_placeholder)[0], bins = self.bin_boundaries)[0]
                 # CpG is single-stranded, so duplicate for consistency with motif format
                 tokenized = np.concatenate([np.expand_dims(tokenized, axis = 1), \
                                            np.expand_dims(tokenized, axis = 1)], axis = 1)
            elif feat.startswith("MA"): #for motifs
                motif_len = get_motif_len(feat)
                ph_dir = f"PATH_TO_PLACEHOLDER_DIR/chr_annotation_place_holders/all_motifs/"
                feat_array_pos = np.load(ph_dir+f"chr{self.chr_num}_feat_{feat}_pos_strand.npz")['arr_0'].astype(np.int32)
                feat_array_neg = np.load(ph_dir+f"chr{self.chr_num}_feat_{feat}_neg_strand.npz")['arr_0'].astype(np.int32)
                pos_tokenized = self.get_feature_from_placeholder_motif(motif_len,feat_array_pos)
                neg_tokenized = self.get_feature_from_placeholder_motif(motif_len,feat_array_neg)

                tokenized = np.concatenate([np.expand_dims(pos_tokenized, axis = 1), \
                                            np.expand_dims(neg_tokenized, axis = 1)], axis = 1)
                print(tokenized.shape)

            else:
                #this is for the genomic features
                ph_dir = f"PATH_TO_PLACEHOLDER_DIR/chr_annotation_place_holders/all_motifs/"
                feat_array_pos = np.load(ph_dir+f"chr{self.chr_num}_feat_{feat}_pos_strand.npz")['arr_0'].astype(np.int32)
                feat_array_neg = np.load(ph_dir+f"chr{self.chr_num}_feat_{feat}_neg_strand.npz")['arr_0'].astype(np.int32)
                # Use simple histogram for genomic features
                pos_tokenized = np.histogram(np.where(feat_array_pos)[0], bins = self.bin_boundaries)[0]
                neg_tokenized = np.histogram(np.where(feat_array_neg)[0], bins = self.bin_boundaries)[0]

                tokenized = np.concatenate([np.expand_dims(pos_tokenized, axis = 1), \
                                            np.expand_dims(neg_tokenized, axis = 1)], axis = 1)
                print(tokenized.shape)
            assert np.min(tokenized)>=0., np.min(tokenized)
            np.savez_compressed(
f'PATH_TO_OUTPUT_DIR/motif_tokenized_genome/'\
f'{self.motif_dir_string}/chr{self.chr_num}_feat_{feat}_16bp_bins.npz',tokenized)

            print(tokenized.shape)
       
                    
        return
        
        

        
# 851 total features
# 9 genomic annotations (exon, intron, utr5, utr3, sine, line, ltr,dna, rna)
# 841 motifs
# 1 CpG

def main():
    """Entry point to compute per-bin features for a small chromosome list."""
    motif_MID_to_name = get_motif_name_dict(MOTIF_DIR)

    features = ['cpg']
    features += [mid.split(".")[0] for mid in motif_MID_to_name.keys()]
    
    chr_list = [str(i) for i in range(1,23)]
    # chr_list += ['X', 'Y']
    for chr_num in chr_list: #chr_list:
        print(len(features), "total features")
        # convert_transcript_to_intron([chr_num])

        tokenizer = Tokenizer(chr_num)
        tokenizer.tokenize_for_features(features)
       
        
        
if __name__ == "__main__":
    main()