import os
import pandas as pd
import numpy as np


def get_motif_name_dict(motif_dir):
    motif_names_dict = {}
    for file_name in sorted(os.listdir(motif_dir)):
        # print(file_name)   #THIS NEVER PRINTS!!! (BUT IT DOES PRINT IN LLSUB)
        if file_name.endswith(".tsv"):
            motif_df = pd.read_csv(motif_dir+"/"+file_name, header = None, sep = '\t', nrows = 5)
            # display(motif_df)
            motif_names_dict[file_name[:-4]] = motif_df.iloc[0,3]
            # print(motif_names_dict)
    print("got motif names")
    return motif_names_dict

def read_single_fasta_file(file_name):
    f=open(file_name,'r')
    lines=f.readlines()
    seq = ''.join(''.join(lines[1:]).splitlines())
    print("length of seq: {} Mbp".format(len(seq)//1e6))
    print("Number of Ns: {} Mbp".format(seq.count('N')//1e6))
    return seq
def average_intra_cluster_corr(inidices, IDS):
    if len(inidices)<2:
        return 1
    else:
        return np.mean(IDS[np.ix_(inidices,inidices)])
def assign_noise_to_singletons(cluster_labels):
    cluster_labels = np.array(cluster_labels)
    max_label = cluster_labels.max()  # Find the maximum cluster label
    next_label = max_label + 1  # Start assigning singleton clusters from this label
    
    # Find indices of noise points
    noise_indices = np.where(cluster_labels == -1)[0]
    
    # Assign each noise point to a unique singleton cluster
    for idx in noise_indices:
        cluster_labels[idx] = next_label
        next_label += 1
    
    return cluster_labels
