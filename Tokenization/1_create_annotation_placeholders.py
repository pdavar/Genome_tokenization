import gffpandas.gffpandas as gffpd
import numpy as np
import pyranges as pr
import os
import pandas as pd
from tqdm import tqdm
import re
import os


"""
Create per-chromosome binary placeholders for TF motif hits.

For each chromosome and each motif present in `MOTIF_DIR` (JASPAR-derived TSVs),
the script writes two compressed arrays: one for the positive strand and one for
the negative strand. Indices within motif hit intervals are 1, others are 0.

Notes
-----
- Motif IDs are discovered from TSV files in `MOTIF_DIR`.
- Outputs are saved under `chr_annotation_place_holders/all_motifs/` with the
  pattern: `chr{chr}_feat_{motifID}_{pos|neg}_strand.npz`.
- Currently processes autosomes 1–22 (no environment variable control).
"""



FASTA_DIR = "PATH_TO_FASTA_DIR/" 
ANNOTATION_DIR = "PATH_TO_ANNOTATION_DIR/"
MOTIF_DIR = "PATH_TO_MOTIF_TSV_DIR/"
CHROM_SIZES_FILE = "PATH_TO_HG38_CHROM_SIZES"
genomic_features = ['exon', 'transcript', 'utr3', 'utr5', 'sine', 'line', 'ltr', 'dna', 'rna']
chr_annotation = gffpd.read_gff3(ANNOTATION_DIR + 'gencode.v43.basic.annotation.gff3').df #for exon and introns
TE_annotations = pr.read_gtf(ANNOTATION_DIR + "hg38_rmsk_TE_20200804.gtf", full=True).df
TE_annotations = TE_annotations.rename(columns={"class_id": "type", "Start": "start", "End": "end", "Chromosome":'seq_id', "Strand": "strand" })
print("loaded the files")





def annotation_to_array(df, # a dataframe with two columns {start, end}
                         zero_seq): # array of zeros whose length is the same as the chromosome of interest               
    """Convert start/end intervals with strand into two binary arrays.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain `start`, `end`, and `strand` columns.
    zero_seq : np.ndarray
        A zero-initialized array of chromosome length (dtype will be cast to uint8).

    Returns
    -------
    (np.ndarray, np.ndarray)
        Tuple of `(pos_strand, neg_strand)` uint8 arrays with 1s covering
        inclusive intervals [start, end].
    """
    pos_strand = zero_seq.copy()
    neg_strand = zero_seq.copy()
    for index, row in df.iterrows():
        start = row['start']
        end = row['end']
        if row.strand == "+":
            pos_strand[start:end+1] = 1
        else:
            neg_strand[start:end+1] = 1
    return pos_strand.astype(np.uint8), neg_strand.astype(np.uint8)

#dic containing the mapppings  matrix_id (in JASPAR):TF_name
def get_motif_name_dict(motif_dir):
    """Map motif matrix IDs (file stems) to TF names using the TSV headers."""
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

def get_feature_annotation_df(chr_num, feature, motif_MID_to_name):
    """Return a dataframe of intervals for a given feature and chromosome.

    Handles standard genomic features (exon, utrs, transcripts, TE classes)
    as well as motif IDs present in `motif_MID_to_name`.
    """
    if feature == 'exon':
        return chr_annotation.loc[(chr_annotation.seq_id == 'chr{}'.format(chr_num)) & (chr_annotation.type == "exon")]
    elif feature == 'transcript':
        return chr_annotation.loc[(chr_annotation.seq_id == 'chr{}'.format(chr_num)) & (chr_annotation.type == "transcript")]
    elif feature == 'utr3':
        return chr_annotation.loc[(chr_annotation.seq_id == 'chr{}'.format(chr_num)) & (chr_annotation.type == "three_prime_UTR")]    
    elif feature == 'utr5':
        return chr_annotation.loc[(chr_annotation.seq_id == 'chr{}'.format(chr_num)) & (chr_annotation.type == "five_prime_UTR")]
    elif feature == 'sine':
        return TE_annotations.loc[(TE_annotations.seq_id == 'chr{}'.format(chr_num)) & (TE_annotations.type == "SINE")]
    elif feature == 'line':
        return TE_annotations.loc[(TE_annotations.seq_id == 'chr{}'.format(chr_num)) & (TE_annotations.type == "LINE")]
    elif feature == 'ltr':
        return TE_annotations.loc[(TE_annotations.seq_id == 'chr{}'.format(chr_num)) & (TE_annotations.type == "LTR")]
    elif feature == 'dna':
        return TE_annotations.loc[(TE_annotations.seq_id == 'chr{}'.format(chr_num)) & (TE_annotations.type == "DNA")]
    elif feature == 'rna':
        return TE_annotations.loc[(TE_annotations.seq_id == 'chr{}'.format(chr_num)) & (TE_annotations.type == "RNA")]
    elif feature in motif_MID_to_name.keys():
        mat_id = feature
        motif_df = pd.read_csv(MOTIF_DIR + "/" + f"{mat_id}.tsv", sep = '\t', header = None)
        motif_df.columns = ['chr_num', 'start', 'end','TF_name', '__', '___','strand']
        assert motif_df.TF_name.nunique() == 1, f"motif dataframe {feature} has more than one motif"
        return motif_df.loc[motif_df.chr_num == f"chr{chr_num}"]
    else:
        raise Exception(f"feature {feature} is not a valid feature")
        return None
    
    

    
 



def main(chr_list):
    """Build and save placeholder arrays for all requested chromosomes."""
    print("entered main")
    chr_lens = {}
  
            
    motif_MID_to_name = get_motif_name_dict(MOTIF_DIR)

    features = list(motif_MID_to_name.keys())

    print(motif_MID_to_name.keys())
    chr_lens = {}
    with open(CHROM_SIZES_FILE, "r+") as file1:
        for line in file1.readlines()[:24]:
            chr_num = line.split('\t')[0][3:]
            chr_len = int(line.split('\t')[1])
            chr_lens[chr_num] = chr_len
        
  
    # Ensure output directory exists
    os.makedirs("chr_annotation_place_holders/all_motifs", exist_ok=True)

    for chr_num in chr_list:

        chr_len = chr_lens[chr_num]
        print("chromosome {} of length {}Mbp".format(chr_num, chr_len/1e6))

        for feat in tqdm(features):
            feat_df = get_feature_annotation_df(chr_num, feat, motif_MID_to_name)
            pos_strand, neg_strand = annotation_to_array(feat_df, np.zeros(chr_lens[chr_num], dtype = np.uint8))
            if feat.startswith("MA"): feat = feat.split(".")[0]
            np.savez_compressed(f"chr_annotation_place_holders/all_motifs/chr{chr_num}_feat_{feat}_pos_strand", pos_strand)
            np.savez_compressed(f"chr_annotation_place_holders/all_motifs/chr{chr_num}_feat_{feat}_neg_strand", neg_strand)
            
    print("done!")
    return
    
# argv options: none, x, or two numbers
if __name__ == '__main__':
  
    chr_list = [str(i) for i in range(1,23)]  
    
    main(chr_list)

print("done")