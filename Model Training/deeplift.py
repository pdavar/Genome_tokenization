import numpy as np
from model import *
from data_loader import *
from utils import *
import torch
import torch.nn as nn
from profiler import *
import seaborn as sns
from captum.attr import DeepLift
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
import logomaker

with open('training_params.yaml', 'r') as f:
    params = yaml.safe_load(f)

torch.manual_seed(123)
np.random.seed(123)

plot = False
chr_num = 15
# bp_start = 1000000
peak_window = 50 #used for visualization 
input_window = 400 #context length in bins

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
my_exp_num= 65426942   
baseline_exp_num=  65917496
my_epoch = 2
baseline_epoch = 9
bb_file_path = "/home/parmida/orcd/pool/data/motif_tokenized_genome/"
bb = np.load(f"{bb_file_path}/chr{chr_num}_bin_boundaries.npy")
motif_names = np.load("/home/parmida/orcd/pool/data/ordered_motif_names.npy")
ids_clusters =  np.load("/home/parmida/orcd/pool/data/eps_0.38_871_clusters_cluster_labels.npy")


# path = "~/orcd/pool/data/genomic_features/CTCF_K562_peaks_ENCFF519CXF.bed"
path = "~/orcd/pool/data/genomic_features/ATAC_K562_thresholded_peaks_ENCFF948AFM.bed"
peaks_df = pd.read_csv(path, sep="\t",header=None).iloc[:,:3]#, comment="#",  names=columns[:len(pd.read_csv(bed_file, sep="\t", nrows=1).columns)])
peaks_df.columns = ["chrom", "start_bp", "end_bp"]
chr_peaks = peaks_df[peaks_df['chrom']==f"chr{chr_num}"]
chr_peaks = chr_peaks.drop_duplicates()
chr_peaks.sort_values(by='start_bp')




my_model = VGG14_1D( input_encoding = 'ours', 
                     our_feat_num = 871, num_tracks=params['num_tracks'], 
                     dropout = params['dropout'], n_conv = 11).to(device, dtype)
    
model_dict = my_model.state_dict()
pretrained_dict = torch.load(f"experiment_logs/exp_{my_exp_num}/model_params_epch_{my_epoch}.pt", map_location=device)
# Filter out unnecessary keys
pretrained_dict = {key.replace('module.', '', 1): value for key, value in pretrained_dict.items()}
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
my_model.load_state_dict(pretrained_dict)
my_model.eval()

enformer = VGG14_1D(input_encoding = 'onehot', 
                 our_feat_num = params['num_feats'], num_tracks=params['num_tracks'], 
                 dropout = params['dropout'], n_conv = 11).to(device, dtype)


model_dict = enformer.state_dict()
pretrained_dict = torch.load(f"experiment_logs/exp_{baseline_exp_num}/model_params_epch_{baseline_epoch}.pt", map_location=device)
pretrained_dict = {key.replace('module.', '', 1): value for key, value in pretrained_dict.items()}
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
enformer.load_state_dict(pretrained_dict)
enformer.eval()


top_ids_inds = []

for peak_start in tqdm(sorted(chr_peaks.start_bp)):
    bp_start = peak_start - 2000
    
    bin_start = np.searchsorted(bb, bp_start, side='right') - 1
    bin_end = bin_start + input_window
    bin_mid = (bin_start+bin_end)//2
    
    
  
    peak_intervals = [(bin_start, bin_end)]
    nonpeak_intervals = [(bin_start-2000, bin_end-2000), 
                         (bin_start+2000, bin_end+2000),
                         (bin_start-4000, bin_end-4000), 
                         (bin_start+4000, bin_end+4000)]
    
    
    
    chr_dataset_peak = get_chromosome_dataset('ours',
                                        chr_num=chr_num, 
                                        params=params,
                                        logger=Logger(f'val_dataset.log'), 
                                        data_augmentation = False,
                                        keep_only_peaks=False, 
                                        custom_intervals = peak_intervals,
                                        testing_mode=False)
    
    chr_dataset_onehot_peak = get_chromosome_dataset('onehot',
                                        chr_num=chr_num, 
                                        params=params,
                                        logger=Logger(f'val_dataset.log'), 
                                        data_augmentation = False,
                                        keep_only_peaks=False, 
                                        custom_intervals = peak_intervals,
                                        testing_mode=False)
    
    chr_dataset_background = get_chromosome_dataset('ours',
                                        chr_num=chr_num, 
                                        params=params,
                                        logger=Logger(f'val_dataset.log'), 
                                        data_augmentation = False,
                                        keep_only_peaks=False, 
                                        custom_intervals = nonpeak_intervals,
                                        testing_mode=False)
    
    chr_dataset_onehot_background = get_chromosome_dataset('onehot',
                                        chr_num=chr_num, 
                                        params=params,
                                        logger=Logger(f'val_dataset.log'), 
                                        data_augmentation = False,
                                        keep_only_peaks=False, 
                                        custom_intervals = nonpeak_intervals,
                                        testing_mode=False)
    
    in_seq_ours = torch.tensor(chr_dataset_peak[0][0]).unsqueeze(0).to(device,dtype)
    ref_seq_ours = torch.tensor(chr_dataset_background[0][0]).unsqueeze(0).to(device,dtype)
    in_seq_baseline = torch.tensor(chr_dataset_onehot_peak[0][0]).unsqueeze(0).to(device,dtype)
    ref_seq_baseline = torch.tensor(chr_dataset_onehot_background[0][0]).unsqueeze(0).to(device,dtype)
    
    
    
    wrapped_model_baseline = ForwardWrapper(enformer)
    wrapped_model_baseline.eval()
    dl_baseline = DeepLift(wrapped_model_baseline)
    attributions_baseline = dl_baseline.attribute(in_seq_baseline, baselines = ref_seq_baseline)
    attributions_baseline = attributions_baseline.cpu().detach().numpy()
    # np.save(f"attribution_scores/DeepLIFTattr_baseline_{baseline_exp_num}_chr{chr_num}_bb{bp_start}_window_{input_window}.npy", attributions_baseline)
    # print("saved baseline attributions")
    
    wrapped_model_ours = ForwardWrapper(my_model)
    wrapped_model_ours.eval()
    dl_ours = DeepLift(wrapped_model_ours)
    attributions_ours = dl_ours.attribute(in_seq_ours, baselines = ref_seq_ours)
    attributions_ours = attributions_ours.cpu().detach().numpy()
    attributions_ours /= np.abs(attributions_ours).max()
    top_ids_inds.append(attributions_ours[0])
    
    # np.save(f"attribution_scores/DeepLIFTattr_ours_{my_exp_num}_chr{chr_num}_bb{bp_start}_window_{input_window}.npy", attributions_ours)
    # print("saved our attributions")
   
    # ctcf_hits.append(is_ctcf_hit(attributions_ours))


# np.save(f"ctcf_hits_VGG_{my_exp_num}
    
    
    

############################## plotting the results #####################
    if plot:
        x_peak = torch.tensor(chr_dataset_peak[0][0]).unsqueeze(0).to(device, dtype)
        y_peak = chr_dataset_peak[0][1]
        our_pred_peak = my_model(x_peak).cpu().detach().numpy()[0]
        x_peak_onehot = torch.tensor(chr_dataset_onehot_peak[0][0]).unsqueeze(0).to(device, dtype)
        y_peak_onehot = chr_dataset_onehot_peak[0][1]
        onehot_pred_peak = enformer(x_peak_onehot).cpu().detach().numpy()[0]
        
        attributions_baseline/=np.abs(attributions_baseline).max()
        max_idx = np.where(np.abs(attributions_baseline)[0]==1)[0][0]
        baseline_peak_attributions = attributions_baseline[0,max_idx-30:max_idx+30,:]
        df = pd.DataFrame(baseline_peak_attributions, columns=['A', 'T', 'C', 'G'])
        
        
        attributions_ours /= np.abs(attributions_ours).max()#bin_start:bin_end
        max_attr_bin = np.where(np.abs(attributions_ours)[0]==1)[0][0]
        our_peak_attributions = attributions_ours[0,max_attr_bin-peak_window//2:max_attr_bin+peak_window//2,:]
        attr_thres = 0.75
        max_bin_ids, max_motif_ids = np.where(np.abs(our_peak_attributions)>attr_thres)[0], np.where(np.abs(our_peak_attributions)>attr_thres)[1]
        
        
        # Create 4 stacked subplots with the bottom one split into heatmap + colorbar
        fig = plt.figure(figsize=(10, 20))
        gs = gridspec.GridSpec(
            4, 2,
            height_ratios=[1, 1, 1, 8],  # ours, onehot, logo, heatmap
            width_ratios=[20, 1],        # main plots + colorbar
            hspace=0.2, wspace=0.05
        )
        
        # Top prediction plots
        ax_ours = fig.add_subplot(gs[0, 0])
        ax_onehot = fig.add_subplot(gs[1, 0])
        
        ax_ours.plot(y_peak, label='gt')
        ax_ours.plot(our_pred_peak, label='pred')
        ax_ours.set_ylim(0, 2)
        ax_ours.set_xticks([])
        ax_ours.set_title("Ours")
        ax_ours.legend()
        
        ax_onehot.plot(y_peak_onehot, label='gt')
        ax_onehot.plot(onehot_pred_peak, label='pred')
        ax_onehot.set_ylim(0, 2)
        ax_onehot.set_title("One-hot")
        ax_onehot.legend()
        
        # Shared x-axis labels on bottom
        x_ticks = np.array([
            bin_mid - input_window//2,
            bin_mid,
            bin_mid + input_window//2
        ])
        x_tick_positions = x_ticks - x_ticks[0]
        x_tick_labels = [str(x) + f"\n({bb[x]})" for x in x_ticks]
        
        ax_onehot.set_xticks(x_tick_positions)
        ax_onehot.set_xticklabels(x_tick_labels, rotation=0)
        ax_onehot.set_xlabel("Bin position\n(bp position)")
        
        # Attribution logo
        ax_logo = fig.add_subplot(gs[2, 0])
        logomaker.Logo(df, color_scheme='classic', ax=ax_logo)
        ax_logo.set_title("Attribution Logo")
        ax_logo.set_ylabel("Attribution Score")
        ax_logo.set_xticks([])
        
        # Heatmap and colorbar
        ax_heatmap = fig.add_subplot(gs[3, 0])
        ax_cbar = fig.add_subplot(gs[3, 1])
        
        sns_heatmap = sns.heatmap(
            our_peak_attributions.T,
            ax=ax_heatmap,
            vmin=-1, vmax=1,
            cmap='vlag',
            cbar=False
        )
        
        cbar = fig.colorbar(sns_heatmap.get_children()[0], cax=ax_cbar)
        cbar.ax.set_ylabel("Attribution", rotation=-90, va="bottom")
        
        # Optional annotations
        for (max_bin_idx, max_motif_idx) in zip(max_bin_ids, max_motif_ids):
            ax_heatmap.text(
                max_bin_idx + 2,
                max_motif_idx,
                f"{motif_names[np.where(ids_clusters == max_motif_idx)[0]]}"
            )
            print(motif_names[np.where(ids_clusters == max_motif_idx)[0]])
        
        ax_heatmap.set_ylabel("Motif features")
        ax_heatmap.set_xlabel("Genomic bin \n(bp position)")
        ax_heatmap.set_yticks(np.arange(0, 872, 100))
        ax_heatmap.set_yticklabels(np.arange(0, 872, 100))
        
        # Shared x-ticks for heatmap
        first_bin_plotted = bin_start + max_attr_bin - peak_window//2
        x_ticks_hm = first_bin_plotted + np.array([max_attr_bin - peak_window//2, max_attr_bin, max_attr_bin + peak_window//2])
        x_tick_positions_hm = x_ticks_hm - x_ticks_hm[0]  # align with predictions
        x_tick_labels_hm = [str(x) + f"\n({bb[x]})" for x in x_ticks_hm]
        ax_heatmap.set_xticks(x_tick_positions_hm)
        ax_heatmap.set_xticklabels(x_tick_labels_hm, rotation=0)
        
        plt.suptitle(f"Chromosome {chr_num}")
        # plt.tight_layout()
        plt.savefig(f"../figures/VGG_attributions_chr{chr_num}_bb{bp_start}_window_{input_window}.png")
        print("saved plots")

top_ids_inds = np.array(top_ids_inds)
print(top_ids_inds.shape)
np.save( f"ATAC_peak_attributions_chr{chr_num}.npy", top_ids_inds)
print(f"chromosome {chr_num} total ATAC peaks: {len(top_ids_inds)}")
# print("proportion of regions with ctcf in 3rd top 3 hit: ", np.mean(ctcf_hits==3))
# print("proportion of regions with ctcf in 2nd top 2 hit: ", np.mean(ctcf_hits==2))
# print("proportion of regions with ctcf in top hit: ", np.mean(ctcf_hits==1))


print("done")