import math
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from time import time
from utils import *
import pysam
from profiler import Logger
from data_loader import *
from model import *
import glob
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
with open('training_params.yaml', 'r') as f:
    params = yaml.safe_load(f)




chr_num=10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
epoch = 10
# Transformer
# onehot_exp_num=  65468595 
# my_exp_num= 65468582   

## VGG
my_exp_num=65889195
onehot_exp_num=65917496

max_metric = 'recall'
crop_size = params['crop_size']


track_to_range={
    'CTCF': (0,5),
    'H3K27ac':(0,5),
    'H3K9ac':(0,5),
    'RAD21':(0,5),
    'H3K4me1':(0,5),
    'H3K9me3':(0,5),
    'H3K4me3':(0,5),
    'H3K27me3':(0,5),
    'DNASE':(0,1),
    'ATAC':(0,2)}
track_names = sorted(list(track_to_range.keys()))


### Load the validation dataset
chr_dataset = get_chromosome_dataset('ours',
                                    chr_num=chr_num, 
                                    params=params,
                                    logger=Logger(f'val_dataset.log'), 
                                    data_augmentation=False, 
                                    keep_only_peaks=False, 
                                    testing_mode=False)

our_dataloader = DataLoader(
        chr_dataset,
        batch_size=1,#total_batch_size//world_size = batch_size per GPU
        shuffle=False,
        num_workers=4, #num_workers per GPU
        prefetch_factor = 2,
        pin_memory=True)

chr_dataset_onehot = get_chromosome_dataset('onehot',
                                    chr_num=chr_num, 
                                    params=params,
                                    logger=Logger(f'val_dataset.log'), 
                                    data_augmentation=False, 
                                    custom_intervals = chr_dataset.bin_intervals,
                                    keep_only_peaks=False, 
                                    testing_mode=False)

onehot_dataloader = DataLoader(
        chr_dataset_onehot,
        batch_size=1,#total_batch_size//world_size = batch_size per GPU
        shuffle=False,
        num_workers=4, #num_workers per GPU
        prefetch_factor = 2,
        pin_memory=True)

assert len(our_dataloader) == len(onehot_dataloader)
print("dataset length: ", len(our_dataloader))

### Load the models
my_model = VGG14_1D( input_encoding = 'ours', 
                     our_feat_num = 871, num_tracks=params['num_tracks'], 
                     dropout = params['dropout'], n_conv = 11).to(device, dtype)


# my_model = My_transformer( input_encoding='ours',our_feat_num = params['num_feats'], num_tracks=params['num_tracks'], 
#                  dropout = params['dropout']).to(device, dtype)

model_dict = my_model.state_dict()
pretrained_dict = torch.load(f"experiment_logs/exp_{my_exp_num}/model_params_epch_{epoch}.pt", map_location=device)
pretrained_dict = {key.replace('module.', '', 1): value for key, value in pretrained_dict.items()}
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
my_model.load_state_dict(pretrained_dict)
my_model.eval()


# enformer = My_transformer( input_encoding='onehot',our_feat_num = params['num_feats'], num_tracks=params['num_tracks'], 
#                  dropout = params['dropout']).to(device, dtype)

enformer = VGG14_1D(input_encoding = 'onehot', 
                 our_feat_num = params['num_feats'], num_tracks=params['num_tracks'], 
                 dropout = params['dropout'], n_conv = 11).to(device, dtype)


model_dict = enformer.state_dict()
pretrained_dict = torch.load(f"experiment_logs/exp_{onehot_exp_num}/model_params_epch_{epoch}.pt", map_location=device)
pretrained_dict = {key.replace('module.', '', 1): value for key, value in pretrained_dict.items()}
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
enformer.load_state_dict(pretrained_dict)
enformer.eval()



for track_idx in range(1):
    track_name = track_names[track_idx]
    our_metric_list = []
    onehot_metric_list = []
    
    our_corrs = []
    onehot_corrs = []
    for (our_batch, onehot_batch) in zip(our_dataloader, onehot_dataloader):
        our_features, our_tracks,our_bb, _,_ = our_batch
        onehot_features, onehot_tracks,onehot_bb,_,_ = onehot_batch
        
        our_pred = 4*my_model(our_features.to(device, dtype))[:,:,track_idx].detach().cpu().numpy().reshape(-1)#, our_bb.to(device,dtype)
        onehot_pred = 4*enformer(onehot_features.to(device, dtype))[:,:,track_idx].detach().cpu().numpy().reshape(-1)#, onehot_bb.to(device,dtype)
        #currently, all tracks in a batch are flattened: may need to adjust
        our_target = 4*our_tracks[:,:,track_idx].numpy().reshape(-1)
        onehot_target = 4*onehot_tracks[:,:,track_idx].numpy().reshape(-1)

        our_corrs.append(evaluate_correlation(our_pred, our_target))
        onehot_corrs.append(evaluate_correlation(onehot_pred, onehot_target))    
     
        
        #TO DO: aggregate the different regions together and handle overlap
        # params_dict = get_peakcalling_params_for_max_metric(metric = max_metric, binning='ourbins', track_name = track_name)
        params_dict = {"gaussian_sigma":1,
                       "noise_percentile": 85,
                       "min_peak_height": 0.9,
                       "peak_dist": 6,
                       "prom": 0.2,
                       "max_peak_width": 120}
        our_target_binary = binarize_track_with_peaks(our_target, **params_dict)
        
        our_pred_binary = binarize_track_with_peaks(our_pred, **params_dict)
        onehot_target_binary = binarize_track_with_peaks(onehot_target, **params_dict)
        onehot_pred_binary = binarize_track_with_peaks(onehot_pred, **params_dict)
        print(np.unique(our_target_binary), np.unique(our_pred_binary))
        
        our_metric_list.append(evaluate_binary_predictions(our_pred_binary, our_target_binary))
        onehot_metric_list.append(evaluate_binary_predictions(onehot_pred_binary, onehot_target_binary))


    our_metrics = convert_to_dict_of_lists(our_metric_list) #metric: list
    onehot_metrics = convert_to_dict_of_lists(onehot_metric_list)

    with open(f'../data_for_figures/{my_exp_num}_chr{chr_num}_binary_peak_metrics.pkl', 'wb') as f:
        pickle.dump(our_metrics, f)
    with open(f'../data_for_figures/{onehot_exp_num}_chr{chr_num}_binary_peak_metrics.pkl', 'wb') as f:
        pickle.dump(onehot_metrics, f)
    np.save(f'../data_for_figures/{my_exp_num}_chr{chr_num}_our_correlations.npy', our_corrs)
    np.save(f'../data_for_figures/{onehot_exp_num}_chr{chr_num}_onehot_correlations.npy', onehot_corrs)
    
    
    """"plotting"""
    """
    metric_names = ["AUC","AUPR","MCC","acc", "recall", "peak_level_recall", "peak_level_precision","precision", "specificity"]#list(our_metrics.keys())  # Extract keys

         
    # Prepare the data for plotting
    data = []
    categories = []  # To hold "Ours" or "onehot" labels
    metrics = []  # To hold metric names for labeling

    # Combine the "ours" and "onehot" values for each metric
    for i, metric in enumerate(metric_names):
        ours = [v for v in our_metrics[metric] if v is not None]  # Remove None values
        onehots = [v for v in onehot_metrics[metric] if v is not None]  # Remove None values

        if not ours or not onehots:  # Skip if data is missing
            continue

        # Create data for "Ours"
        data.extend(ours)
        categories.extend(["Ours"] * len(ours))
        metrics.extend([metric] * len(ours))

        # Create data for "onehot"
        data.extend(onehots)
        categories.extend(["Onehot"] * len(onehots))
        metrics.extend([metric] * len(onehots))

    # Create a DataFrame for easier manipulation and plotting
    df = pd.DataFrame({
        'Metric': metrics,
        'Value': data,
        'Category': categories
    })

    # Create split violin plots
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Metric', y='Value', hue='Category', data=df, gap=0.1,
                palette={"Ours": "blue", "onehot": "orange"})

    # Adjust appearance
    plt.title(track_name)
    plt.ylabel("Metric Value")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(title='Category', frameon=False)

    plt.title(track_name)
    plt.savefig(f"eval_results/transformer_binary_metrics_{onehot_exp_num}_{my_exp_num}_epoch{epoch}_chr_{chr_num}.png")
    print("Finished!")

""" 
#### PLOTTING CORRELATION
"""
corrs = []
labels = []
tracks = []

for (our_batch, onehot_batch) in zip(our_dataloader, onehot_dataloader):
    our_features, our_tracks,our_bb = our_batch
    onehot_features, onehot_tracks,onehot_bb = onehot_batch
    
    our_pred_all = my_model(our_features.to(device, dtype), our_bb.to(device,dtype))
    onehot_pred_all = enformer(onehot_features.to(device, dtype), onehot_bb.to(device,dtype))
    
    for track_idx in range(1):
        track_name = track_names[track_idx]
        our_target = our_tracks[:,crop_size:-crop_size,track_idx].numpy().reshape(-1)
        onehot_target = onehot_tracks[:,crop_size:-crop_size,track_idx].numpy().reshape(-1)
        our_pred = our_pred_all[:,crop_size:-crop_size,track_idx].detach().cpu().numpy().reshape(-1)
        onehot_pred =  onehot_pred_all[:,crop_size:-crop_size,track_idx].detach().cpu().numpy().reshape(-1)

        our_corr = evaluate_correlation(our_pred, our_target)
        onehot_corr = evaluate_correlation(onehot_pred, onehot_target)       
     
    
        # Create data for "Ours"
        corrs.append(our_corr)
        labels.append("ours" )
        tracks.append(track_name)
        # Create data for "onehot"
        corrs.append(onehot_corr)
        labels.append("onehot" )
        tracks.append(track_name)

# Create a DataFrame for easier manipulation and plotting
df = pd.DataFrame({
    'tracks': tracks,
    'corrs': corrs,
    'labels': labels
})

# Create split violin plots
plt.figure(figsize=(10, 6))
sns.boxplot(x='tracks', y='corrs', hue='labels', data=df, gap=0.1,
            palette={"ours": "blue", "onehot": "orange"})

# Adjust appearance
plt.title(f"Chr {chr_num} correlations")
plt.ylabel("Correlation")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend(title='experiment', frameon=False)

plt.savefig(f"eval_results/transformer_evaluation_correlation_{onehot_exp_num}_{my_exp_num}_epoch{epoch}_chr_{chr_num}.png")


##histogram
plt.figure(figsize=(10, 6))


sns.histplot(data=df, x='corrs', hue='labels',
             palette={"ours": "blue", "onehot": "orange"},
             kde=True, bins=20)
plt.title(f'Chromosome {chr_num}')
plt.xlabel('Correlation')
plt.ylabel('Count')
plt.savefig(f"eval_results/transformer_hist_{onehot_exp_num}_{my_exp_num}_epoch{epoch}_chr_{chr_num}.png")

"""

print("Finished!")







