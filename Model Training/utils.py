import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
import pandas as pd
from scipy.ndimage import label
import torch.nn.functional as F
from functools import partial







file_name = '/home/parmida/TF_project/GeneToken/training_params.yaml'
with open(file_name, 'r') as f:
    data = yaml.load(f,Loader=yaml.FullLoader)

class ForwardWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)                # (B, 1250, 1)
        return out[:,data['crop_size']:-data['crop_size'],0].mean(dim=1)  # (B,)

def get_loss(loss_name, **kwargs):
    if loss_name == 'pnll':
        return torch.nn.PoissonNLLLoss(log_input=False)
    elif loss_name == 'pnll+multi':
        return partial(pnll_multinomial, **kwargs)

def pnll_multinomial(predictions, targets, poisson_weight = 1.0, multinomial_weight=4.0, eps=1e-6):
    """
    Borzoi loss with predictions shape: (batch_size, input_length, 1)
    Targets shape: (batch_size, input_length)
    """

    # Squeeze predictions from (B, L, 1) → (B, L)
    predictions = predictions.squeeze(-1)
    targets = targets.squeeze(-1)

    print()

    # Sum across sequence to get total counts per sample (B, 1)
    total_targets = targets.sum(dim=-1, keepdim=True) + eps
    total_preds = predictions.sum(dim=-1, keepdim=True) + eps
    # Poisson loss on total counts (shape: [batch])
    poisson_loss = F.poisson_nll_loss(
        predictions,
        targets,
        log_input=False,
        full=False,
        reduction='mean'
    )

    # Normalize to get probability distributions (B, L)
    targets_norm = targets / total_targets
    preds_norm = predictions / total_preds

    # Multinomial negative log-likelihood
    multinomial_loss = -torch.sum(
        targets_norm * torch.log(preds_norm + eps),
        dim=-1
    ).mean()
    return poisson_weight * poisson_loss + multinomial_weight * multinomial_loss


def stable_exp_nll_from_mu(mu, y,eps = 1e-6):
    log_mu = torch.log(mu+eps)
    return (log_mu + y * torch.exp(-log_mu)).mean()

def my_exponential_loss(y_pred, y_true, eps = 1e-4):
    return torch.mean(torch.log(y_pred+eps)+(y_true/(y_pred+eps)))

def get_collate_fn(input_type):
    if input_type == 'onehot':
        return collate_fn_onehot
    elif input_type == 'ours':
        return None
    else:
        raise ValueError(f"Unknown input type for collate function: {input_type}")

def get_PE_from_bb(batch_bb):
    batch_bb = np.array(batch_bb)
    block_size = data['block_size']
    n_embd = data['d_embd']
    #batch_bb is of size batch_sizexblock_size
    position = torch.tensor(batch_bb).unsqueeze(2)  # batch_size x N x 1
    div_term = torch.exp(torch.arange(0, n_embd, 2) * -0.015)  # (n_embd//2)
    pe = torch.zeros(batch_bb.shape[0], batch_bb.shape[1], n_embd)  # batch_size x N x n_embd
    
    pe[:, :, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
    pe[:, :, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
    return pe

def collate_fn_onehot(batch):
    sequences, tracks, pe = zip(*batch)
    
    # Find the maximum sequence length in the batch
    seq_max_length = data['block_size']*(data['output_bin_res'])
    
    # Pad sequences to the same length (max_length)
    padded_sequences = []
    for seq in sequences:
        # Create a tensor of size max_length x 4, either by padding or by truncating
        if seq_max_length <= seq.shape[0]:
            middle_idx = seq.shape[0] // 2
            start_idx = max(0, middle_idx - seq_max_length // 2)
            end_idx = start_idx + seq_max_length
            padded_sequence = torch.tensor(seq[start_idx:end_idx])
        else:
            pad_size = seq_max_length - seq.shape[0]
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left

            # Apply symmetric padding with zeros
            padded_sequence = torch.cat([torch.zeros(pad_left, seq.shape[1]), torch.tensor(seq), torch.zeros(pad_right, seq.shape[1])], dim=0)
        padded_sequences.append(padded_sequence)
    padded_sequences = torch.stack(padded_sequences)
    
    
    track_max_length = data['block_size']
    padded_tracks = []
    for track in tracks:
        if track_max_length <= track.shape[0]:
            middle_idx = track.shape[0] // 2
            start_idx = max(0, middle_idx - track_max_length // 2)
            end_idx = start_idx + track_max_length
            padded_track = torch.tensor(track[start_idx:end_idx])
        else:
            pad_size = track_max_length - track.shape[0]
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left

            # Apply symmetric padding with zeros
            padded_track = torch.cat([torch.zeros(pad_left, track.shape[1]), torch.tensor(track), torch.zeros(pad_right, track.shape[1])], dim=0)
        padded_tracks.append(padded_track)
    padded_tracks = torch.stack(padded_tracks)
    return padded_sequences, padded_tracks, pe

def collate_fn_onehot_multitask(batch):
    sequences, tracks, motifs = zip(*batch)
    
    # Find the maximum sequence length in the batch
    seq_max_length = data['block_size']*(2**data['n_conv'])
    
    # Pad sequences to the same length (max_length)
    padded_sequences = []
    for seq in sequences:
        # Create a tensor of size max_length x 4, either by padding or by truncating
        if seq_max_length <= seq.shape[0]:
            middle_idx = seq.shape[0] // 2
            start_idx = max(0, middle_idx - seq_max_length // 2)
            end_idx = start_idx + seq_max_length
            padded_sequence = torch.tensor(seq[start_idx:end_idx])
        else:
            pad_size = seq_max_length - seq.shape[0]
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left

            # Apply symmetric padding with zeros
            padded_sequence = torch.cat([torch.zeros(pad_left, seq.shape[1]), torch.tensor(seq), torch.zeros(pad_right, seq.shape[1])], dim=0)
        padded_sequences.append(padded_sequence)
    padded_sequences = torch.stack(padded_sequences)
    
    
    track_max_length = data['block_size']
    padded_tracks = []
    for track in tracks:
        if track_max_length <= track.shape[0]:
            middle_idx = track.shape[0] // 2
            start_idx = max(0, middle_idx - track_max_length // 2)
            end_idx = start_idx + track_max_length
            padded_track = torch.tensor(track[start_idx:end_idx])
        else:
            pad_size = track_max_length - track.shape[0]
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left

            # Apply symmetric padding with zeros
            padded_track = torch.cat([torch.zeros(pad_left, track.shape[1]), torch.tensor(track), torch.zeros(pad_right, track.shape[1])], dim=0)
        padded_tracks.append(padded_track)
    padded_tracks = torch.stack(padded_tracks)
    return padded_sequences, padded_tracks


def create_exp_directory(experiments_dir, job_id, rank):
    # Find all directories with a similar name
    new_directory_name = experiments_dir+f'/exp_{job_id}'
    if rank==0:
        if not os.path.exists(new_directory_name):
            os.makedirs(new_directory_name)
    return new_directory_name


def write_dict_to_yaml(dictionary, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(dictionary, file)

        
def get_avg_correlation_per_track(target, pred):
    track_corrs = []
    for j in range(target.shape[2]):
        corrs = []
        for i in range(pred.shape[0]):
            if torch.all(target[i,:,j]==0).item() and torch.all(pred[i,:,j]==0).item():
                corrs.append(1)
            if torch.all(target[i,:,j]==0).item() or torch.all(pred[i,:,j]==0).item():
                corrs.append(np.nan)
            else:
                corrs.append(np.corrcoef(target[i,:,j].detach().cpu().numpy(), pred[i,:,j].detach().cpu().numpy())[0,1])
        track_corrs.append(np.nanmean(corrs))
    return track_corrs

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


# CTCF	200–600 bp	Transcription factor (TF) that binds insulators; sharp peaks.
# DNase	100–300 bp	DNase hypersensitive sites indicate open chromatin; very sharp peaks.
# H3K27ac	500–2000 bp	Active enhancers and promoters; broad peaks.
# H3K27me3	1000–5000+ bp	Repressive heterochromatin mark; very broad peaks.
# H3K4me1	500–2000 bp	Primed enhancers; broad peaks.
# H3K4me3	400–1000 bp	Active promoters; broad peaks.
# H3K9ac	500–1500 bp	Active chromatin, associated with transcriptional activation.
# H3K9me3	1000–10,000+ bp	Heterochromatin mark; extremely broad peaks.
# RAD21	200–600 bp	Cohesin complex subunit; forms sharp peaks at CTCF sites.
track_to_peak_width_bp={
    'CTCF': (200,600),  
    'H3K27ac':(500,2000), #few hundred to couple of Kb: 300-3000
    'H3K9ac':(500,1500),
    'RAD21':(200,600),
    'H3K4me1':(500,2000),
    'H3K9me3':(400,1000),
    'H3K4me3':(1000,15000),
    'H3K27me3':(1000,5000),
    'DNASE':(100,300),
    'ATAC':(50,800)}

def plot_outputs(model, chr_dataset, epoch, dataset_idx, exp_dir, append_to_title = ""):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = next(model.parameters()).dtype
    x = torch.unsqueeze(torch.tensor(chr_dataset[dataset_idx][0]),dim = 0).to(device, dtype)
    y = chr_dataset[dataset_idx][1]#.to( device, dtype)
    bb = torch.unsqueeze(torch.tensor(chr_dataset[dataset_idx][2]),dim = 0).to(device, dtype) #batch_bb needs to be of shape Bxblock_size
    
    
    model.eval()
    with torch.no_grad():
        out = model(x)
    fig = plt.figure(figsize = (12,len(chr_dataset.track_names)))

    for i,track_name in enumerate(chr_dataset.track_names):
        plt.subplot(len(chr_dataset.track_names),1,i+1)
        plt.plot(np.arange(out.shape[1]), out[0,:,i].detach().cpu().numpy(), color = 'orange',label = f"pred epoch {epoch}")

        plt.plot(np.arange(y.shape[0]), y[:,i], color = 'green',label = "ground truth", alpha = 0.5)
        plt.ylabel(track_name)
        plt.ylim(track_to_range[track_name])

        
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0))
    plt.suptitle(f" bins {chr_dataset.bin_intervals[dataset_idx]} _ {append_to_title}")
    plt.savefig(exp_dir+f"/out_tracks_{dataset_idx}.png")
    model.train()
    return fig

def evaluate_combined(criterion, model, dataloader, crop, num_tracks):
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    with torch.no_grad():
        val_loss = 0
        val_pred = None
        val_y = None
        num_batches = 0
        
        for data in dataloader:
            x_ours, x_onehot, bb_ours, bb_onehot, y = data
            x_ours, x_onehot = x_ours.to(device, dtype), x_onehot.to(device, dtype)
            bb_ours, bb_onehot = bb_ours.to(device, dtype), bb_onehot.to(device, dtype)
            y = y.to(device, dtype)
            
            predicted_tracks = model(x_onehot, x_ours, bb_onehot, bb_ours)
            val_loss += criterion(predicted_tracks[:,crop:-crop,:], y[:,crop:-crop,:]).item()
            
            if val_pred is None:
                val_pred = predicted_tracks[:,crop:-crop,:].cpu()
                val_y = y[:,crop:-crop,:].cpu()
            else:
                val_pred = torch.cat([val_pred, predicted_tracks[:,crop:-crop,:].cpu()], dim=0)
                val_y = torch.cat([val_y, y[:,crop:-crop,:].cpu()], dim=0)
            
            num_batches += 1
            
        val_track_corrs = get_avg_correlation_per_track(val_y, val_pred)
        
    model.train()
    return val_loss/num_batches, val_track_corrs, val_pred, val_y

def get_binary_metrics_per_track(y_true, y_pred, y_true_thresh = 0.5):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    
    sorted_names = sorted(list(track_to_range.keys()))
    acc=[]
    TP=[]
    FP=[]
    TN=[]
    FN=[]
    auc = []
    for i,track in enumerate(sorted_names):
        t = y_true_thresh*track_to_range[track][1] 
        y_true_binary = y_true[:,:,i]>t
        y_pred_binary = y_pred[:,:,i]>t
    
        acc.append(np.mean(y_true_binary==y_pred_binary))
        TP.append(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
        FP.append(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
        TN.append(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
        FN.append(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))
        auc.append(roc_auc_score())
    
        
    return {'acc':acc , 'TP': TP, 'FP': FP, 'TN':TN, 'FN':FN}
    
    
    
    
    
    
    
@torch.no_grad()
def evaluate(loss_func, model,  val_dataloader,crop_size, num_tracks):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = next(model.parameters()).dtype
    model.eval()
 
    
    #validation loss:
    losses = torch.zeros(len(val_dataloader))
    corrs = np.zeros((len(val_dataloader),num_tracks))
    for it, data in enumerate(val_dataloader):
        X, Y, bb =  data
        X, Y, bb = X.to(device,dtype), Y.to(device, dtype), bb.to(device,dtype)

        predicted = model(X)
        loss = loss_func(predicted[:,crop_size:-crop_size,:], Y[:,crop_size:-crop_size,:])
        track_corrs = get_avg_correlation_per_track(Y[:,crop_size:-crop_size,:], predicted[:,crop_size:-crop_size,:])
        losses[it] = loss.item()
        corrs[it,:]=track_corrs
    model.train()
    return losses.mean(), np.nanmean(corrs,axis = 0),predicted, Y   #out


def run_evaluation(epoch, val_losses, logger, this_exp_dir, wandb_logger, tb_writer, loss_func, model,  val_dataloader,crop_size, num_tracks):
    val_loss, val_track_corrs, val_pred, val_y = evaluate(
        loss_func, model, val_dataloader, crop_size, num_tracks
    )
    logger.log(f"[Eval] Epoch {epoch} | Loss: {val_loss}")
    val_losses.append(val_loss)
    np.save(f"{this_exp_dir}/val_loss.npy", val_losses)
    wandb_logger.log({"val_loss": val_loss})
    wandb_logger.log({"val_corr": np.nanmean(val_track_corrs)})
    tb_writer.add_scalars("Loss", {"val": val_loss }, epoch) 
    tb_writer.add_scalars("Correlation", {"val": np.nanmean(val_track_corrs) }, epoch) 
    return


@torch.no_grad()
def evaluate_multitask(loss_func_tracks, loss_func_motifs, model,  val_dataloader,crop_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = next(model.parameters()).dtype
    model.eval()
 
    
    #validation loss:
    track_losses = torch.zeros(len(val_dataloader))
    motif_losses = torch.zeros(len(val_dataloader))
    corrs = np.zeros((len(val_dataloader),10))
    for it, data in enumerate(val_dataloader):
        X, Y, motifs =  data
        X, Y, motifs = X.to(device,dtype), Y.to(device, dtype), motifs.to(device,dtype)
        predicted_tracks, predicted_motifs = model(X) 
        track_loss = loss_func_tracks(predicted_tracks[:,crop:-crop,:], Y[:,crop:-crop,:])
        motif_loss = loss_func_motifs(predicted_motifs[:,crop:-crop,:], motifs[:,crop:-crop,:])
        loss =  track_loss + params['lambda']*motif_loss
        track_corrs = get_avg_correlation_per_track(Y[:,crop_size:-crop_size,:], predicted_tracks[:,crop_size:-crop_size,:])
        track_losses[it] = track_loss.item()
        motif_losses[it] = motif_loss.item()
        corrs[it,:]=track_corrs
    model.train()
    return track_losses.mean(), motif_losses.mean(), np.nanmean(corrs,axis = 0),predicted_tracks, predicted_motifs, Y   #out


@torch.no_grad()
def get_max_grad(model):
    max_grads = []
    for n, p in model.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            try:
                max_grads.append(p.grad.abs().max())
            except:
                print(n, p)
                raise ValueError
    return max(max_grads).item()



def plot_outputs_multitask(model, chr_dataset, epoch, dataset_idx, exp_dir, append_to_title = ""):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = next(model.parameters()).dtype
    x = torch.unsqueeze(torch.tensor(chr_dataset[dataset_idx][0]),dim = 0).to(device, dtype)
    y = chr_dataset[dataset_idx][1]#.to( device, dtype)
    
    
    model.eval()
    with torch.no_grad():
         #Enables autocasting (data type) for the forward pass (model + loss)
        out,motifs = model(x)
    fig = plt.figure(figsize = (10,13))

    i = 0
    for filename in sorted(os.listdir(chr_dataset.output_directory)):
        name = chr_dataset.output_directory+filename
        if name.endswith("bigWig"):# and filename.startswith("DNASE_K562_ENCFF413AHU"):
            track_name = filename.split("_")[0]
            plt.subplot(len(os.listdir(chr_dataset.output_directory)),1,i+1)
            plt.plot(np.arange(y.shape[0]), out[0,:,i].detach().cpu().numpy(), color = 'orange',label = f"pred epoch {epoch}")
            
            plt.plot(np.arange(y.shape[0]), y[:,i], color = 'green',label = "ground truth", alpha = 0.5)
            plt.ylabel(track_name)
            plt.ylim(track_to_range[track_name])

            i+=1
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0))
    plt.suptitle(f" bins {chr_dataset.bin_intervals[dataset_idx]} _ {append_to_title}")
    plt.savefig(exp_dir+f"/out_tracks_{dataset_idx}.png")
    model.train()
    return fig


@torch.no_grad()
def get_grad_max_min(model):
    grad_maxes = []
    grad_mins = []
    for n, p in model.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            try:
                grad_maxes.append(p.grad.max().item())
                grad_mins.append(p.grad.min().item())
            except:
                print(n, p)
                raise ValueError
    return max(grad_maxes), min(grad_mins)

    
def compute_weights_stats(model):
    """
    Computes the mean and standard deviation of the weights in each layer of the model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        dict: A dictionary where keys are layer names, and values are (mean, std) tuples.
    """
    stats = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:  # Ensure we only compute stats for trainable parameters
            # print(name, param.shape)
            param_mean = torch.mean(param).item()
            param_std = torch.std(param).item()
            stats[name] = (param_mean, param_std)
    
    return stats



"""
applies peak calling and returns a binary version of the track using those peaks
"""
def binarize_track_with_peaks(binned_track,
               gaussian_sigma = 1,
               noise_percentile = 75,
               min_peak_height = 0.7, #minimum peak height relative to noise
               peak_dist = 10, #minimum distance between two consecutive peaks
               prom = 0.5,
               max_peak_width = 40 #max peak width, this must be adjusted based on each track given the bed files
              ):

    if peak_dist<1: peak_dist = 1
    smoothed_signal = gaussian_filter1d(binned_track.astype(np.float32), sigma=gaussian_sigma)

    # Adaptive thresholding: Set height dynamically based on background noise
    background_noise = np.percentile(smoothed_signal, noise_percentile)  # 75th percentile as background level
    min_peak_height = background_noise + min_peak_height  # Ensure peaks are well above noise

    peaks, properties = find_peaks(
    smoothed_signal,
    height=min_peak_height,  # Ensures peaks are above background
    distance=peak_dist,  # ensures peaks are not too close
    prominence=prom,  # Higher prominence to filter out noise
    width=(1, max_peak_width)  # Ensure peaks are not too broad
    )
    left_bases = np.floor(peaks - properties['widths']/2).astype(int)
    right_bases = np.ceil(peaks + properties['widths']/2).astype(int)


    mask = np.zeros_like(binned_track, dtype=int)
    if len(right_bases)<1:
        return mask
    else:
        # Set the regions between peak_left_base and peak_right_base to 1
        for left, right in zip(left_bases, right_bases):
            mask[left:right + 1] = 1  # Include the right boundary
    
    return mask



def evaluate_binary_predictions(pred_track_binary, gt_track_binary):
    """
    Computes AUC, AUPR, Matthews Correlation Coefficient, TP, FP, TN, FN rates 
    given two binary arrays: predicted and ground truth tracks.

    Parameters:
    - pred_track_binary (1D np.array): Predicted binary values (0 or 1)
    - gt_track_binary (1D np.array): Ground truth binary values (0 or 1)

    Returns:
    - results (dict): Dictionary containing AUC, AUPR, MCC, TP, FP, TN, FN, and their rates
    """
    # Convert to numpy arrays (ensure they are binary)
    pred = np.array(pred_track_binary)
    gt = np.array(gt_track_binary)

    # Compute confusion matrix components
    TP = np.sum((pred == 1) & (gt == 1))  # True Positives
    FP = np.sum((pred == 1) & (gt == 0))  # False Positives
    TN = np.sum((pred == 0) & (gt == 0))  # True Negatives
    FN = np.sum((pred == 0) & (gt == 1))  # False Negatives
    acc = np.mean(pred==gt)

    # Compute rates
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity / Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    specificity = TN / (TN + FP)

    #computes the fraction of gt peaks that overlap with at least one called peak
    peak_level_recall = compute_peak_overlap(gt, pred) 
    peak_level_precision = compute_peak_overlap(pred, gt) 
    
    # Compute F1 Score
    F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


    # Compute metrics
    AUC = roc_auc_score(gt, pred) if len(np.unique(gt)) > 1 else None  # Avoid error if only one class
    AUPR = average_precision_score(gt, pred) if len(np.unique(gt)) > 1 else None
    MCC = matthews_corrcoef(gt, pred)

    # Return results as a dictionary
    return {
        "AUC": AUC,
        "AUPR": AUPR,
        "MCC": MCC,
        "acc": acc,
        "recall": recall,
        "peak_level_recall": peak_level_recall,
        "peak_level_precision": peak_level_precision,
        "precision": precision,
        "specificity": specificity,
        "F1": F1
    }


def evaluate_correlation(pred_track, gt_track):
      # Convert to numpy arrays 
    pred = np.array(pred_track)
    gt = np.array(gt_track)
    if np.all(gt==0) and np.all(pred==0):
        corr = 1
    if np.all(gt==0) or np.all(pred==0): #if only one of them is all 0
        corr = np.nan
    else:
        corr = np.corrcoef(gt, pred)[0,1]
     
    return corr

def convert_to_dict_of_lists(dict_list):
    """
    Converts a list of dictionaries (each with single values per key) 
    into a dictionary of lists, where each key maps to a list of values.

    Parameters:
    - dict_list (list of dicts): List of dictionaries with the same keys.

    Returns:
    - dict_of_lists (dict): Dictionary where each key maps to a list of values.
    """
    dict_of_lists = defaultdict(list)

    for d in dict_list:
        for key, value in d.items():
            dict_of_lists[key].append(value)

    return dict(dict_of_lists)  # Convert back to a regular dictionary


def get_peakcalling_params_for_max_metric(metric = 'recall', binning='ourbins', track_name = 'CTCF'):
    df = pd.read_csv(f"eval_results/{track_name}_{binning}_peak_calling_param_sweep.csv")
    return dict(df.loc[df[metric].idxmax()].iloc[:6])


def compute_peak_overlap(true_peaks, detected_peaks):
    """
    Computes the fraction of true peak regions that overlap with at least one detected peak.

    Parameters:
    - true_peaks: 1D NumPy binary array (1s where true peaks exist, 0s elsewhere).
    - detected_peaks: 1D NumPy binary array (1s where detected peaks exist, 0s elsewhere).

    Returns:
    - Fraction of true peak regions that overlap with detected peaks.
    """
    labeled_true_peaks, num_true_regions = label(true_peaks)  # Label contiguous true peaks
    overlapping_regions = np.unique(labeled_true_peaks[detected_peaks.astype(bool)])  # Find regions with overlap
    num_overlapping = np.count_nonzero(overlapping_regions)  # Count unique overlapping regions

    return num_overlapping / num_true_regions if num_true_regions > 0 else 0.0

def get_intervals_at_peaks(chr_num,bed_file, block_size, bb):
    peaks_df = pd.read_csv(bed_file, sep="\t",header=None).iloc[:,:3]#, comment="#",  names=columns[:len(pd.read_csv(bed_file, sep="\t", nrows=1).columns)])
    peaks_df.columns = ["chrom", "start_bp", "end_bp"]
    chr_peaks = peaks_df[peaks_df['chrom']==f"chr{chr_num}"]
    chr_peaks = chr_peaks.sort_values(by='start_bp')
    
    chr_peaks = chr_peaks.drop_duplicates()
    chr_peaks['start_bin'] = np.searchsorted(bb, chr_peaks['start_bp'], side='right') 
    chr_peaks['end_bin'] = np.searchsorted(bb, chr_peaks['end_bp'], side='right') -1
    chr_peaks['width_bp'] =  chr_peaks.end_bp - chr_peaks.start_bp
    chr_peaks['width_bin'] =  chr_peaks.end_bin - chr_peaks.start_bin
    chr_peaks = chr_peaks.sort_values(by='start_bin')
    print(f"Fetching {chr_peaks.shape[0]} peak intervals from chromosome {chr_peaks.chrom.unique()[0]}")

    peak_mids = np.array((chr_peaks.end_bin + chr_peaks.start_bin) //2)
    peak_intervals = np.stack([peak_mids-block_size//2, peak_mids+block_size//2]).T
   
    return peak_intervals, np.array(chr_peaks.width_bin)
#returns 1 if the largest attribution score corresponds to a CTCF feature, 2 if the second largest attribution score...

def get_top_inds(attr_scores):
    attr_scores/=np.max(np.abs(attr_scores))
    third_largest, second_largest, largest = np.unique(attr_scores)[-3:]
    return np.where(np.abs(attr_scores[0])==largest)[1][0] 

    
def is_ctcf_hit(attr_scores):
    attr_scores/=np.max(np.abs(attr_scores))
    ctcf_ids_inds = [305, 306, 532, 533, 773, 774, 775, 776] #IDS features that correspond to CTCF motifs
    third_largest, second_largest, largest = np.unique(attr_scores)[-3:]
    if np.where(np.abs(attr_scores[0])==largest)[1][0] in ctcf_ids_inds:
        return 1
    elif np.where(np.abs(attr_scores[0])==second_largest)[1][0] in ctcf_ids_inds:
        return 2
    elif np.where(np.abs(attr_scores[0])==third_largest)[1][0] in ctcf_ids_inds:
        return 3
    else:
        return 0

def extract_float(x):
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            return float(val[0]) if isinstance(val, list) and len(val) == 1 else float(val)
        except:
            return np.nan  # or raise/log error if preferred
    return float(x)  # already a number




        