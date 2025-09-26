from data_loader import *
from model import *
import os
from utils import *
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from time import time
import argparse
import pytorch_warmup as warmup
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from profiler import Logger
import wandb
from torch.optim.lr_scheduler import StepLR, LambdaLR

wandb.login()

torch.manual_seed(0)
torch.cuda.manual_seed(0) if torch.cuda.is_available() else None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
num_GPU = torch.cuda.device_count()


job_id = os.environ["SLURM_JOB_ID"] if os.getenv("SLURM_JOB_ID") else "000"
torch.backends.cudnn.benchmark = True

with open('training_params.yaml', 'r') as f:
    params = yaml.safe_load(f)



this_exp_dir = create_exp_directory(params['experiments_dir'], job_id, rank=0)
logger = Logger(f'{this_exp_dir}/profile_DDP.log')
writer = SummaryWriter(this_exp_dir)
run = wandb.init(project="Tokenization", name = f"{job_id}", dir = this_exp_dir, config=params)


logger.write_params_file(this_exp_dir)
logger.log(f"running file {os.path.basename(__file__)}")
logger.log(f"Num GPUs: {num_GPU}")
logger.log(f"Job ID: {job_id}")



train_dataset = GenomeDataset(params, #dictionary containing all required parameters
                             logger,
                             mode = 'train', 
                             data_augmentation=True,
                             keep_only_peaks = False,
                             testing_mode = False)
                             
                              
val_dataset = GenomeDataset(params, #dictionary containing all required parameters
                             logger,
                             mode = 'val', 
                             data_augmentation=False,
                             keep_only_peaks = False,
                             testing_mode = False)


#for visualizing tracks
chr1_peaks_dataset = get_chromosome_dataset(params['input_encoding'],
                                            chr_num=1, 
                                            params=params,
                                            logger=logger, 
                                            data_augmentation=False, 
                                            keep_only_peaks=True, 
                                            testing_mode=True)

chr15_peaks_dataset = get_chromosome_dataset(params['input_encoding'],
                                             chr_num=15, 
                                             params=params,
                                             logger=logger, 
                                             data_augmentation=False, 
                                             keep_only_peaks=True, 
                                             testing_mode=True)



train_dataloader = DataLoader(
        train_dataset,
        batch_size=params['batch_size_per_GPU']*num_GPU,#data parallel will divide this by the available GPUs (it expects total batch size)
        shuffle=True,
        num_workers=params['num_workers'], #num_workers per GPU
        prefetch_factor = 4,
        pin_memory=True)




val_dataloader = DataLoader(
        val_dataset,
        batch_size=params['batch_size_per_GPU']*num_GPU,#total_batch_size//world_size = batch_size per GPU
        shuffle=False,
        num_workers=params['num_workers'], #num_workers per GPU
        prefetch_factor = 4,
        pin_memory=True)

feature_sparsities = np.load("/orcd/home/002/parmida/TF_project/genomewide_IDSfeature_sparsities.npy")
num_feats = np.sum(feature_sparsities<params['feature_sparsity_threshold'])
logger.log(f"Using a feature sparsity threshold of {params['feature_sparsity_threshold']} which leaves us with {num_feats} features")


model = VGG14_1D( input_encoding = params['input_encoding'], 
                 our_feat_num = params['num_feats'], num_tracks=params['num_tracks'], 
                 dropout = params['dropout'], n_conv = params['n_conv']).to(device, dtype)
# My_transformer(
#             input_encoding=params['input_encoding'],
#             our_feat_num=num_feats,
#             num_tracks=params['num_tracks'],
#             dropout=params['dropout'],
#            n_layer = params['n_layer']).to(device, dtype)


run.watch(model)





model_size = sum(p.numel() for p in model.parameters())
logger.log(f"model size: {model_size/1e6} M parameters")
if torch.cuda.device_count() > 1:
    logger.log(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)



if params['starting_epoch']!=0:
    map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
    model_dict = model.state_dict()
    pretrained_dict = torch.load(params['experiments_dir']+f"exp_{params['saved_epoch_exp']}/model_params_epch_{params['starting_epoch']}.pt", map_location=map_location, weights_only=True)
    model.load_state_dict(pretrained_dict)


model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])

# --- Linear warmup for 5 epochs, then step decay every 5 epochs ---
num_warmup_epochs = 5
step_size = 5
gamma = params['gamma']
base_lr = params['lr']

def lr_lambda(epoch):
    if epoch < num_warmup_epochs:
        return float(epoch + 1) / float(num_warmup_epochs)
    else:
        # After warmup, decay by gamma every step_size epochs
        return gamma ** ((epoch - num_warmup_epochs + 1) // step_size)

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
# ---------------------------------------------------------------

# By setting log_input=False, you inform PyTorch that the inputs are not in log space and it will internally take the logarithm before computing the loss. This matches the behavior of the softplus activation function, which outputs positive values.
criterion = get_loss('pnll+multi', poisson_weight = params['poisson_weight'], \
                     multinomial_weight = params['multinomial_weight'])
# scaler = torch.cuda.amp.GradScaler() #used with autocast to prevent the small gradients from zeroing out
 



train_losses = []
lrs = []
train_corrs = []
val_corrs = []
val_losses = []
grad_list = []


# accumulation_steps = 4  # Accumulate gradients over 4 batches ###########


metrics = {}
crop = params['crop_size']
t0 = time()
for epoch in tqdm(np.arange(params['starting_epoch'], params['epochs'])):
    epoch = int(epoch)
    logger.log(f"Epoch {epoch}, time passed: {(time() - t0)/3600} Hours")
    train_loss_to_avg = []
    train_corr_to_avg = []
    for it, data in enumerate(train_dataloader):
        x, y, bb = data
        
        if params['input_encoding']=='ours': 
            x = x + torch.randn_like(x) * params['gaussian_noise_std']*0.4  #0.4 is the std of our input features
        x, y, bb = x.to(device, dtype), y.to(device, dtype), bb.to(device, dtype)

        
        predicted_tracks = model(x) 

        loss = criterion(predicted_tracks[:,crop:-crop,:], y[:,crop:-crop,:]) #/ accumulation_steps############
        

        if torch.isnan(loss).any().item():
            print("nan loss: ", it, predicted_tracks[:,crop:-crop,:])
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
     

        train_loss_to_avg.append(loss.item())  #* accumulation_steps
        train_corr_to_avg.append(get_avg_correlation_per_track(y[:,crop:-crop,:], predicted_tracks[:,crop:-crop,:]))
    scheduler.step()
    
    val_loss,val_track_corrs, val_pred, val_y = evaluate(criterion, model, val_dataloader, crop,params['num_tracks'])
    train_loss,train_track_corrs = np.mean(train_loss_to_avg), np.nanmean(train_corr_to_avg)
    
    logger.log(f"Training loss: {train_loss}") 
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    np.save(f"{this_exp_dir}/train_loss.npy", train_losses)    
    np.save(f"{this_exp_dir}/val_loss.npy", val_losses)
    metrics['train_loss'] = train_loss
    metrics['val_loss'] = val_loss
    metrics['epoch'] = epoch
    metrics['train_corr'] = np.nanmean(train_track_corrs)
    metrics['val_corr'] = np.nanmean(val_track_corrs)
    metrics['LR'] = optimizer.param_groups[0]['lr']
    # metrics['max_grad'] = max_grad
    wandb.log(metrics)
    writer.add_scalars("Loss", {"train": train_loss}, epoch) 
    writer.add_scalars("Correlation", {"train": np.nanmean(train_track_corrs)}, epoch) 
    writer.add_scalars("Loss", {"train": train_loss,"val": val_loss }, epoch) 
    writer.add_scalars("Correlation", {"train": np.nanmean(train_track_corrs),"val": np.nanmean(val_track_corrs) }, epoch) 
    writer.add_scalar("LR",optimizer.param_groups[0]['lr'] , epoch)
    # writer.add_scalar("max gradient",max_grad, epoch)
    torch.save(model.state_dict(), this_exp_dir + f"/model_params_epch_{epoch}.pt")
    if epoch%5==0:
        im = plot_outputs(model, chr1_peaks_dataset, epoch, dataset_idx = 10, exp_dir=this_exp_dir,append_to_title="")
        writer.add_figure("ouput tracks train",im, epoch)
        wandb.log({"train example":wandb.Image(im)})
        for i in range(3):
            im = plot_outputs(model, chr15_peaks_dataset, epoch, dataset_idx = i, exp_dir=this_exp_dir,append_to_title="")
            writer.add_figure("ouput tracks val {i}",im, epoch)
            wandb.log({f"val example {i}":wandb.Image(im)})
        
        wandb.save(this_exp_dir + f"/model_params_epch_{epoch}.pt")
        writer.flush()       

torch.save(model.state_dict(), this_exp_dir + f"/model_params_epch_{epoch}.pt")
writer.close()
logger.log("Finished training and saved model parameters")
wandb.finish()


    
