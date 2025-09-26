import math
import os
import model 
import data_loader
import torch
from importlib import reload  # Python 3.4+
import matplotlib.pyplot as plt
import numpy as np
from time import time
import utils
import seaborn as sns
import pysam
from matplotlib.gridspec import GridSpec


from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torch.utils.data import DataLoader
from profiler import Logger
from torch.optim.lr_scheduler import LambdaLR

data_loader = reload(data_loader)
model = reload(model)
utils = reload(utils)

from data_loader import *
from model import *
from utils import *
        
      
with open('training_params.yaml', 'r') as f:
    params = yaml.safe_load(f)

chr_num=1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
torch.manual_seed(42)



num_epochs = 50
num_warmup_epochs = 5  # Number of epochs for linear warmup




chr_dataset =get_chromosome_dataset(params['input_encoding'],
                                    chr_num=chr_num, 
                                    params=params,
                                    logger=Logger(f'val_dataset.log'), 
                                    data_augmentation = False,
                                    keep_only_peaks=False, 
                                    testing_mode=True)


dataloader = DataLoader(chr_dataset, 
                        batch_size=params['batch_size_per_GPU'],
                        pin_memory=True, 
                        shuffle=False, 
                        num_workers = 0)


# Create custom figure layout
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, height_ratios=[1, 1])  # 2 rows, 2 columns

ax_loss = fig.add_subplot(gs[0, 0])     # top-left
ax_grad = fig.add_subplot(gs[0, 1])     # top-right
ax_sample = fig.add_subplot(gs[1, :])   # bottom, spans both columns


colors = ['red', 'blue', 'orange', 'green', 'purple', 'magenta', 'green', 'gray']
i = 0
loss_name = 'pnll+multi'
lr = 1e-4

for multinomial_weight in [3]:
    for blah in [1]:
        lossfunc = get_loss(loss_name, poisson_weight = 1, multinomial_weight = multinomial_weight)
        onehot_epoch_losses = []
        model = My_transformer(
            input_encoding=params['input_encoding'],
            our_feat_num=params['num_feats'],
            num_tracks=params['num_tracks'],
            dropout=params['dropout'],
           n_layer = params['n_layer']).to(device, torch.float32)

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # --- Linear LR warmup scheduler ---
        def lr_lambda(current_epoch):
            if current_epoch < num_warmup_epochs:
                return float(current_epoch + 1) / float(num_warmup_epochs)
            else:
                return 1.0
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        # -----------------------------------

        for epch in tqdm(range(num_epochs)):
            losses = []
            for it, data in enumerate(dataloader):
                x, target, bb = data
                x, target, bb = x.to(device, dtype), target.to(device, dtype), bb.to(device, dtype)

                pred = model(x, bb)
                loss = lossfunc(pred, target)
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                grad_maxes, grad_mins = get_grad_max_min(model)
                optimizer.zero_grad()
                break  # single batch per epoch

            onehot_epoch_losses.append(np.mean(losses))
            ax_grad.scatter(epch * np.ones_like(grad_maxes), grad_maxes, color=colors[i], marker=".", alpha=0.5)
            ax_grad.scatter(epch * np.ones_like(grad_mins), grad_mins, color=colors[i], marker="v")

            scheduler.step()  # <-- Step the scheduler at the end of each epoch

        ax_loss.plot(onehot_epoch_losses, color=colors[i], label=f"lr_{lr}_{multinomial_weight}")
        ax_sample.plot(np.arange(target.shape[1]), pred[0, :, 0].detach().cpu(), alpha=0.5, label='pred_' + f"lr_{lr}_{multinomial_weight}")

        i += 1

        ax_loss.plot(lossfunc(target, target).item() * np.ones(len(onehot_epoch_losses)), label='perfect prediction')

# Labels and titles
ax_loss.set_title(f"{params['input_encoding']} encoding, original target")
ax_loss.set_xlabel('epoch')
ax_loss.legend()

ax_grad.set_title("Gradient max and min")

ax_sample.set_title("Overfitting example")
ax_sample.plot(np.arange(target.shape[1]), target[0, :, 0].detach().cpu(), label='gt')
ax_sample.legend()

plt.tight_layout()
plt.savefig('overfitting.png')
