import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
import yaml
import copy
import einops
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torch.nn import functional as F
# from learnable_fourier_positional_encoding import LearnableFourierPositionalEncoding



class VGG14_1D_combined(nn.Module):
    def __init__(self, our_feat_num = 871, num_tracks=10, dropout = 0.2, n_conv = 11):
        super(VGG14_1D_combined, self).__init__()
        self.dropout = dropout
        self.n_conv_stem = n_conv - 4 #b/c the encoder section has 4 convolutions

          
    
        self.our_encoder = nn.Sequential(
                nn.Conv1d(in_channels=our_feat_num, out_channels=400, kernel_size=1, padding='same'),
                nn.BatchNorm1d(400),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=400, out_channels=100, kernel_size=1, padding='same'),
                nn.BatchNorm1d(100),
                nn.ReLU(inplace=True),

                nn.Conv1d(100, 128, kernel_size=3, padding='same'),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                #RF = 3
                
                nn.Conv1d(128, 128, kernel_size=3, padding='same'),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                #RF = 5
    
                nn.Conv1d(128, 128, kernel_size=3, padding='same'),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                #RF = 7
                
                nn.Conv1d(128, 512, kernel_size=3, padding='same'),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                #RF = 9 bins
            )
        
        # else:#onehot input
        self.onehot_encoder = nn.Sequential(
                nn.Conv1d(in_channels=4, out_channels=64, kernel_size=5, padding='same'),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 10
                
                nn.Conv1d(64, 128, kernel_size=5, padding='same'),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                # #RF = 28
    
                nn.Conv1d(128, 256, kernel_size=3, padding='same'),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                # #RF = 60
                
                nn.Conv1d(256, 512, kernel_size=3, padding='same'),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 124 bp
            )
            
        self.W = nn.Linear(1024, 512)
        conv_layers = []
        for i in range(self.n_conv_stem):
            conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True)))
        self.stem = nn.Sequential(*conv_layers)
        
        self.linear_block = nn.Sequential(
            nn.Linear(512, 4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, 1028),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(1028, num_tracks),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_onehot, x_ours): # (B,L,num_features)
        x_ours = rearrange(x_ours, 'B L C -> B C L')
        x_ours_deep = self.our_encoder(x_ours) # (B, 512, L)
        
        # x_onehot = rearrange(x_onehot, 'B L C -> B C L')
        # x_onehot = self.onehot_encoder(x_onehot) # (B, 512, L)
        # x = torch.cat([x_onehot, x_ours], dim = 1) # (B, 1024, L)
        # x_ours_shallow = self.our_encoder_shallow(x_ours)
        x = torch.cat([x_ours_deep, x_ours_deep], dim = 1) # (B, 1024, L)
        
        x = rearrange(x, 'B C L -> B L C')
        x = self.W(x) # (B, L, 512)
        x = rearrange(x, 'B L C -> B C L')
        x = self.stem(x) # (B,512, L)
        x = rearrange(x, 'B C L -> B L C')
        x = self.linear_block(x) # (B,L, num_tracks)
        return x

class VGG14_1D(nn.Module):
    def __init__(self, input_encoding = 'ours', our_feat_num = 871, num_tracks=10, dropout = 0.2, n_conv = 11):
        super(VGG14_1D, self).__init__()
        self.dropout = dropout
        self.n_conv_stem = n_conv - 4 #b/c the encoder section has 4 convolutions

        if input_encoding=='ours':
            self.encoder = nn.Sequential(

                nn.Conv1d(in_channels=our_feat_num, out_channels=400, kernel_size=1, padding='same'),
                nn.BatchNorm1d(400, affine=True),  # Using Batch Normalization with learnable parameters
                nn.ReLU(inplace=True),
                
                nn.Conv1d(in_channels=400, out_channels=100, kernel_size=1, padding='same'),
                nn.BatchNorm1d(100, affine=True),
                nn.ReLU(inplace=True),

                nn.Conv1d(100, 128, kernel_size=3, padding='same'),
                nn.BatchNorm1d(128, affine=True),
                nn.ReLU(inplace=True),
                #RF = 3
                
                nn.Conv1d(128, 128, kernel_size=3, padding='same'),
                nn.BatchNorm1d(128, affine=True),
                nn.ReLU(inplace=True),
                #RF = 5
    
                nn.Conv1d(128, 128, kernel_size=3, padding='same'),
                nn.BatchNorm1d(128, affine=True),
                nn.ReLU(inplace=True),
                # #RF = 7
                
                nn.Conv1d(128, 512, kernel_size=3, padding='same'),
                nn.BatchNorm1d(512, affine=True),
                nn.ReLU(inplace=True),
                #RF = 9 bins
            )
        else:  # onehot input
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=4, out_channels=64, kernel_size=5, padding='same'),
                nn.BatchNorm1d(64, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 10
                
                nn.Conv1d(64, 128, kernel_size=5, padding='same'),
                nn.BatchNorm1d(128, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 28
    
                nn.Conv1d(128, 256, kernel_size=3, padding='same'),
                nn.BatchNorm1d(256, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 60
                
                nn.Conv1d(256, 512, kernel_size=3, padding='same'),
                nn.BatchNorm1d(512, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 124 bp
            )
            
        conv_layers = []
        for i in range(self.n_conv_stem):
            conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
                nn.BatchNorm1d(512, affine=True),
                nn.ReLU(inplace=True)))
        self.stem = nn.Sequential(*conv_layers)

        self.linear_block = nn.Sequential(
            nn.Linear(512, 4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, 1028),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(1028, num_tracks),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): # (B,L,num_features)
        x = rearrange(x, 'B L C -> B C L')
        x = self.encoder(x) # (B, 512, L)
        x = self.stem(x) # (B,512, L)
        x = rearrange(x, 'B C L -> B L C')
        x = self.linear_block(x) # (B,L, num_tracks)
        return x

        
class My_transformer_combined(nn.Module):
    def __init__(self,our_feat_num=871, n_embd=512, n_head=8, masking=False,
                 n_layer=10, d_feedforward=1024, dropout=0.2,  num_tracks=1, block_size=1250):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_feedforward = d_feedforward
        self.dropout = dropout
        self.num_tracks = num_tracks
        self.block_size = block_size

        # Convolutional encoder
        self.our_conv_encoder = nn.Sequential(
                nn.Conv1d(in_channels=our_feat_num, out_channels=400, kernel_size=1, padding='same'),
                nn.BatchNorm1d(400, affine=True),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=400, out_channels=100, kernel_size=1, padding='same'),
                nn.BatchNorm1d(100, affine=True),
                nn.ReLU(inplace=True),

                nn.Conv1d(100, 256, kernel_size=3, padding='same'),
                nn.BatchNorm1d(256, affine=True),
                nn.ReLU(inplace=True),

                nn.Conv1d(256, n_embd, kernel_size=3, padding='same'),
                nn.BatchNorm1d(n_embd, affine=True),
                nn.ReLU(inplace=True),
            )
        # self.our_second_conv_encoder = nn.Sequential(
        #         nn.Conv1d(in_channels=our_feat_num, out_channels=400, kernel_size=1, padding='same'),
        #         nn.BatchNorm1d(400, affine=True),
        #         nn.ReLU(inplace=True),

        #         nn.Conv1d(in_channels=400, out_channels=100, kernel_size=1, padding='same'),
        #         nn.BatchNorm1d(100, affine=True),
        #         nn.ReLU(inplace=True),

        #         nn.Conv1d(100, 256, kernel_size=3, padding='same'),
        #         nn.BatchNorm1d(256, affine=True),
        #         nn.ReLU(inplace=True),

        #         nn.Conv1d(256, n_embd, kernel_size=3, padding='same'),
        #         nn.BatchNorm1d(n_embd, affine=True),
        #         nn.ReLU(inplace=True),
        #     )
        self.onehot_conv_encoder = nn.Sequential(
                nn.Conv1d(in_channels=4, out_channels=64, kernel_size=3, padding='same'),
                nn.BatchNorm1d(64, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 6

                nn.Conv1d(64, 128, kernel_size=3, padding='same'),
                nn.BatchNorm1d(128, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 16

                nn.Conv1d(128, 256, kernel_size=3, padding='same'),
                nn.BatchNorm1d(256, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 36

                nn.Conv1d(256, n_embd, kernel_size=3, padding='same'),
                nn.BatchNorm1d(n_embd, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 76
            )
        # self.onehot_second_conv_encoder = nn.Sequential(
        #         nn.Conv1d(in_channels=4, out_channels=64, kernel_size=3, padding='same'),
        #         nn.BatchNorm1d(64, affine=True),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool1d(kernel_size=2, stride=2),
        #         #RF = 6

        #         nn.Conv1d(64, 128, kernel_size=3, padding='same'),
        #         nn.BatchNorm1d(128, affine=True),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool1d(kernel_size=2, stride=2),
        #         #RF = 16

        #         nn.Conv1d(128, 256, kernel_size=3, padding='same'),
        #         nn.BatchNorm1d(256, affine=True),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool1d(kernel_size=2, stride=2),
        #         #RF = 36

        #         nn.Conv1d(256, n_embd, kernel_size=3, padding='same'),
        #         nn.BatchNorm1d(n_embd, affine=True),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool1d(kernel_size=2, stride=2),
        #         #RF = 76
        #     )

        # Positional encoding
        self.positional_encoding = CustomPositionalEncoding(n_embd, block_size)
        self.W = nn.Linear(2*n_embd, n_embd)

        # Transformer encoder
        _enc_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=d_feedforward,
            dropout=dropout, batch_first=True, norm_first=True, activation="relu"
        )
        self.transformer = TransformerEncoder(_enc_layer, num_layers=n_layer)

        self.lin3 = nn.Linear(n_embd, num_tracks)
        self.last_activation = nn.Softplus()

    def forward(self, x_onehot, x_ours, bb_onehot, bb_ours):
        # x: (B, L, C)
        x_onehot = rearrange(x_onehot, 'B L C -> B C L')
        
        x_onehot_first = self.onehot_conv_encoder(x_onehot)  # (B, n_embd, L)
        x_onehot_first = rearrange(x_onehot_first, 'B C L -> B L C')  # (B, L, n_embd)
        x_onehot_first = self.positional_encoding(x_onehot_first, bb_onehot)  # (B, L, n_embd)

        # noise = torch.randn_like(x_onehot)
        # x_onehot_second = self.onehot_second_conv_encoder(noise)  # (B, n_embd, L)
        # x_onehot_second = rearrange(x_onehot_second, 'B C L -> B L C')  # (B, L, n_embd)
        # x_onehot_second = self.positional_encoding(x_onehot_second, bb_onehot)  # (B, L, n_embd)

        x_ours = rearrange(x_ours, 'B L C -> B C L') 
        x_ours_first = self.our_conv_encoder(x_ours)  # (B, n_embd, L)
        x_ours_first = rearrange(x_ours_first, 'B C L -> B L C')  # (B, L, n_embd)
        x_ours_first = self.positional_encoding(x_ours_first, bb_ours)  # (B, L, n_embd)

        # x_ours_second = self.our_second_conv_encoder(x_ours)  # (B, n_embd, L)
        # x_ours_second = rearrange(x_ours_second, 'B C L -> B L C')  # (B, L, n_embd)
        # x_ours_second = self.positional_encoding(x_ours_second, bb_ours)  # (B, L, n_embd)

        x = torch.cat([x_onehot_first, x_ours_first], dim = 2) # (B, L, 1024)
        
        x = self.W(x) # (B, L, 512)
        
        x = self.transformer(x)  # (B, L, n_embd)
        x = self.last_activation(self.lin3(x))  # (B, L, num_tracks)
        return x

class My_transformer(nn.Module):
    def __init__(self, input_encoding='ours', our_feat_num=871, n_embd=512, n_head=8, masking=False,
                 n_layer=10, d_feedforward=1024, dropout=0.2,  num_tracks=1, block_size=1250):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_feedforward = d_feedforward
        self.dropout = dropout
        self.num_tracks = num_tracks
        self.block_size = block_size

        # Convolutional encoder
        if input_encoding == 'ours':
            self.conv_encoder = nn.Sequential(
                nn.Conv1d(in_channels=our_feat_num, out_channels=400, kernel_size=1, padding='same'),
                nn.BatchNorm1d(400, affine=True),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=400, out_channels=100, kernel_size=1, padding='same'),
                nn.BatchNorm1d(100, affine=True),
                nn.ReLU(inplace=True),

                nn.Conv1d(100, 256, kernel_size=3, padding='same'),
                nn.BatchNorm1d(256, affine=True),
                nn.ReLU(inplace=True),

                nn.Conv1d(256, n_embd, kernel_size=3, padding='same'),
                nn.BatchNorm1d(n_embd, affine=True),
                nn.ReLU(inplace=True),
            )
        elif input_encoding == 'onehot_extended': # onehot input
            self.conv_encoder = nn.Sequential(
                nn.Conv1d(in_channels=400, out_channels=200, kernel_size=3, padding='same'),
                nn.BatchNorm1d(200, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 6

                nn.Conv1d(200, 100, kernel_size=3, padding='same'),
                nn.BatchNorm1d(100, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 16

                nn.Conv1d(100, 256, kernel_size=3, padding='same'),
                nn.BatchNorm1d(256, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 36

                nn.Conv1d(256, n_embd, kernel_size=3, padding='same'),
                nn.BatchNorm1d(n_embd, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 76
            )
        else:  # onehot input
            self.conv_encoder = nn.Sequential(
                nn.Conv1d(in_channels=4, out_channels=64, kernel_size=3, padding='same'),
                nn.BatchNorm1d(64, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 6

                nn.Conv1d(64, 128, kernel_size=3, padding='same'),
                nn.BatchNorm1d(128, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 16

                nn.Conv1d(128, 256, kernel_size=3, padding='same'),
                nn.BatchNorm1d(256, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 36

                nn.Conv1d(256, n_embd, kernel_size=3, padding='same'),
                nn.BatchNorm1d(n_embd, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                #RF = 76
            )

        # Positional encoding
        self.positional_encoding = CustomPositionalEncoding(n_embd, block_size)

        # Transformer encoder
        _enc_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=d_feedforward,
            dropout=dropout, batch_first=True, norm_first=True, activation="relu"
        )
        self.transformer = TransformerEncoder(_enc_layer, num_layers=n_layer)

        self.lin3 = nn.Linear(n_embd, num_tracks)
        self.last_activation = nn.Softplus()

    def forward(self, x, bb):
        # x: (B, L, C)
        x = rearrange(x, 'B L C -> B C L')
        x = self.conv_encoder(x)  # (B, n_embd, L)
        x = rearrange(x, 'B C L -> B L C')  # (B, L, n_embd)

        x = self.positional_encoding(x, bb)  # (B, L, n_embd)
        x = self.transformer(x)  # (B, L, n_embd)
        x = self.last_activation(self.lin3(x))  # (B, L, num_tracks)
        return x

        

class LinearBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.activation = nn.GELU()
        self.norm = nn.BatchNorm1d(num_features=output_size)
        
    def forward(self, x): # (B,T,num_features)
             
        x = self.fc1(x) # (B,T,n_embd)
        x = rearrange(x, 'B T D -> B D T')
        x = self.norm(x)
        x = rearrange(x, 'B D T-> B T D')
        
        x = self.activation(x)
        
        return x
    


class CustomPositionalEncoding(nn.Module):
    def __init__(self, n_embd, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.block_size = block_size

    def forward(self, x: Tensor, bb: torch.Tensor) -> Tensor:
        """
        x: Tensor of shape (batch_size, block_size, n_embd)
        bb: Tensor of shape (batch_size, block_size)
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        # bb: (batch_size, block_size, 1)
        position = bb.unsqueeze(-1).to(device, dtype)  # (B, block_size, 1)
        div_term = torch.exp(torch.arange(0, self.n_embd, 2, device=device, dtype=dtype) * -0.02)  # (n_embd//2,)

        # (B, block_size, n_embd)
        pe = torch.zeros(batch_size, self.block_size, self.n_embd, device=device, dtype=dtype)
        pe[:, :, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, :, 1::2] = torch.cos(position * div_term)  # odd indices

        return x + pe
        
class PositionalEncoding(nn.Module):

    def __init__(self, n_embd, block_size):
        super().__init__()
        
        position = torch.arange(block_size).unsqueeze(1) # 1 x block_size
        div_term = torch.exp(torch.arange(0, n_embd, 2)* -0.01) #n_embd//2
        pe = torch.zeros(1, block_size, n_embd)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor of shape B, T, n_embd
        """
        x = x + self.pe[:,:x.size(1),:]
        return  x

    

class My_transformer_with_conv(nn.Module):

    def __init__(self, num_feats = 852, n_embd = 852, n_head = 6, n_layer = 10, d_feedforward=1024, dropout = 0.4, num_tracks = 1, block_size = 300, n_conv = 4):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_feedforward = d_feedforward
        self.dropout = dropout
        self.num_tracks = num_tracks
        
        
        conv_layers = []
        for i in range(n_conv):
            conv_layers.append(nn.Sequential(
                ConvBlock(n_embd, n_embd, kernel_size = 5),
                Residual(ConvBlock(n_embd, n_embd, 1)),
                AttentionPool(n_embd, pool_size = 2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)
           
        self.lin1 = nn.Linear(2, 1)
        self.lin2 = nn.Linear(num_feats, self.n_embd) 
        self.position_embedding = PositionalEncoding(self.n_embd, block_size, dropout)   
             
        _enc_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward = d_feedforward,\
                                                dropout = dropout,batch_first=True, norm_first=True, activation = "relu")
  
        self.encoder = TransformerEncoder(_enc_layer, num_layers= self.n_layer)
        self.lin3 = nn.Linear(self.n_embd, num_tracks)
        self.first_activation = nn.GELU()
        self.last_activation = nn.Softplus()#nn.LeakyReLU()
  
    def forward(self, x):# B,T,num_feats, 2, T=100,000
        x = self.first_activation(self.lin1(x)).squeeze(3) # (B,T,num_feats)
        x = self.first_activation(self.lin2(x))# (B,T,n_embd)
        # x = rearrange(x, 'B T C -> B C T')
        # x = self.conv_tower(x) # (B,C,N) N=T/(2**num_conv)
        # x = rearrange(x, 'B C N -> B N C')
        x = self.position_embedding(x) # (B,N,C)
        x = self.encoder(x) # (B,N,C)
        x = self.last_activation(self.lin3(x))
        assert torch.isnan(x).any() == False
        return x
    
class SmallEnformer(nn.Module):

    def __init__(self, n_embd = 852, n_head = 6, n_layer = 10, d_feedforward=1024, dropout = 0.4, num_tracks = 1, block_size = 300, n_conv = 4):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_feedforward = d_feedforward
        self.dropout = dropout
        self.num_tracks = num_tracks
        self.n_conv = n_conv
        self.register_buffer("batch_PE", torch.zeros(1, block_size, n_embd))
        
        self.stem = nn.Sequential(
                ConvBlock(4, n_embd, kernel_size = 15),
                Residual(ConvBlock(n_embd, n_embd, 1)),
                AttentionPool(n_embd, pool_size = 2))
        
        conv_layers = []
        for i in range(self.n_conv-1):
            conv_layers.append(nn.Sequential(
                ConvBlock(n_embd, n_embd, kernel_size = 5),
                Residual(ConvBlock(n_embd, n_embd, 1)),
                AttentionPool(n_embd, pool_size = 2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)
           
        
        # self.position_embedding = PositionalEncoding(self.n_embd, block_size)   
             
        _enc_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward = d_feedforward,\
                                                dropout = dropout,batch_first=True, norm_first=True, activation = "relu")
  
        self.encoder = TransformerEncoder(_enc_layer, num_layers= self.n_layer)
        self.lin3 = nn.Linear(self.n_embd, num_tracks)
        self.first_activation = nn.GELU()
        self.last_activation = nn.Softplus()#nn.LeakyReLU()
  
    def forward(self, x):# B,T,4     T=100,000
        x = rearrange(x, 'B T C -> B C T')
        x = self.stem(x) # (B,n_embd, T/2) 
        x = self.conv_tower(x) # (B,n_embd,N) N=T/(2**num_conv)
        x = rearrange(x, 'B C N -> B N C')
        # x = self.position_embedding(x) # (B,N,C)
        x = x + self.batch_PE
        x = self.encoder(x) # (B,N,C)
        x = self.last_activation(self.lin3(x)) 
        assert torch.isnan(x).any() == False, print(torch.where(torch.isnan(x)), x)
        return x
    
class SmallEnformerMultitask(nn.Module):

    def __init__(self, n_embd = 852, n_head = 6, n_layer = 10, d_feedforward=1024, dropout = 0.4, num_tracks = 1, block_size = 300, n_conv = 4, multitask_feats = 871):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_feedforward = d_feedforward
        self.dropout = dropout
        self.num_tracks = num_tracks
        self.n_conv = n_conv
        
        self.stem = nn.Sequential(
                ConvBlock(4, n_embd, kernel_size = 15),
                Residual(ConvBlock(n_embd, n_embd, 1)),
                AttentionPool(n_embd, pool_size = 2))
        
        conv_layers = []
        for i in range(self.n_conv-1):
            conv_layers.append(nn.Sequential(
                ConvBlock(n_embd, n_embd, kernel_size = 5),
                Residual(ConvBlock(n_embd, n_embd, 1)),
                AttentionPool(n_embd, pool_size = 2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)
           
        
        self.position_embedding = PositionalEncoding(self.n_embd, block_size*(2**self.n_conv), dropout)   
             
        _enc_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward = d_feedforward,\
                                                dropout = dropout,batch_first=True, norm_first=True, activation = "relu")
  
        self.encoder = TransformerEncoder(_enc_layer, num_layers= self.n_layer)
        self.track_head = nn.Linear(self.n_embd, num_tracks)
        self.motif_head = nn.Linear(self.n_embd, multitask_feats)
        self.first_activation = nn.GELU()
        self.last_activation = nn.Softplus()#nn.LeakyReLU()
  
    def forward(self, x):# B,T,4     T=100,000
        x = rearrange(x, 'B T C -> B C T')
        x = self.stem(x) # (B,n_embd, T/2) 
        x = self.conv_tower(x) # (B,n_embd,N) N=T/(2**num_conv)
        x = rearrange(x, 'B C N -> B N C')
        x = self.position_embedding(x) # (B,N,C)
        x = self.encoder(x) # (B,N,C)
        tracks = self.last_activation(self.track_head(x)) 
        motifs = self.last_activation(self.motif_head(x)) 
        assert torch.isnan(tracks).any() == False, print(torch.where(torch.isnan(tracks)), tracks)
        assert torch.isnan(motifs).any() == False, print(torch.where(torch.isnan(motifs)), motifs)
        
        return tracks, motifs
    
def ConvBlock(dim, dim_out, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        nn.GELU(),
        nn.Conv1d(dim, dim_out, kernel_size, padding = kernel_size // 2)
    )

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape #B, C, T
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x) # B C T/2 2
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1)

    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x   
    
    
    
    




# # ## for testing the residual ADMIN module


# # class TransformerEncoderLayer(nn.Module):

# #     def __init__(self):
# #         super().__init__()

# #         num_layer =  2 * n_layer # number of residual layers

# #         self.attn = nn.MultiheadAttention(n_embd, n_head)
# #         self.residual_attn = admin_torch.as_module(num_layer)
# #         self.ln_attn = nn.LayerNorm(n_embd)

# #         self.ffn = nn.Sequential(
# #             nn.Linear(n_embd, d_feedforward),
# #             nn.ReLU(),
# #             nn.Linear(d_feedforward, n_embd)
# #         )
# #         self.residual_ffn = admin_torch.as_module(num_layer)
# #         self.ln_ffn = nn.LayerNorm(n_embd)

# #     def forward(self, x):

# #         f_x, _ = self.attn(x)
# #         x = self.residual_attn(x, f_x)
# #         x = self.ln_attn(x)

# #         f_x = self.ffn(x)
# #         x = self.residual_ffn(x, f_x)
# #         x = self.ln_ffn(x)

# #         return x

class TransformerLayer(torch.nn.TransformerEncoderLayer):
    # Pre-LN structure
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        # MHA section
        #input is passed in batch first: B, T, C
        src = rearrange(src, 'b t c -> t b c')
        src_norm = self.norm1(src)
        src_side, attn_weights = self.self_attn(src_norm, src_norm, src_norm, 
                                    attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src_side)

        # MLP section
        src_norm = self.norm2(src)
        src_side = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src_side)
        src = rearrange(src, 't b c -> b t c')
        return src, attn_weights

# class TransformerEncoder(torch.nn.TransformerEncoder):

#     def __init__(self, encoder_layer, num_layers, norm=None, record_attn = False):
#         super(TransformerEncoder, self).__init__(encoder_layer, num_layers)
#         self.layers = self._get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm
#         self.record_attn = record_attn

#     def forward(self, src, mask = None, src_key_padding_mask = None):
#         r"""Pass the input through the encoder layers in turn.

#         Args:
#             src: the sequence to the encoder (required).
#             mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """
#         output = src

#         attn_weight_list = []

#         for mod in self.layers:
#             output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
#             attn_weight_list.append(attn_weights.unsqueeze(0).detach())
#         if self.norm is not None:
#             output = self.norm(output)

#         if self.record_attn:
#             return output, torch.cat(attn_weight_list)
#         else:
#             return output

#     def _get_clones(self, module, N):
#         return torch.nn.modules.ModuleList([copy.deepcopy(module) for i in range(N)])


