import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from tqdm import tqdm, trange

import utils

EPS = 1e-6
MAX = 1e6

class model(nn.Module):
    def __init__(self, video_size, video_channels, video_length, hidden_dim):
        super(model, self).__init__()  
        self.project_ir02 = nn.Linear(video_size[0] * video_size[1], hidden_dim//4)
        self.project_swir = nn.Linear(video_size[0] * video_size[1], hidden_dim//4)
        self.project_ir01 = nn.Linear(video_size[0] * video_size[1], hidden_dim//4)
        self.project_wv = nn.Linear(video_size[0] * video_size[1], hidden_dim//4)
        
    def embed_video(self, data):
        # Embed each channel.
        emb = {
            'ir02': self.project_ir02(data[:,:,0,:]),
            'swir': self.project_swir(data[:,:,1,:]),
            'ir01': self.project_ir01(data[:,:,2,:]),
            'wv': self.project_wv(data[:,:,3,:])
        }
        
        # Concatanate four channels.
        emb = torch.cat((emb['ir02'], emb['swir'], emb['ir01'], emb['wv']), 2)
        
        # Average 20 images.
        emb = torch.mean(emb, 1)
        
        return emb
        
    def forward(self, data_x, data_y, data_z, sim_xy, sim_xz):
        data_x = torch.flatten(data_x, 3, 4)
        data_y = torch.flatten(data_y, 3, 4)
        data_z = torch.flatten(data_z, 3, 4)
        
        emb_x = self.embed_video(data_x)
        emb_y = self.embed_video(data_y)
        emb_z = self.embed_video(data_z)

        # compute distances
        dist_xy = torch.norm(emb_x - emb_y, dim=1)
        dist_xz = torch.norm(emb_x - emb_z, dim=1)

        # compute loss
        loss = torch.pow(torch.log(dist_xy / dist_xz) - torch.log((1 - sim_xy + EPS) / (1 - sim_xz + EPS)), 2)
        return loss.mean()
    