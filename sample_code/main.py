import os
import time
import itertools
import numpy as np
from tqdm import trange, tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utils
import model

args = utils.parse_args()
channels = ['ir01.png', 'ir02.png', 'swir.png', 'wv.png']

########## Check GPU ##########
if torch.cuda.is_available():
    device = torch.device("cuda:" + args.gpu)
else:
    device = torch.device("cpu")
print(device, '\n')
    
########## Read Data ##########
start_time = time.time()

video2image = {}
with open('video2image.txt', 'r') as f:
    for i, line in enumerate(f):
        video2image[i+1] = [int(x) for x in line.split(',')]
        
data_x, data_y, data_z, sim_xy, sim_xz = [], [], [], [], []
with open('train.txt', 'r') as f:
    for line in f:
        terms = line.split('\t')

        idx_x, idx_y, idx_z = int(terms[0]), int(terms[1]), int(terms[2])
        video_x, video_y, video_z = [], [], []
        for i in range(args.video_size):
            v_x = torch.stack([utils.read_image('images/image_{}/{}'.format(video2image[idx_x][i], channel)) for channel in channels], 0)
            v_y = torch.stack([utils.read_image('images/image_{}/{}'.format(video2image[idx_y][i], channel)) for channel in channels], 0)
            v_z = torch.stack([utils.read_image('images/image_{}/{}'.format(video2image[idx_z][i], channel)) for channel in channels], 0)
            video_x.append(v_x)
            video_y.append(v_y)
            video_z.append(v_z)
        video_x, video_y, video_z = torch.stack(video_x, 0), torch.stack(video_y, 0), torch.stack(video_z, 0)

        data_x.append(video_x)
        data_y.append(video_y)
        data_z.append(video_z)
        sim_xy.append(float(terms[3]))
        sim_xz.append(float(terms[4]))
    
data_x, data_y, data_z = torch.stack(data_x, 0), torch.stack(data_y, 0), torch.stack(data_z, 0)
sim_xy, sim_xz = torch.tensor(sim_xy), torch.tensor(sim_xz)
data_size, video_length, video_channels, video_height, video_width = data_x.shape

print('# of train triples:\t' + str(data_size))
print('Reading data done:\t{:.2f} seconds'.format(time.time() - start_time), '\n')

########## Define the model ##########
model = model.model((video_height, video_width), video_channels, video_length, args.dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-7)

########## Train Model ##########
for epoch in range(1, args.epochs+1):
    print('\nEpoch:\t', epoch)
    
    ########## Train ##########
    train_time = time.time()
    model.train()
    epoch_loss = 0
    
    batches = utils.generate_batches(data_size, args.batch_size)

    for i in trange(len(batches), position=0, leave=False):
        batch = batches[i]
        
        loss = model(data_x[batch].to(device), data_y[batch].to(device), data_z[batch].to(device), sim_xy[batch].to(device), sim_xz[batch].to(device))
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
    print('Loss:\t', epoch_loss)
    train_time = time.time() - train_time
    
    ########## Save the Model ##########
    if epoch % 1 == 0:
        log_dic = {
            'epoch': epoch,
            'loss': epoch_loss,
            'runtime': train_time
        }
        utils.write_log(args.log_path, log_dic)

    if epoch % 5 == 0:
        torch.save(model, os.path.join('models', 'model_ep_{}.pt'.format(epoch)))