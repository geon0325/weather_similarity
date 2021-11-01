import os
import math
import imageio
from skimage import io
import numpy as np
import pandas as pd
import argparse
import torch
import pickle
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default='0', type=str, help='gpu number')
    parser.add_argument("--epochs", default=100, type=int, help='number of epochs')
    parser.add_argument("--dim", default=128, type=int, help='hidden dimension')
    parser.add_argument("--learning_rate", default=1e-5, type=float, help='learning rate')
    parser.add_argument("--batch_size", default=32, type=int, help='batch size') 
    parser.add_argument("--N", default='24', type=str, help='number of division')
    parser.add_argument("--B", default='20', type=str, help='number of bins')
    parser.add_argument("--video_size", default=20, type=int, help='number of images per video')
    parser.add_argument("--log_path", default='log.txt', type=str, help='log path')
    return parser.parse_args()

def generate_batches(data_size, batch_size, shuffle=True):
    data = np.arange(data_size)
    
    if shuffle:
        np.random.shuffle(data)
    
    batch_num = math.ceil(data_size / batch_size)
    batches = np.split(np.arange(batch_num * batch_size), batch_num)
    batches[-1] = batches[-1][:(data_size - batch_size * (batch_num - 1))]
    
    for i, batch in enumerate(batches):
        batches[i] = [data[j] for j in batch]
        
    return batches

def rgb2gray(rgb):
    ratio = [0.2989, 0.5870, 0.1140]
    ret = np.expand_dims(np.dot(rgb[...,:3], ratio), axis=2)
    return ret

def read_image(filename):
    image = torch.tensor(io.imread(filename)).float()
    return image
    #image = torch.tensor(rgb2gray(io.imread(filename))).squeeze(2)
    #return image

def write_log(log_path, log_dic):
    with open(log_path, 'a') as f:
        for _key in log_dic:
            f.write(_key + '\t' + str(log_dic[_key]) + '\n')
        f.write('\n') 