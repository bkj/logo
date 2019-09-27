#!/usr/bin/env python

"""

"""

from rsub import *
from matplotlib import pyplot as plt

import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from model import GalleryEncoder, GalleryDecoder, QueryEncoder

np.random.seed(123)
torch.manual_seed(123 + 111)
torch.cuda.manual_seed(123 + 222)

# --
# Helpers

def path2class(path):
    return os.path.basename(os.path.dirname(path))

def path2img(path, resize=None):
    img = Image.open(path)
    if resize is not None:
        img = img.resize((resize, resize))
    
    return np.array(img)

def path2mask(path, resize=None):
    path = re.sub('/jpg/', '/masks/', path) + '.mask.merged.png'
    img  = Image.open(path)
    if resize is not None:
        img = img.resize((resize, resize))
    
    return np.array(img).astype(np.float64)

# --
# IO

paths   = open('data/all.relpaths.txt').read().splitlines()
paths   = [os.path.join('data', p.lower()) for p in paths]
paths   = [p for p in paths if 'no-logo' not in p]

bboxes = pd.read_csv('data/all_bbox.tsv', sep='\t')

classes = [path2class(p) for p in paths]
# imgs    = np.stack([path2img(p, resize=256) for p in tqdm(paths)])
# mask    = np.stack([path2mask(p, resize=256) for p in tqdm(paths)])
# queries = np.stack([path2img(p, resize=64) for p in tqdm(bboxes.query_path)])

# # Normalize
# imgs_mean = imgs.mean(axis=(0, 1, 2), keepdims=True)
# imgs_std  = imgs.std(axis=(0, 1, 2), keepdims=True)

# imgs    = (imgs - imgs_mean) / imgs_std
# queries = (queries - imgs_mean) / imgs_std
# mask    = mask / 255

# imgs    = torch.FloatTensor(imgs).permute(0, 3, 1, 2)
# queries = torch.FloatTensor(queries).permute(0, 3, 1, 2)
# mask    = torch.FloatTensor(mask).unsqueeze(1)

# torch.save(imgs, 'imgs')
# torch.save(queries, 'queries')
# torch.save(mask, 'mask')

imgs    = torch.load('imgs')
queries = torch.load('queries')
mask    = torch.load('mask')


# --
# Make train/test datasets

class2idx = pd.Series(range(len(classes))).groupby(classes).apply(list).to_dict()
all_pairs = sum([list(combinations(v, 2)) for v in class2idx.values()], [])
all_pairs += [(b, a) for a, b in all_pairs]

train_pairs, valid_pairs = train_test_split(all_pairs, train_size=0.9, test_size=0.1, shuffle=True)

# --
# Define models

class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.query_encoder   = QueryEncoder()
        self.gallery_encoder = GalleryEncoder()
        self.gallery_decoder = GalleryDecoder()
        
    def forward(self, x_batch, q_batch):
        x_batch, q_batch = x_batch.cuda(), q_batch.cuda()
        return self.gallery_decoder(self.gallery_encoder(x_batch), self.query_encoder(q_batch))


model = Pipeline().to('cuda')
model = nn.DataParallel(model)
opt   = torch.optim.Adam(model.parameters())

train_idx_loader = DataLoader(
    TensorDataset(torch.LongTensor(train_pairs)),
    shuffle=True,
    batch_size=256,
)

# >>
idx, = next(iter(train_idx_loader))

with torch.no_grad():
    x_batch = imgs[idx[:,0]]
    q_batch = queries[idx[:,1]]
    x_dec   = model(x_batch, q_batch)
# <<

loss_hist = []
for batch_idx, (idx,) in enumerate(tqdm(train_idx_loader)):
    
    x_batch = imgs[idx[:,0]]
    q_batch = queries[idx[:,1]]
    x_dec   = model(x_batch, q_batch)
    
    y_batch = mask[idx[:,0]].cuda()
    
    loss = F.binary_cross_entropy_with_logits(x_dec, y_batch)
    loss_hist.append(float(loss))
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if (batch_idx > 0) and (batch_idx % 100 == 0):
        plt.plot(loss_hist[10:])
        show_plot()



