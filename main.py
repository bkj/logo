#!/usr/bin/env python

"""
    main.py
"""

from rsub import *
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import QueryEncoder, GalleryEncoder, GalleryDecoder

# # --
# # Testing

# q = torch.rand(1, 3, 64, 64)
# x = torch.rand(1, 3, 256, 256)

# query_encoder   = QueryEncoder()
# gallery_encoder = GalleryEncoder()
# gallery_decoder = GalleryDecoder()

# query_enc    = query_encoder(q)
# gallery_encs = (x1, x2, x3, x4, x5) = gallery_encoder(x)

# print('query_enc.shape', query_enc.shape)
# print('gallery_encs.shape', [g.shape for g in gallery_encs])

# gallery_decoder(gallery_encs, query_enc).shape

# --
# IO


def is_valid_jpg(path):
    return 'no-logo' not in path

def is_valid_mask(path):
    return 'merged.png' in path

jpg_root  = 'data/FlickrLogos-v2/classes/jpg'
mask_root = 'data/FlickrLogos-v2/classes/masks'

dataset_jpg  = ImageFolder(root=jpg_root, transform=transforms.ToTensor(), is_valid_file=is_valid_jpg)
dataset_mask = ImageFolder(root=mask_root, transform=transforms.ToTensor(), is_valid_file=is_valid_mask)

assert len(dataset_jpg) == len(dataset_mask)

x, y = jpg[0]

_ = plt.imshow(x.permute(1, 2, 0))
show_plot()