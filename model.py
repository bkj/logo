#!/usr/bin/env python

"""
    model.py
    
    !! This is similar in spirit but wrong in the details of the code from the repo
    !! I was implementing mostly following the picture, which is not what the code is
"""

import torch
from torch import nn
from torch.nn import functional as F

# --
# Helpers

class RConv2d(nn.Conv2d):
    def forward(self, x):
        return F.relu(super().forward(x))


class DoubleRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        
        self.conv1 = RConv2d(in_channels, out_channels, **kwargs)
        self.conv2 = RConv2d(out_channels, out_channels, **kwargs)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def spatial_tile(x, target_shape):
    return x.repeat(1, 1, target_shape[2], target_shape[3])

# --
# Query encoder

class QueryEncoder(nn.Module):
    # !! Matches original
    def __init__(self):
        super().__init__()
        
        conv_kwargs = {"kernel_size" : 3, "stride" : 1, "padding" : 1}
        self.layers = nn.Sequential(
            # Block 1
            RConv2d(3, 32, **conv_kwargs),
            RConv2d(32, 32, **conv_kwargs),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            RConv2d(32, 64, **conv_kwargs),
            RConv2d(64, 64, **conv_kwargs),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            RConv2d(64, 128, **conv_kwargs),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            RConv2d(128, 256, **conv_kwargs),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            RConv2d(256, 512, **conv_kwargs),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 6
            RConv2d(512, 512, kernel_size=2),
        )
    
    def forward(self, x):
        return self.layers(x)

# --
# Gallery encoder


class DoubleUpConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        
        self.conv1  = RConv2d(in_channels, out_channels, **kwargs)
        self.conv2  = RConv2d(out_channels, out_channels, **kwargs)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2)
        self.act3   = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.deconv(x)[:,:,:-1,:-1] # !! 
        x = self.act3(x)
        return x


class GalleryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        conv_kwargs = {"kernel_size" : 3, "stride" : 1, "padding" : 1}
        self.block1 = DoubleRConv2d(3, 64, **conv_kwargs)
        self.block2 = DoubleRConv2d(64, 128,  **conv_kwargs)
        self.block3 = DoubleRConv2d(128, 256, **conv_kwargs)
        self.block4 = DoubleRConv2d(256, 512, **conv_kwargs)
        self.block5 = DoubleRConv2d(512, 512, **conv_kwargs)
    
    def forward(self, x0):
        x1 = self.block1(x0)
        p1 = F.max_pool2d(x1, kernel_size=2, stride=2)
        
        x2 = self.block2(x1)
        p2 = F.max_pool2d(x1, kernel_size=2, stride=2)
        
        x3 = self.block3(x2)
        p3 = F.max_pool2d(x1, kernel_size=2, stride=2)
        
        x4 = self.block4(x3)
        p4 = F.max_pool2d(x1, kernel_size=2, stride=2)
        
        x5 = self.block5(x4)
        p5 = F.max_pool2d(x1, kernel_size=2, stride=2)
        
        return (x1, x2, x3, x4, x5, p5)


class GalleryDecoder(nn.Module):
    def __init__(self, q_channels=512):
        super().__init__()
        
        conv_kwargs = {"kernel_size" : 3, "stride" : 1, "padding" : 1}
        
        self.up5 = DoubleUpConv(2 * 512, 512, **conv_kwargs)
        self.up4 = DoubleUpConv(2 * 512, 512, **conv_kwargs) # There's a difference between code and drawing here
        self.up3 = DoubleUpConv(2 * 256, 128, **conv_kwargs) # ^
        self.up2 = DoubleUpConv(2 * 128, 64, **conv_kwargs)  # ^
        self.up1 = DoubleUpConv(2 * 64, 64, **conv_kwargs)   # ^
        
        self.s_conv4 = RConv2d(512 + q_channels, 512, kernel_size=1)
        self.s_conv3 = RConv2d(256 + q_channels, 256, kernel_size=1)
        self.s_conv2 = RConv2d(128 + q_channels, 128, kernel_size=1)
        self.s_conv1 = RConv2d(64 + q_channels, 64, kernel_size=1)
        
        self.out = nn.Conv2d(64, 1, **conv_kwargs)
        
    def forward(self, gallery_encs, query_enc):
        
        (x1, x2, x3, x4, x5, p5) = gallery_encs
        
        up5 = self.up5(torch.cat([p5, spatial_tile(query_enc, p5.shape)], dim=1)) # 1024 -> 512
        
        shortcut4 = self.s_conv4(torch.cat([x5, spatial_tile(query_enc, x5.shape)], dim=1)) # 1024 -> 512
        up4       = self.up4(torch.cat([up5, shortcut4], dim=1))
        
        shortcut3 = self.s_conv3(torch.cat([x4, spatial_tile(query_enc, x4.shape)], dim=1)) # 
        up3       = self.up3(torch.cat([up4, shortcut3], dim=1))
        
        shortcut2 = self.s_conv2(torch.cat([x3, spatial_tile(query_enc, x3.shape)], dim=1))
        up2       = self.up2(torch.cat([up3, shortcut2], dim=1))
        
        shortcut1 = self.s_conv1(torch.cat([x2, spatial_tile(query_enc, x2.shape)], dim=1))
        up1       = self.up1(torch.cat([up2, shortcut1], dim=1))
        
        return self.out(up1)
