#!/usr/bin/env python

"""
    make_queries.py
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from glob import glob

# --
# Helpers

def load_bbox(paths):
    for path in paths:
        for bbox in open(path).read().splitlines()[1:]:
            bbox = [int(bb) for bb in bbox.split(' ')]
            out = {
                "id"     : os.path.basename(path).split('.')[0],
                "class"  : os.path.basename(os.path.dirname(path)),
                "left"   : bbox[0],
                "upper"  : bbox[1],
                "right"  : bbox[0] + bbox[2],
                "lower"  : bbox[1] + bbox[3],
            }
            
            out['img_path'] = os.path.join('classes/jpg', out['class'], out['id'] + '.jpg')
            
            yield out

def make_query(row):
    outpath = os.path.join('classes/query', row['class'], row['id'] + '.jpg')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    
    img   = Image.open(row.img_path)
    query = img.crop((row['left'], row['upper'], row['right'], row['lower']))
    query.save(outpath)


if __name__ == "__main__":
    bbox_paths = glob('classes/masks/*/*bboxes.txt')
    all_bbox   = list(load_bbox(bbox_paths))
    
    all_bbox = pd.DataFrame(all_bbox, columns=all_bbox[0].keys())
    
    for _, row in tqdm(all_bbox.iterrows()):
        make_query(row)