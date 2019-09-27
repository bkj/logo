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
        for i, bbox in enumerate(open(path).read().splitlines()[1:]):
            bbox = [int(bb) for bb in bbox.split(' ')]
            out = {
                "id"     : os.path.basename(path).split('.')[0],
                "class"  : os.path.basename(os.path.dirname(path)),
                "left"   : bbox[0],
                "upper"  : bbox[1],
                "right"  : bbox[0] + bbox[2],
                "lower"  : bbox[1] + bbox[3],
            }
            
            out['img_path']   = os.path.join('data/classes/jpg', out['class'], out['id'] + '.jpg')
            out['query_path'] = os.path.join('data/classes/query', out['class'], out['id'] + '.' + str(i) + '.jpg')
            
            yield out

def make_query(row):
    os.makedirs(os.path.dirname(row.query_path), exist_ok=True)
    
    img   = Image.open(row.img_path)
    query = img.crop((row['left'], row['upper'], row['right'], row['lower']))
    query.save(row.query_path)


if __name__ == "__main__":
    bbox_paths = glob('data/classes/masks/*/*bboxes.txt')
    all_bbox   = list(load_bbox(bbox_paths))
    
    all_bbox = pd.DataFrame(all_bbox, columns=all_bbox[0].keys())
    all_bbox.to_csv('data/all_bbox.tsv', sep='\t', index=False)
    
    for _, row in tqdm(all_bbox.iterrows(), total=all_bbox.shape[0]):
        make_query(row)