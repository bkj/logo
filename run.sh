#!/bin/bash

# run.sh

# --
# Setup env

conda create -y --name logo_env python=3.7
conda activate logo_env

conda install -y -c pytorch pytorch==1.2.0 torchvision
conda install -y pandas
conda install -y tqdm
conda install -y scikit-learn

pip install git+https://github.com/bkj/rsub
pip install matplotlib

# --
# Run

python main.py