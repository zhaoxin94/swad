#!/bin/bash
gpu=$1

CUDA_VISIBLE_DEVICES=$gpu python train_all.py PACS0 --dataset PACS --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir ~/data/DG/domainbed
CUDA_VISIBLE_DEVICES=$gpu python train_all.py PACS1 --dataset PACS --deterministic --trial_seed 1 --checkpoint_freq 100 --data_dir ~/data/DG/domainbed
CUDA_VISIBLE_DEVICES=$gpu python train_all.py PACS2 --dataset PACS --deterministic --trial_seed 2 --checkpoint_freq 100 --data_dir ~/data/DG/domainbed