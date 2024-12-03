## This file is test but shared the same interface as train

import os
import glob
import argparse

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from torch.utils.data import DataLoader

from datasets import MAPData
# from .transforms import make_transform
from models import SegmentationModel
from test import test_main

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train_data_dir', type=str, default='/projects/bbym/shared/all_patched_data/training')
parser.add_argument('--test_data_dir', type=str, default='/projects/bbym/shared/all_patched_data/validation')
parser.add_argument('--map_data_dir', type=str, default='/projects/bbym/shared/data/cma/validation/')

parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--project', type=str, default='DARPA')
# parser.add_argument('--name', type=str, default='DARPA_Unet')
parser.add_argument('--model', type=str, default='Unet')
parser.add_argument('--encoder', type=str, default='None')
parser.add_argument('--ckpt', type=str, default='./checkpoints/DARPA_Unet_fold02_val_vanilla/jaccard_index_value=0.9284.ckpt')

parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--kfold', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--optimizer', type=str, default='adamp')
parser.add_argument('--scheduler', type=str, default='reducelr')
# parser.add_argument('--loss', type=str, default='ce')

# parser.add_argument('--crop_image_size', type=int, default=512)
# parser.add_argument('--ShiftScaleRotateMode', type=int, default=4)
# parser.add_argument('--ShiftScaleRotate', type=float, default=0.2)
# parser.add_argument('--HorizontalFlip', type=float, default=0.2)
parser.add_argument('--filp_rate', type=float, default=0.4)
parser.add_argument('--color_jitter_rate', type=float, default=0.2)
parser.add_argument('--edge', type=bool, default=False)
parser.add_argument('--superpixel', type=str, default='')

args = parser.parse_args()

args.name = args.project + "_" + args.model

if __name__ == '__main__':
    pl.seed_everything(args.seed)
    print("cuda:", torch.cuda.is_available())
    # args.ckpt = None #"./checkpoints/DARPA_Unet_fold02_val/jaccard_index_value=0.9229.ckpt"
    mapnames = glob.glob(args.map_data_dir + "/*.tif")
    f1_scores = []
    # thres = [0.5]
    thres = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    for i in range(len(thres)):
        f1_scores.append([])
    
    for map_file in mapnames:
        # args.map_file = map_file[map_file.rfind("/") + 1:]
        # # test for one legend
        # test_main(args)
        try:
            args.map_file = map_file[map_file.rfind("/") + 1:]
            # test for one legend
            f1 = test_main(args, thres = thres)
            for i in range(len(thres)):
                f1_scores[i] += f1[i]
            # f1_scores += f1
        except Exception as e:
            print("error when processing " + args.map_file + ": ")
            print(e)
        
        for i in range(len(thres)):
            print("current score for threshold:", thres[i], np.mean(f1_scores[i]) , np.median(f1_scores[i]))

    # print(sum(f1_scores) / len(f1_scores))
    for i in range(len(thres)):
        print("for threshold:", thres[i], np.mean(f1_scores[i]) , np.median(f1_scores[i]))
    print(f1_scores)
            