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
from tqdm import tqdm

from datasets import MAPData, eval_MAPData
# from .transforms import make_transform
from models import SegmentationModel

parser = argparse.ArgumentParser()
#system setting
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=32)

# model setting
parser.add_argument('--project', type=str, default='DARPA')
parser.add_argument('--model', type=str, default='Unet')
parser.add_argument('--ckpt', type=str, default= None) #'./checkpoints/DARPA_Unet_fold02_val/jaccard_index_value=0.9229.ckpt')  #default="jaccard_index_value=0.9229.ckpt"

# dataloader setting
parser.add_argument('--map_data_dir', type=str, default='/projects/bbym/shared/data/cma/validation/')
parser.add_argument('--map_file', type=str, default="CA_AZ_Needles.tif")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)

args = parser.parse_args()
args = parser.parse_args()

args.name = args.project + "_" + args.model
# with torch.no_grad():
def test_map(model, val_loader, args):
    model.eval()
    gen = tqdm(val_loader, leave=True)
    preds = []
    for i, batch in enumerate(gen, start=1):
        cur_preds = model.test_step(batch,i)
        for p in cur_preds.detach().cpu().numpy():
            preds.append(p)
    return preds


def inference(ckpt, map_file, legend, map_data_dir = "/projects/bbym/shared/data/cma/validation/", batch_size = 32, num_workers = 8, seed = 42, model = "Unet", project = "DARPA", edge = False, viz = False):
    # make args according to input values
    args = parser.parse_args()
    args.ckpt = ckpt
    args.map_file = map_file
    args.legend = legend
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.seed = seed
    args.map_data_dir = map_data_dir
    args.model = model
    args.project = project
    args.name = args.project + "_" + args.model
    args.superpixel = ""
    args.edge = edge

    pl.seed_everything(args.seed)
    print("cuda:", torch.cuda.is_available())
    # init model and data
    
    model_weights = args.ckpt
    if args.ckpt is None:
        ckpt_folder = "checkpoints/" + args.project + "_" + args.model + "_fold02_val"
        print(ckpt_folder)
        model_weights = sorted(glob.glob(ckpt_folder+"/*.ckpt"))[-1]
        args.ckpt = model_weights
    print("loading model weights file:" + model_weights)

    model = SegmentationModel.load_from_checkpoint(checkpoint_path = args.ckpt,args = args)
    val_dataset = eval_MAPData(data_path=args.map_data_dir,type="poly",range=(30000,36000),mapName = args.map_file, legend = args.legend, edge = args.edge, viz = viz)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,num_workers=args.num_workers)

    # pred loop and save res
    preds = test_map(model, val_loader, args)
    preds = np.array(preds)
    # np.save("res",preds)
    # preds = np.load("res.npy")
    print(preds.shape)

    # cal metrics
    thres = [0.1, 0.15, 0.2, 0.25]
    # thres = [0.5, 0.7]
    
    metrics = val_dataset.metrics(preds, thres = thres)
    for i in range(len(metrics)):
        print("threshold: ", thres[i])
        print(np.median(metrics[i]),np.mean(metrics[i]))
    return metrics

if __name__ == '__main__':
    # inference(ckpt="./checkpoints/DARPA_Unet_fold02_val/jaccard_index_value=0.9229.ckpt", map_file="CA_AZ_Needles.tif", legend=None, map_data_dir = "/projects/bbym/shared/data/cma/validation/", batch_size = 32, num_workers = 8, seed = 42, model = "Unet", project = "DARPA")
    # inference(ckpt="./checkpoints/DARPA_Unet_fold02_val_vanilla/jaccard_index_value=0.9284.ckpt", map_file="AR_StJoe.tif", legend="Qayi_poly", map_data_dir = "/projects/bbym/shared/data/cma/validation/", batch_size = 32, num_workers = 8, seed = 42, model = "Unet", project = "DARPA")
    # inference(ckpt="./checkpoints/DARPA_Unet_fold02_val_vanilla/jaccard_index_value=0.9284.ckpt", map_file="OR_JosephineCounty.tif", legend=None, map_data_dir = "/projects/bbym/shared/data/cma/final_evaluation/", batch_size = 32, num_workers = 8, seed = 42, model = "Unet", project = "DARPA")
    inference(ckpt="./checkpoints/DARPA_Unet_fold02_val/jaccard_index_value=0.4806.ckpt", map_file="USGS_B-961_6.tif", legend=None, map_data_dir = "/projects/bbym/shared/data/cma/validation/", batch_size = 32, num_workers = 8, seed = 42, model = "Unet", project = "DARPA", edge = True, viz = True)
