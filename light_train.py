from typing import Any
import lightning.pytorch as pl
import argparse
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch 
from model import  DARPA_DET, DARPA_SEG
from data.dataset import MAPData
from data.dataset_det import DetData, collect_fn_det
from data.dataset_balance import BalanceData
from data.test_data import TestData
from data.dataset_gen import GENData
from torch.utils.data import DataLoader
import torchmetrics
import os 
from tqdm import tqdm
# from detectron2.engine import DefaultTrainer as d2Trainer
ckpt = "/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/exp/lightning_logs/version_31/checkpoints/epoch=49-step=19600.ckpt"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str,default="../all_patched/all_patched_data/training", help='Root train data path')
    parser.add_argument('--val_data', type=str, default="../all_patched/all_patched_data/validation",  help='Root val data path')
    parser.add_argument('--type', type=str, default="point",  help='Root val data path')
    parser.add_argument('--fct_cfg', type=str, default="./config/fct.yaml", help='fct config')
    parser.add_argument('--out_dir', type=str, default="output", help='output_dir')
    parser.add_argument('--model', type=str, default="unet_cat", help='backbone model')
    parser.add_argument('--patches', type=int, default=1, help='Patch size.')
    parser.add_argument('--input_size', type=int, default=112, help='Patch size.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs.')
    parser.add_argument('--pretrained', action='store_true', help='Whether use pretrained model.')
    parser.add_argument('--freeze', action='store_true',  help='Whether freeze layers.')

    return parser.parse_args()

    
def train():
    args = parse_args()
    
    pl.seed_everything(42)
    train_range =  None
    val_range = None

    if args.out_dir!="":
        os.makedirs(args.out_dir,exist_ok=True)
    # model = DARPA_SEG(args)
    
    # model = DARPA.load_from_checkpoint("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/exp/lightning_logs/version_1/checkpoints/epoch=3-step=3136.ckpt",args=args)
    # model = DARPA.load_from_checkpoint("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/lightning_logs/res_64/checkpoints/epoch=20-step=16464.ckpt")
    # train_dataset = MAPData(data_path=args.train_data,type="point")
    
    # val_dataset = MAPData(data_path=args.val_data,type="point")
    # symbol = ["horiz_bedding_pt.png","horizon_bedding_pt.png", "bedding_horiz_pt.png"]
    symbol = None
    # symbol = ["divide_pt.png"]
    # model = DARPA_DET(args)
    # model = DARPA_SEG(args)
    # model = DARPA_SEG.load_from_checkpoint("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/exp/lightning_logs/version_0/checkpoints/epoch=1-step=10250.ckpt", args)
    model = DARPA_DET.load_from_checkpoint("exp/lightning_logs/val_cat/checkpoints/epoch=14-step=11760.ckpt",args=args)
    # train_dataset1 = DetData(data_path=args.train_data,type=args.type,data_range = train_range, args=args,size = 10000, phase="train")
    train_dataset1 = BalanceData(data_path=args.train_data,type=args.type,data_range = train_range, args=args,size = 10000, phase="train",zero_ratio=0, paste_ratio=0,end=symbol)
    # val_dataset = GENData(data_path=args.train_data,type="point", args=args)
    # train_dataset2 = GENData(data_path=args.train_data,type="point", args=args, size=10000, end=symbol)
    train_dataset = train_dataset1
    # val_dataset = train_dataset2
    # val_dataset = GENData(data_path=args.val_data,type="point", args=args)
    # val_dataset = BalanceData(data_path=args.val_data,type=args.type,data_range = val_range, args=args, phase="val", end = symbol)
    val_dataset = DetData(data_path=args.val_data,type=args.type,data_range = val_range, args=args, phase="val", end = symbol)
    # train_dataset[0]
    
    # exit(0)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collect_fn_det)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,num_workers=args.workers, collate_fn=collect_fn_det)
    trainer = pl.Trainer(devices=1, max_epochs=args.epochs,precision=32,check_val_every_n_epoch=1,default_root_dir="./exp/")
    # trainer.validate(model,val_loader)
    trainer.fit(model, train_loader, val_loader)
    



if __name__ == "__main__":
    train()
    
    # model = build_model("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/config/fct.yaml")

        
        