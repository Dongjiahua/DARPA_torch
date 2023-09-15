from typing import Any
import lightning.pytorch as pl
import argparse
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch 
from model import DARPA_SEG, DARPA_DET
from data.dataset import MAPData
from data.dataset_det import DetData, collect_fn_det
from torch.utils.data import DataLoader
import torchmetrics
import os 
from tqdm import tqdm
from detectron2.engine import DefaultTrainer as d2Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str,default="/media/jiahua/FILE/uiuc/NCSA/processed/training", help='Root train data path')
    parser.add_argument('--val_data', type=str, default="/media/jiahua/FILE/uiuc/NCSA/processed/validation",  help='Root val data path')
    parser.add_argument('--out_dir', type=str, default="heat_out", help='output_dir')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=10, help='Total training epochs.')
    parser.add_argument('--pretrained', action='store_true', help='Whether use pretrained model.')
    parser.add_argument('--freeze', action='store_true',  help='Whether freeze layers.')

    return parser.parse_args()

    
def train():
    args = parse_args()
    torch.manual_seed(0)
    train_range = (0,2000)
    val_range = None
    mod = "det"
    if args.out_dir!="":
        os.makedirs(args.out_dir,exist_ok=True)
    # model = DARPA_SEG(args)
    
    # model = DARPA.load_from_checkpoint("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/exp/lightning_logs/version_1/checkpoints/epoch=3-step=3136.ckpt",args=args)
    # model = DARPA.load_from_checkpoint("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/lightning_logs/res_64/checkpoints/epoch=20-step=16464.ckpt")
    # train_dataset = MAPData(data_path=args.train_data,type="point")
    
    # val_dataset = MAPData(data_path=args.val_data,type="point")
    if mod=="seg":
        model = DARPA_SEG(args)
        train_dataset = MAPData(data_path=args.train_data,type="point",range = train_range)
        val_dataset = MAPData(data_path=args.val_data,type="point",range = val_range)
    else:
        model = DARPA_DET(args)
        train_dataset = DetData(data_path=args.train_data,type="point",range = train_range)
        val_dataset = DetData(data_path=args.val_data,type="point",range = val_range)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collect_fn_det)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,num_workers=args.workers, collate_fn=collect_fn_det)
    trainer = pl.Trainer(devices=1, max_epochs=args.epochs,precision=32,check_val_every_n_epoch=1,default_root_dir="./exp/")
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    train()
    
    # model = build_model("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/config/fct.yaml")

        
        