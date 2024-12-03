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

from datasets import MAPData, COCOData
# from .transforms import make_transform
from models import SegmentationModel
from test import test_main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_data_dir', type=str, default='/projects/bbym/shared/all_patched_data/training')
    parser.add_argument('--test_data_dir', type=str, default='/projects/bbym/shared/all_patched_data/validation')
    parser.add_argument('--map_data_dir', type=str, default='/projects/bbym/shared/data/cma/validation/')

    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--project', type=str, default='DARPA_Rail_own')
    parser.add_argument('--name', type=str, default='Unet')
    parser.add_argument('--model', type=str, default='Unet')
    parser.add_argument('--encoder', type=str, default='None')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--kfold', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
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
    parser.add_argument('--hist', default=False, action='store_true')
    parser.add_argument('--line', default=False, action='store_true')
    parser.add_argument('--fuse', default=False, action='store_true')
    parser.add_argument('--paint', default=False, action='store_true')
    # parser.add_argument('--superpixel', type=str, default='/u/xiyuez2/xiyuez2/Darpa_Unet/models/SpixelNet_bsd_ckpt.tar')
    parser.add_argument('--superpixel', type=str, default='')
    parser.add_argument('--heatmap', type=str, default='', choices=['', "L2"])
    parser.add_argument('--color', type=str, default="rgb", choices=["rgb", "lab"])
    parser.add_argument('--encode_boundary',  default=False, action='store_true')
    parser.add_argument('--pretrain', default=False, action='store_true')
    parser.add_argument('--pw', type=str, default=None)
    args = parser.parse_args()

    print("heat map:", args.heatmap)
    args.name = args.project + "_" + args.model
    pl.seed_everything(args.seed)
    print("cuda:", torch.cuda.is_available())
    for idx in [1]:
        wandb_logger = WandbLogger(project=args.project, group=args.name, name=f'{args.name}_fold{idx + 1:02d}')
        checkpoint_callback = ModelCheckpoint(
            monitor="val/jaccard_index_value",
            dirpath="checkpoints",
            filename=f"{args.name}_fold{idx + 1:02d}_"  +"{val/jaccard_index_value:.4f}",
            save_top_k=3,
            mode="max",
            save_last=True,
            # save_weights_only=True
        )
        early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=50, verbose=True,
                                            mode="min") 

        model = SegmentationModel(args)
    
        if args.pretrain:
            train_dataset = COCOData("/u/jiahuad2/data/COCO/train2017", "/u/jiahuad2/data/COCO/annotations/instances_train2017.json")
            val_dataset = COCOData("/u/jiahuad2/data/COCO/val2017", "/u/jiahuad2/data/COCO/annotations/instances_val2017.json")
        else:
        # train_dataset = MAPData(data_path="/projects/bbym/shared/all_patched_data/training",type="poly",range=(0,10000), filp_rate = args.filp_rate, color_jitter_rate = args.color_jitter_rate, edge=args.edge,heatmap = args.heatmap)
        # val_dataset = MAPData(data_path="/projects/bbym/shared/all_patched_data/validation",type="poly",range=(0, 2000), edge=args.edge, train=False, heatmap = args.heatmap)
            train_dataset = MAPData(data_path="/u/jiahuad2/data/COCO/train2017",type="poly",range=None, filp_rate = args.filp_rate, color_jitter_rate = args.color_jitter_rate, edge=args.edge,heatmap = args.heatmap, color = args.color, hist = args.hist, fuse=args.fuse, args=args)
            val_dataset = MAPData(data_path="/projects/bbym/shared/all_patched_data/validation",type="poly",range=None, edge=args.edge, train=False, heatmap = args.heatmap, color = args.color, hist = args.hist, fuse=args.fuse, args=args)

            if args.pw is not None:
                weight = torch.load(args.pw)
                target_model = model.model
                target_model_dict = target_model.state_dict()
                filtered_dict = {}
                for k in target_model_dict.keys():
                    if k not in weight.keys():
                        print("key not found: ", k)
                    # check if the shape is the same
                    elif target_model_dict[k].shape != weight[k].shape:
                        print("shape not match: ", k, target_model_dict[k].shape, weight[k].shape)
                    else:
                        filtered_dict[k] = weight[k]
                target_model.load_state_dict(filtered_dict, strict=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,num_workers=args.num_workers)

        trainer = pl.Trainer(accelerator='gpu',
                             devices=args.gpus,
                             precision=args.precision,
                             max_epochs=args.epochs,
                             log_every_n_steps=1,
                             # strategy='ddp',
                             # num_sanity_val_steps=0,
                             # limit_train_batches=5,
                             # limit_val_batches=1,
                             logger=wandb_logger,
                             callbacks=[checkpoint_callback, early_stop_callback])

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # trainer.test(dataloaders=test_dataloader)
        wandb.finish()
    
    # ## test files
    # args.ckpt = None #"./checkpoints/DARPA_Unet_fold02_val/jaccard_index_value=0.9229.ckpt"
    # mapnames = glob.glob(args.map_data_dir + "/*.tif")
    # f1_scores = []
    # for map_file in mapnames:
    #     # args.map_file = map_file[map_file.rfind("/") + 1:]
    #     # # test for one legend
    #     # test_main(args)
    #     try:
    #         args.map_file = map_file[map_file.rfind("/") + 1:]
    #         # test for one legend
    #         f1 = test_main(args)
    #         f1_scores += f1
    #     except Exception as e:
    #         print("error when processing " + args.map_file + ": ")
    #         print(e)

    # print(sum(f1_scores) / len(f1_scores))
    # print(f1_scores)
