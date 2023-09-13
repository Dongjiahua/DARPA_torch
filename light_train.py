from typing import Any
import lightning.pytorch as pl
import argparse
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch 
from model.unet import MAP_UNet
from data.dataset import MAPData
from torch.utils.data import DataLoader
import torchmetrics
import os 
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str,default="/media/jiahua/FILE/uiuc/NCSA/processed/training", help='Root train data path')
    parser.add_argument('--val_data', type=str, default="/media/jiahua/FILE/uiuc/NCSA/processed/validation",  help='Root val data path')
    parser.add_argument('--out_dir', type=str, default="./output", help='output_dir')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs.')
    parser.add_argument('--pretrained', action='store_true', help='Whether use pretrained model.')
    parser.add_argument('--freeze', action='store_true',  help='Whether freeze layers.')

    return parser.parse_args()

class DARPA(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.model = MAP_UNet(n_channels=6,n_classes=1,pretrained=args.pretrained,freeze=args.freeze)
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.train_f1 = torchmetrics.F1Score(task = "binary")
        self.val_f1 = torchmetrics.F1Score(task = "binary")
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
    def training_step(self, batch, batch_idx):
        map, legend, seg = batch['map_img'], batch['legend_img'], batch['seg_img']
        model_input = torch.cat([map,legend],dim=1)
        output = self.model(model_input)
        
        output = torch.nn.functional.interpolate(output,size=seg.shape[-2:],mode="nearest")
        
        loss = self.criterion(output, seg)
        acc = self.train_acc(output, seg)
        f1 = self.train_f1(output, seg)
        self.log("train_f1",f1)
        self.log("train_loss", loss)
        self.log("acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        map, legend, seg = batch['map_img'], batch['legend_img'], batch['seg_img']
        model_input = torch.cat([map,legend],dim=1)
        output = self.model(model_input)
        
        output = torch.nn.functional.interpolate(output,size=seg.shape[-2:],mode="nearest")
        # image = torch.sigmoid(output[0]).squeeze().cpu().detach().numpy().permute(1,2,0)
        # import matplotlib.pyplot as plt
        # plt.imshow(image,cmap="gray")
        # plt.show()
        loss = self.criterion(output, seg)
        acc = self.val_acc(output, seg)
        f1 = self.val_f1(output, seg)
        self.log("val_f1",f1)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        


    def on_train_epoch_end(self) -> None:
        self.log('train_acc_epoch', self.train_acc.compute())
        self.log("train_f1_epoch",self.train_f1.compute())
        print(f"train_acc_epoch: {self.train_acc.compute()}, train_f1_epoch: {self.train_f1.compute()}")
        self.train_acc.reset()
        self.train_f1.reset()
    
    def on_validation_epoch_end(self) -> None:
        self.log('val_acc_epoch', self.val_acc.compute())
        self.log("val_f1_epoch",self.val_f1.compute())
        print(f"val_acc_epoch: {self.val_acc.compute()}, val_f1_epoch: {self.val_f1.compute()}")
        self.val_acc.reset()
        self.val_f1.reset()
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
def train():
    args = parse_args()
    torch.manual_seed(0)
    
    model = DARPA(args)
    model = DARPA.load_from_checkpoint("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/exp/lightning_logs/version_1/checkpoints/epoch=3-step=3136.ckpt",args=args)
    # model = DARPA.load_from_checkpoint("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/lightning_logs/res_64/checkpoints/epoch=20-step=16464.ckpt")
    train_dataset = MAPData(data_path=args.train_data,type="point")
    
    val_dataset = MAPData(data_path=args.val_data,type="point")

    # train_dataset = MAPData(data_path=args.train_data,type="point",range=(0,20000))
    # val_dataset = MAPData(data_path=args.val_data,type="point",range=(-2000,-1))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,num_workers=args.workers)
    
    trainer = pl.Trainer(devices=1, max_epochs=args.epochs,precision=32,check_val_every_n_epoch=1,default_root_dir="./exp/")
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    train()

        
        