from .backbones import model_dict
import lightning.pytorch as pl
import torchmetrics
import torch 
import os 
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import EventStorage
from utils.metrics import KeypointAPMetrics,Keypoint,DetectedKeypoint
from utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from typing import Callable, Dict, List, Tuple
from utils.vis import visualize_pred_kpts,  visualize_pred, predict_final_map
from utils.extract import update_channel_ap_metrics
from einops import rearrange, repeat

class DARPA_DET(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.model = model_dict[args.model](args)

        self.loss_total = 0
        self.count = 0
        self.val_loss_total = 0
        self.val_count = 0
        ap_thres = [5,10,20,40]
        # self.model = MAP_UNet(n_channels=6,n_classes=1,pretrained=args.pretrained,freeze=args.freeze)
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.train_f1 = torchmetrics.F1Score(task = "binary")
        self.val_f1 = torchmetrics.F1Score(task = "binary")
        self.criterion = torch.nn.BCELoss()
        self.ap_metrics = KeypointAPMetrics(keypoint_threshold_distances=ap_thres)
        self.val_ap_metrics = KeypointAPMetrics(keypoint_threshold_distances=ap_thres)
        self.max_keypoints = 20
        self.minimal_keypoint_pixel_distance = 5
    
    @torch.no_grad()
    def pre_process_batch(self, batch):
        x= batch['map_img']
        if self.args.patches>1:

            assert x.shape[2]%self.args.patches==0 & x.shape[3]%self.args.patches==0
            assert x.shape[2] == batch["metadata"]["img_size"][0] & x.shape[3] == batch["metadata"]["img_size"][1]
            x = rearrange(x, "b c (p1 h) (p2 w) -> (b p1 p2) c h w", p1=self.args.patches, p2=self.args.patches)
            x = torch.nn.functional.interpolate(x,(self.args.input_size,self.args.input_size),mode="bilinear")
            if "seg_img" in batch.keys():
                batch["seg_img"] = rearrange(batch['seg_img'], "b c (p1 h) (p2 w) -> (b p1 p2) c h w", p1=self.args.patches, p2=self.args.patches)
            legend_img = torch.nn.functional.interpolate(batch['legend_img'],(self.args.input_size,self.args.input_size),mode="bilinear")
            # print(legend_img.shape)
            batch['legend_img'] = legend_img.unsqueeze(1).repeat(1,self.args.patches**2,1,1,1).reshape(-1,*legend_img.shape[-3:])
            # print(batch['legend_img'].shape)
        else:
            assert x.shape[2] == self.args.input_size & x.shape[3] == self.args.input_size
        batch['map_img'] = x
    
    def post_process_batch(self, out, batch):
        img_size = batch["metadata"]["img_size"]
        if self.args.patches==1:
            return out
        p1_size = img_size[0]//self.args.patches
        p2_size = img_size[1]//self.args.patches
        out = torch.nn.functional.interpolate(out,(p1_size,p2_size),mode="bilinear")
        out = rearrange(out, "(b p1 p2) c h w -> b c (p1 h) (p2 w)", p1=self.args.patches, p2=self.args.patches)
        # imgs = batch['map_img']
        # imgs = torch.nn.functional.interpolate(imgs,(p1_size,p2_size),mode="bilinear")
        # imgs = rearrange(imgs, "(b p1 p2) c h w -> b c (p1 h) (p2 w)", p1=self.args.patches, p2=self.args.patches)
        # batch['map_img'] = imgs
        return out
    
    def training_step(self, batch, batch_idx):
        batch['seg_img'] = batch['seg_img'].unsqueeze(1)
        self.pre_process_batch(batch)
        
        
        gt_keypoints = batch["keypoints"]
        with EventStorage() as storage:
            output = self.model(batch)
        # seg[seg<0.1]=-1
        loss = self.criterion(output, batch['seg_img'])
        with torch.no_grad():
            output = self.post_process_batch(output, batch)
            update_channel_ap_metrics(output[:,0,...], gt_keypoints, self.ap_metrics)
            if self.args.out_dir!="" and batch_idx%10==0:
                visualize_pred(batch, output, batch_idx, self.current_epoch, self.args.out_dir, postix = "train")
        return loss
    
    def on_train_epoch_end(self) -> None:

        print(f"train_ap_epoch: {self.ap_metrics.compute()}")
        self.ap_metrics.reset()
        self.loss_total = 0
        self.count = 0
        self.scheduler1.step()
    
    def on_validation_epoch_end(self) -> None:
        print(f"val_ap_epoch: {self.val_ap_metrics.compute()}")
        self.val_ap_metrics.reset()

        
    def validation_step(self, batch, batch_idx):
        batch['seg_img'] = batch['seg_img'].unsqueeze(1)
        batch["origin"] = batch["map_img"].clone().detach()
        with torch.no_grad():
            self.pre_process_batch(batch)

            output = self.model(batch)
            loss = self.criterion(output, batch['seg_img'])
            output = self.post_process_batch(output, batch)
            batch["map_img"] = batch["origin"]

        # pred = torch.sigmoid(output)
        gt_keypoints = batch["keypoints"]
        update_channel_ap_metrics(output[:,0,...], gt_keypoints, self.val_ap_metrics)
        if self.args.out_dir!="" and batch_idx%1==0:
            visualize_pred(batch, output, batch_idx, self.current_epoch, self.args.out_dir)
            # final_maps = predict_final_map(output)
            # visualize_pred(batch,final_maps.unsqueeze(1),batch_idx,self.current_epoch, self.args.out_dir)
        
        

        return loss

    def forward(self,batch):
        batch['map_img'], batch['legend_img'] = batch['map_img'].cuda(), batch['legend_img'].cuda()
        self.pre_process_batch(batch)
        output = self.model(batch)
        output = self.post_process_batch(output, batch)
        keypoints, scores = get_keypoints_from_heatmap_batch_maxpool(
                output, 20, 5, return_scores=True
            )
        if self.args.out_dir!="":
            if not hasattr(self,"count"):
                self.count = 0
            batch["seg_img"] = output
            visualize_pred(batch, output, self.count, self.current_epoch, self.args.out_dir)
            final_maps = predict_final_map(output)
            visualize_pred_kpts(batch,final_maps.unsqueeze(1),self.count,self.current_epoch, self.args.out_dir)
            self.count+=1
        return output, keypoints
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return optimizer